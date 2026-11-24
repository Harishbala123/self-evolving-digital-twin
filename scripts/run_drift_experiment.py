"""Concept-drift experiments: detection delay, false alarms, and before/after retraining comparison.

Scenario A: simulated sensor calibration bias (step and ramp) injected at a fixed cycle in FD001 test engines.
Scenario B: real fault-mode change. FD003 engines (HPC + fan degradation) are fed to the FD001 model.

Usage: python scripts/run_drift_experiment.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.preprocess import CMAPSSPreprocessor  # noqa: E402
from src.drift.experiments import (engine_scores, eval_last, load_subset, offset_fn, retrain_on,  # noqa: E402
                                   tune_delta)
from src.drift.monitor import run_page_hinkley  # noqa: E402
from src.twin.registry import ModelRegistry  # noqa: E402
from src.utils.common import get_logger, load_config, load_json, project_path, save_json, set_seed  # noqa: E402
from src.utils.plotting import CRITICAL, INK_2, SERIES, plt, save  # noqa: E402

log = get_logger("drift_experiment")


def detection_table(scores: dict, mon, injected_at: int | None, label: str) -> pd.DataFrame:
    rows = []
    for u, s in scores.items():
        idx = run_page_hinkley(s, mon.ref_mean, mon.delta, mon.threshold)
        alarm_cycle = None if idx is None else idx + 1          # cycles are 1-based
        row = {"scenario": label, "unit": u, "n_cycles": len(s), "alarm_cycle": alarm_cycle}
        if injected_at is not None:
            row["injected_at"] = injected_at
            row["false_alarm_before_drift"] = alarm_cycle is not None and alarm_cycle < injected_at
            row["detected_after_drift"] = alarm_cycle is not None and alarm_cycle >= injected_at
            row["delay_cycles"] = alarm_cycle - injected_at if row["detected_after_drift"] else None
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    out = {"engines": int(len(df)), "alarm_rate": float(df.alarm_cycle.notna().mean())}
    if "delay_cycles" in df:
        d = df.delay_cycles.dropna().astype(float)
        out.update({"detection_rate": float(df.detected_after_drift.mean()),
                    "false_alarm_before_drift_rate": float(df.false_alarm_before_drift.mean()),
                    "delay_mean": float(d.mean()) if len(d) else None,
                    "delay_median": float(d.median()) if len(d) else None,
                    "delay_p25": float(d.quantile(.25)) if len(d) else None,
                    "delay_p75": float(d.quantile(.75)) if len(d) else None})
    else:
        a = df.alarm_cycle.dropna()
        out["alarm_cycle_median"] = float(a.median()) if len(a) else None
    return out


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    d = cfg["drift"]
    models_dir = project_path(cfg["paths"]["models_dir"])
    res = project_path(cfg["paths"]["results_dir"])
    prep = CMAPSSPreprocessor.from_dict(load_json(models_dir / "preprocessor.json"))
    v1 = keras.models.load_model(models_dir / "lstm_v1.keras")
    info = load_json(res / "metrics" / "training_info.json")

    fd1_tr, fd1_te = load_subset(cfg, cfg["data"]["subset"])
    fd3_tr, fd3_te = load_subset(cfg, cfg["data"]["drift_subset"])
    t0 = d["injection_cycle"]
    val_bias = offset_fn(cfg, fd1_tr, t0, "step")
    mon, tuning = tune_delta(v1, prep, fd1_tr, info["train_units"], info["val_units"], val_bias, t0)
    tuning.to_csv(res / "metrics" / "drift_delta_tuning_validation.csv", index=False)
    save_json(mon.__dict__ | {"esm": mon.esm.to_dict()}, models_dir / "monitor_v1.json")

    eligible = fd1_te.groupby("unit").cycle.max()
    eligible = eligible[eligible >= t0 + 30].index          # engines observed for >= 30 cycles after onset
    te_elig = fd1_te[fd1_te.unit.isin(eligible)]
    step_df = offset_fn(cfg, fd1_tr, t0, "step")(te_elig)
    ramp_df = offset_fn(cfg, fd1_tr, t0, "ramp")(te_elig)

    tables = [
        detection_table(engine_scores(v1, prep, mon.esm, fd1_te), mon, None, "FD001 test (clean)"),
        detection_table(engine_scores(v1, prep, mon.esm, step_df), mon, t0, "FD001 test + step bias"),
        detection_table(engine_scores(v1, prep, mon.esm, ramp_df), mon, t0, "FD001 test + ramp bias"),
        detection_table(engine_scores(v1, prep, mon.esm, fd3_te), mon, None, "FD003 test (new fault mode)"),
    ]
    det = pd.concat(tables, ignore_index=True)
    det.to_csv(res / "metrics" / "drift_detection_per_engine.csv", index=False)
    summary = {t.scenario.iloc[0]: summarise(t) for t in tables}
    summary["monitor"] = {"reference_score": mon.ref_mean, "ph_delta": mon.delta, "ph_threshold": mon.threshold,
                          "offset_sensors": d["offset_sensors"], "offset_sigma": d["offset_sigma"],
                          "injection_cycle": t0}

    # ---------------- retraining (A: post-bias fleet data, B: FD003 run-to-failure data)
    (models_dir / "registry.json").unlink(missing_ok=True)      # rebuild the registry for this run
    reg = ModelRegistry(models_dir / "registry.json")
    if not reg.data["versions"]:
        reg.register(models_dir / "lstm_v1.keras", "initial training on FD001", None,
                     {"fd001_test_rmse": eval_last(v1, prep, fd1_te)["rmse"]})
    bias_all = offset_fn(cfg, fd1_tr, 1, "step")
    v2, info_a = retrain_on(cfg, v1, prep, bias_all(fd1_tr), fd1_tr, cfg["seed"])
    v3, info_b = retrain_on(cfg, v1, prep, fd3_tr, fd1_tr, cfg["seed"])
    v2.save(models_dir / "lstm_v2_sensor_bias.keras")
    v3.save(models_dir / "lstm_v3_fd003.keras")

    biased_test = bias_all(fd1_te)
    comp = pd.DataFrame([
        {"scenario": "A: sensor bias", "eval_set": "FD001 test + bias", "model": "v1 (before)", **eval_last(v1, prep, biased_test)},
        {"scenario": "A: sensor bias", "eval_set": "FD001 test + bias", "model": "v2 (after)", **eval_last(v2, prep, biased_test)},
        {"scenario": "B: new fault mode", "eval_set": "FD003 test", "model": "v1 (before)", **eval_last(v1, prep, fd3_te)},
        {"scenario": "B: new fault mode", "eval_set": "FD003 test", "model": "v3 (after)", **eval_last(v3, prep, fd3_te)},
        {"scenario": "B: new fault mode", "eval_set": "FD001 test (forgetting check)", "model": "v1 (before)", **eval_last(v1, prep, fd1_te)},
        {"scenario": "B: new fault mode", "eval_set": "FD001 test (forgetting check)", "model": "v3 (after)", **eval_last(v3, prep, fd1_te)},
    ])
    comp.to_csv(res / "metrics" / "retraining_before_after.csv", index=False)
    reg.register(models_dir / "lstm_v2_sensor_bias.keras", "Scenario A: sensor-bias drift alarm", "v1",
                 {"biased_test_rmse": comp.rmse[1], "biased_test_rmse_before": comp.rmse[0], **info_a}, activate=False)
    reg.register(models_dir / "lstm_v3_fd003.keras", "Scenario B: new fault mode (FD003) drift alarm", "v1",
                 {"fd003_test_rmse": comp.rmse[3], "fd003_test_rmse_before": comp.rmse[2], **info_b}, activate=False)
    summary["retraining"] = comp.to_dict("records")
    save_json(summary, res / "metrics" / "drift_summary.json")

    # ---------------- graphs
    g = res / "graphs"
    u = int(step_df.unit.iloc[0])
    s_clean = engine_scores(v1, prep, mon.esm, te_elig[te_elig.unit == u])[u]
    s_step = engine_scores(v1, prep, mon.esm, step_df[step_df.unit == u])[u]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(1, len(s_clean) + 1), s_clean, label="Clean stream", color=SERIES[0])
    ax.plot(np.arange(1, len(s_step) + 1), s_step, label="Stream with sensor bias", color=SERIES[1])
    ax.axvline(t0, color=INK_2, ls="--", lw=1, label=f"Bias injected (cycle {t0})")
    a = run_page_hinkley(s_step, mon.ref_mean, mon.delta, mon.threshold)
    if a is not None:
        ax.axvline(a + 1, color=CRITICAL, lw=1.5, label=f"Page-Hinkley alarm (cycle {a + 1})")
    ax.set_xlabel("Operational cycle")
    ax.set_ylabel("Drift score (mean z²)")
    ax.set_title(f"Input-drift monitoring, test engine {u}")
    ax.legend()
    save(fig, g / "drift_score_example.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, lab in enumerate(["FD001 test + step bias", "FD001 test + ramp bias"]):
        dd = det[det.scenario == lab].delay_cycles.dropna().astype(float)
        ax.hist(dd, bins=np.arange(0, max(31, dd.max() + 2 if len(dd) else 31), 2), alpha=0.75, color=SERIES[i],
                label=f"{lab.split('+ ')[1]} (median {dd.median():.0f})", edgecolor="white")
    ax.set_xlabel("Detection delay after drift onset (cycles)")
    ax.set_ylabel("Engines")
    ax.set_title("Drift detection delay")
    ax.legend()
    save(fig, g / "drift_detection_delay.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    labels = ["A: FD001+bias", "B: FD003", "FD001 (forgetting)"]
    before, after = comp.rmse[[0, 2, 4]].to_numpy(), comp.rmse[[1, 3, 5]].to_numpy()
    x = np.arange(3)
    ax.bar(x - 0.2, before, 0.38, label="Before retraining (v1)", color=SERIES[1])
    ax.bar(x + 0.2, after, 0.38, label="After retraining (v2 / v3)", color=SERIES[0])
    for xi, b, a_ in zip(x, before, after):
        ax.text(xi - 0.2, b + 0.4, f"{b:.1f}", ha="center", fontsize=9)
        ax.text(xi + 0.2, a_ + 0.4, f"{a_:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Test RMSE (cycles)")
    ax.set_title("Model performance before vs after drift-triggered retraining")
    ax.legend()
    save(fig, g / "retraining_before_after.png")

    for k, v in summary.items():
        log.info("%s: %s", k, v)


if __name__ == "__main__":
    main()
