"""Fleet digital-twin demonstration: the twin replays engines one after another, tracks their health,
detects drift, triggers retraining, validates and deploys new model versions, and logs every event.

Fleet timeline
  Phase 1  Normal operation: 30 FD001 test engines, no change.
  Phase 2  Sensor calibration drift: a bias appears on 4 sensors. Engines already in service when it appears
           get it from cycle `injection_cycle`; engines that enter service after the twin has adapted carry it
           from cycle 1.
  Phase 3  New fault mode: FD003 engines (fan + HPC degradation) start arriving.

Labelled post-drift data used for retraining comes from run-to-failure engines (C-MAPSS training engines)
under the new condition, i.e. what a maintenance database would accumulate. Test engines used in the replay
are never used for training.

Usage: python scripts/run_digital_twin_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.preprocess import CMAPSSPreprocessor  # noqa: E402
from src.drift.experiments import build_monitor, eval_last, load_subset, offset_fn, retrain_on  # noqa: E402
from src.twin.digital_twin import FleetTwin  # noqa: E402
from src.twin.retrainer import accept  # noqa: E402
from src.utils.common import get_logger, load_config, load_json, project_path, set_seed  # noqa: E402
from src.utils.plotting import CRITICAL, GOOD, INK_2, SERIES, WARNING, plt, save  # noqa: E402

log = get_logger("twin_demo")


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    d = cfg["drift"]
    models_dir = project_path(cfg["paths"]["models_dir"])
    out_models = models_dir / "twin_demo"
    out_models.mkdir(parents=True, exist_ok=True)
    res = project_path(cfg["paths"]["results_dir"])
    prep = CMAPSSPreprocessor.from_dict(load_json(models_dir / "preprocessor.json"))
    info = load_json(res / "metrics" / "training_info.json")
    v1 = keras.models.load_model(models_dir / "lstm_v1.keras")

    fd1_tr, fd1_te = load_subset(cfg, cfg["data"]["subset"])
    fd3_tr, fd3_te = load_subset(cfg, cfg["data"]["drift_subset"])
    bias_mid = offset_fn(cfg, fd1_tr, d["injection_cycle"], "step")
    bias_all = offset_fn(cfg, fd1_tr, 1, "step")

    # use the Page-Hinkley delta tuned on validation engines by run_drift_experiment.py (if it has been run)
    mon_file = models_dir / "monitor_v1.json"
    delta = load_json(mon_file)["delta"] if mon_file.exists() else d["page_hinkley_delta"]
    d["page_hinkley_delta"] = delta
    mon = build_monitor(v1, prep, fd1_tr, info["train_units"], info["val_units"], delta)
    fleet = FleetTwin(v1, "v1", prep, mon.esm, mon.ref_mean, mon.delta, mon.threshold, d["engines_for_trigger"],
                      d["adwin_delta"])
    fleet.log("model_deployed", new_version="v1")

    rng = np.random.default_rng(cfg["seed"])
    long_units = fd1_te.groupby("unit").cycle.max()
    long_units = long_units[long_units >= d["injection_cycle"] + 30].index.to_numpy().copy()
    rng.shuffle(long_units)
    phase1 = long_units[:30]
    phase2 = long_units[30:70]
    phase3 = rng.permutation(fd3_te.unit.unique())[:40]

    # Labelled data the maintenance database holds for each condition (used only when retraining).
    regimes = {"sensor_bias": (bias_all(fd1_tr), fd1_tr), "new_fault_mode": (fd3_tr, bias_all(fd1_tr))}
    rows, histories = [], []
    version_n = 1

    def run(phase: str, key: str, raw: pd.DataFrame, injected_at):
        nonlocal version_n
        h = fleet.process_engine(key, int(raw.unit.iloc[0]), raw.drop(columns="RUL"), raw.RUL.to_numpy(), injected_at)
        h["phase"] = phase
        histories.append(h)
        rows.append({"step": fleet.step, "phase": phase, "engine": key, "model_version": h.model_version.iloc[0],
                     "alarm_cycle": fleet.twins[key].drift_cycle, "injected_at": injected_at,
                     "last_cycle_true_rul": h.true_rul.iloc[-1], "last_cycle_pred_rul": h.predicted_rul.iloc[-1],
                     "trajectory_rmse": float(np.sqrt(np.mean((h.predicted_rul - h.true_rul) ** 2)))})
        reason = fleet.retrain_needed()
        regime = "sensor_bias" if phase.startswith("2") else "new_fault_mode" if phase.startswith("3") else None
        if reason and regime:
            fleet.log("retrain_triggered", reason=reason, regime=regime)
            new_train, replay = regimes[regime]
            cand, rinfo = retrain_on(cfg, fleet.model, prep, new_train, replay, cfg["seed"] + version_n)
            new_val = rinfo["best_val_rmse"]
            if accept(new_val, rinfo["old_val_rmse"]):
                version_n += 1
                ver = f"v{version_n}"
                cand.save(out_models / f"lstm_{ver}.keras")
                val_u = rinfo["val_units"]
                fit_u = rinfo["train_units"]
                new_mon = build_monitor(cand, prep, new_train, fit_u, val_u, d["page_hinkley_delta"])
                fleet.log("candidate_accepted", candidate=ver, val_rmse_old=rinfo["old_val_rmse"], val_rmse_new=new_val)
                fleet.deploy(cand, ver, new_mon.esm, new_mon.ref_mean, new_mon.threshold)
            else:
                fleet.log("retrain_skipped_candidate_rejected", val_rmse_old=rinfo["old_val_rmse"], val_rmse_new=new_val)
                fleet.alarmed_engines, fleet.error_drift = [], False   # require fresh evidence before retrying

    for u in phase1:
        run("1 normal", f"FD001-{u}", fd1_te[fd1_te.unit == u], None)
    for u in phase2:
        adapted = fleet.model_version != "v1"
        raw = (bias_all if adapted else bias_mid)(fd1_te[fd1_te.unit == u])
        run("2 sensor bias", f"FD001b-{u}", raw, 1 if adapted else d["injection_cycle"])
    for u in phase3:
        run("3 new fault mode", f"FD003-{u}", fd3_te[fd3_te.unit == u], None)

    summary = pd.DataFrame(rows)
    events = pd.DataFrame(fleet.events)
    hist = pd.concat(histories, ignore_index=True)
    summary.to_csv(res / "metrics" / "twin_fleet_summary.csv", index=False)
    events.to_csv(res / "reports" / "twin_events.csv", index=False)
    hist.to_csv(res / "reports" / "twin_engine_history.csv", index=False)

    perf = (summary.assign(sq=lambda x: (x.last_cycle_pred_rul - x.last_cycle_true_rul) ** 2)
            .groupby(["phase", "model_version"]).agg(engines=("engine", "count"), rmse_last_cycle=("sq", lambda s: float(np.sqrt(s.mean()))),
                                                      mean_trajectory_rmse=("trajectory_rmse", "mean")).reset_index())
    perf.to_csv(res / "metrics" / "twin_performance_by_phase_version.csv", index=False)
    log.info("\n%s", perf.to_string(index=False))
    log.info("\n%s", events.to_string(index=False))
    delays = summary.dropna(subset=["alarm_cycle", "injected_at"])
    delays = delays[(delays.phase == "2 sensor bias") & (delays.injected_at > 1)]
    if len(delays):
        log.info("Phase-2 detection delays (cycles): %s", (delays.alarm_cycle - delays.injected_at).tolist())

    # ------------------------------- graphs
    g = res / "graphs"
    fig, ax = plt.subplots(figsize=(12, 4.5))
    vers = sorted(summary.model_version.unique())
    for i, v in enumerate(vers):
        s = summary[summary.model_version == v]
        ax.plot(s.step, s.trajectory_rmse, "o", ms=6, color=SERIES[i], label=f"Model {v}")
    ax.set_ylim(-3, summary.trajectory_rmse.max() * 1.15)
    for _, e in events[events.event == "model_deployed"].iloc[1:].iterrows():
        ax.axvline(e.step + 0.5, color=GOOD, lw=1.5)
        ax.text(e.step + 0.8, ax.get_ylim()[1] * 0.94, f"deploy {e.new_version}", color=INK_2, fontsize=9)
    for _, e in events[events.event == "input_drift_alarm"].iterrows():
        ax.plot(e.step, 1, "v", color=CRITICAL, ms=7)
    for b in (len(phase1) + 0.5, len(phase1) + len(phase2) + 0.5):
        ax.axvline(b, color=INK_2, ls="--", lw=1)
    ax.plot([], [], "v", color=CRITICAL, label="Input-drift alarm")
    ax.set_xlabel("Fleet step (engine replayed)   |   Phase 1: normal  →  Phase 2: sensor bias  →  Phase 3: new fault mode")
    ax.set_ylabel("Trajectory RMSE (cycles)")
    ax.set_title("Self-evolving digital twin: drift alarms, retraining and model versions")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    save(fig, g / "twin_fleet_timeline.png")

    key = summary[(summary.phase == "2 sensor bias") & summary.alarm_cycle.notna() & (summary.injected_at > 1)].engine
    key = key.iloc[0] if len(key) else summary.engine.iloc[0]
    h = hist[hist.engine_key == key]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax = axes[0]
    ax.axhspan(0.6 * prep.rul_cap, prep.rul_cap * 1.15, color=GOOD, alpha=0.08)
    ax.axhspan(0.25 * prep.rul_cap, 0.6 * prep.rul_cap, color=WARNING, alpha=0.10)
    ax.axhspan(0, 0.25 * prep.rul_cap, color=CRITICAL, alpha=0.08)
    ax.plot(h.cycle, h.true_rul, color=SERIES[0], label="Actual RUL (capped)")
    ax.plot(h.cycle, h.predicted_rul, color=SERIES[1], label="Twin estimate")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"Engine twin {key}: health trend (bands: Healthy / Degrading / Critical)")
    ax.legend(loc="lower left")
    ax = axes[1]
    ax.plot(h.cycle, h.drift_score, color=SERIES[2], label="Drift score")
    ac = fleet.twins[key].drift_cycle
    ax.axvline(d["injection_cycle"], color=INK_2, ls="--", lw=1, label="Bias injected")
    if ac:
        ax.axvline(ac, color=CRITICAL, lw=1.5, label=f"Alarm (cycle {ac})")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Mean z²")
    ax.legend(loc="upper left")
    save(fig, g / "twin_engine_health_trend.png")

    final = {"FD001 test (clean)": eval_last(fleet.model, prep, fd1_te), "FD003 test": eval_last(fleet.model, prep, fd3_te)}
    log.info("Final active model %s: %s", fleet.model_version, final)


if __name__ == "__main__":
    main()
