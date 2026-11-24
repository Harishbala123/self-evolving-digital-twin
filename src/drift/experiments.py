"""Reusable building blocks for the drift experiments and the fleet digital-twin demo."""
from __future__ import annotations

from dataclasses import dataclass

import keras
import numpy as np
import pandas as pd

from src.data.preprocess import (CMAPSSPreprocessor, add_test_rul, add_train_rul, load_split, load_test_rul,
                                 split_units)
from src.drift.injection import inject_sensor_offset, sensor_stds
from src.drift.monitor import ExpectedSensorModel, calibrate_threshold
from src.evaluation.metrics import regression_report
from src.twin.retrainer import fine_tune
from src.utils.common import get_logger, project_path

log = get_logger(__name__)


@dataclass
class Monitor:
    esm: ExpectedSensorModel
    ref_mean: float
    delta: float
    threshold: float


def predict_engine(model: keras.Model, prep: CMAPSSPreprocessor, raw_engine: pd.DataFrame) -> np.ndarray:
    """Per-cycle causal RUL predictions for one raw engine trajectory."""
    tdf = prep.transform(raw_engine)
    X, _, _ = prep.make_windows(tdf.assign(RUL=0.0))
    p = np.clip(model.predict(X, verbose=0, batch_size=1024).ravel(), 0, None)
    return np.concatenate([np.full(max(0, len(tdf) - len(p)), p[0]), p])[-len(tdf):]


def engine_scores(model, prep, esm, raw_df: pd.DataFrame) -> dict[int, np.ndarray]:
    out = {}
    for u, g in raw_df.groupby("unit"):
        rul = predict_engine(model, prep, g)
        out[u] = esm.score(prep.transform(g), rul)
    return out


def build_monitor(model, prep, raw_train: pd.DataFrame, fit_units, calib_units, delta: float,
                  target_fa: float = 0.05) -> Monitor:
    """Fit the expected-sensor model on `fit_units` (true RUL) and calibrate Page-Hinkley on clean
    `calib_units` (using the model's own RUL estimates, exactly as in deployment)."""
    tdf = prep.transform(raw_train)
    esm = ExpectedSensorModel(rul_cap=prep.rul_cap).fit(tdf[tdf.unit.isin(fit_units)], prep.features)
    calib = engine_scores(model, prep, esm, raw_train[raw_train.unit.isin(calib_units)])
    ref = float(np.mean(np.concatenate(list(calib.values()))))
    thr = calibrate_threshold(list(calib.values()), ref, delta, target_fa)
    log.info("Monitor: reference score %.3f, PH delta %.2f, calibrated threshold %.1f", ref, delta, thr)
    return Monitor(esm, ref, delta, thr)


def tune_delta(model, prep, raw_train, fit_units, calib_units, bias_fn, injected_at: int,
               deltas=(0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0), target_fa: float = 0.05,
               min_detection: float = 0.95) -> tuple[Monitor, pd.DataFrame]:
    """Select the Page-Hinkley tolerance delta on VALIDATION engines only: for each delta the threshold is
    calibrated to the false-alarm target on clean validation engines, then the median detection delay is measured
    on the same validation engines with simulated drift. The delta with the shortest delay (and >= 95 % detection)
    wins. Test engines are never used here."""
    from src.drift.monitor import run_page_hinkley
    tdf = prep.transform(raw_train)
    esm = ExpectedSensorModel(rul_cap=prep.rul_cap).fit(tdf[tdf.unit.isin(fit_units)], prep.features)
    val = raw_train[raw_train.unit.isin(calib_units)]
    clean = engine_scores(model, prep, esm, val)
    drifted = engine_scores(model, prep, esm, bias_fn(val))
    ref = float(np.mean(np.concatenate(list(clean.values()))))
    rows = []
    for dl in deltas:
        lam = calibrate_threshold(list(clean.values()), ref, dl, target_fa)
        alarms = [run_page_hinkley(s, ref, dl, lam) for s in drifted.values()]
        delays = [a + 1 - injected_at for a in alarms if a is not None and a + 1 >= injected_at]
        rows.append({"delta": dl, "threshold": lam, "val_detection_rate": len(delays) / len(alarms),
                     "val_median_delay": float(np.median(delays)) if delays else np.inf})
    table = pd.DataFrame(rows)
    ok = table[table.val_detection_rate >= min_detection]
    best = (ok if len(ok) else table).sort_values(["val_median_delay", "delta"]).iloc[0]
    log.info("Delta tuning on validation engines:\n%s\nselected delta=%.2f", table.to_string(index=False), best.delta)
    return Monitor(esm, ref, float(best.delta), float(best.threshold)), table


def load_subset(cfg: dict, subset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = project_path(cfg["data"]["raw_dir"])
    cap = cfg["preprocessing"]["rul_cap"]
    tr = add_train_rul(load_split(raw, subset, "train"), cap)
    te = add_test_rul(load_split(raw, subset, "test"), load_test_rul(raw, subset), cap)
    return tr, te


def offset_fn(cfg: dict, ref_train: pd.DataFrame, start_cycle: int, mode: str = "step"):
    d = cfg["drift"]
    stds = sensor_stds(ref_train, d["offset_sensors"])
    return lambda df: inject_sensor_offset(df, d["offset_sensors"], stds, d["offset_sigma"], start_cycle, mode)


def windows(prep, raw_df, last_only=False):
    tdf = prep.transform(raw_df)
    return prep.make_last_windows(tdf) if last_only else prep.make_windows(tdf)


def retrain_on(cfg, model, prep, new_train: pd.DataFrame, replay_train: pd.DataFrame, seed: int):
    """Fine-tune `model` on labelled post-drift engines (engine-level train/val split) + replay data."""
    tr_u, va_u = split_units(new_train.unit.to_numpy(), cfg["preprocessing"]["val_fraction"], seed)
    X_new, y_new, _ = windows(prep, new_train[new_train.unit.isin(tr_u)])
    X_val, y_val, _ = windows(prep, new_train[new_train.unit.isin(va_u)])
    X_rep, y_rep, _ = windows(prep, replay_train)
    old_val = regression_report(y_val, np.clip(model.predict(X_val, verbose=0, batch_size=1024).ravel(), 0, None))
    new_model, info = fine_tune(model, X_new, y_new, X_rep, y_rep, X_val, y_val, cfg["drift"]["finetune_epochs"],
                                seed=seed)
    info["old_val_rmse"] = old_val["rmse"]
    info["train_units"], info["val_units"] = tr_u.tolist(), va_u.tolist()
    return new_model, info


def eval_last(model, prep, raw_test) -> dict:
    X, y, _ = windows(prep, raw_test, last_only=True)
    return regression_report(y, np.clip(model.predict(X, verbose=0).ravel(), 0, None))
