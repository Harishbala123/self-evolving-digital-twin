"""Train the base LSTM RUL model on a C-MAPSS subset."""
from __future__ import annotations

import time
from dataclasses import dataclass

import keras
import numpy as np
import pandas as pd

from src.data.preprocess import (CMAPSSPreprocessor, add_test_rul, add_train_rul, load_split, load_test_rul,
                                 split_units)
from src.models.lstm import build_from_config
from src.utils.common import get_logger, project_path, save_json, set_seed

log = get_logger(__name__)


@dataclass
class PreparedData:
    prep: CMAPSSPreprocessor
    train_df: pd.DataFrame       # transformed, with RUL
    test_df: pd.DataFrame        # transformed, with RUL
    train_units: np.ndarray
    val_units: np.ndarray


def prepare_data(cfg: dict, subset: str | None = None, prep: CMAPSSPreprocessor | None = None) -> PreparedData:
    """Load a subset, generate RUL targets, fit (or reuse) the preprocessor on TRAINING engines only."""
    raw = project_path(cfg["data"]["raw_dir"])
    subset = subset or cfg["data"]["subset"]
    pcfg = cfg["preprocessing"]
    train_raw = add_train_rul(load_split(raw, subset, "train"), pcfg["rul_cap"])
    test_raw = add_test_rul(load_split(raw, subset, "test"), load_test_rul(raw, subset), pcfg["rul_cap"])
    train_units, val_units = split_units(train_raw["unit"].to_numpy(), pcfg["val_fraction"], cfg["seed"])
    if prep is None:
        prep = CMAPSSPreprocessor(pcfg["rul_cap"], pcfg["window"], pcfg["min_std"], pcfg["smoothing_alpha"])
        prep.fit(train_raw[train_raw.unit.isin(train_units)])
        log.info("Selected %d features: %s", prep.n_features, prep.features)
    return PreparedData(prep, prep.transform(train_raw), prep.transform(test_raw), train_units, val_units)


def fit_model(model: keras.Model, X_tr, y_tr, X_val, y_val, epochs: int, batch_size: int, patience: int,
              checkpoint_path=None, verbose: int = 2) -> pd.DataFrame:
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_rmse", mode="min", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_rmse", mode="min", factor=0.5, patience=max(2, patience // 3),
                                          min_lr=1e-5),
    ]
    if checkpoint_path is not None:
        callbacks.append(keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_rmse", mode="min",
                                                         save_best_only=True))
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                     callbacks=callbacks, shuffle=True, verbose=verbose)
    return pd.DataFrame(hist.history)


def train(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    data = prepare_data(cfg)
    tr = data.train_df[data.train_df.unit.isin(data.train_units)]
    va = data.train_df[data.train_df.unit.isin(data.val_units)]
    X_tr, y_tr, _ = data.prep.make_windows(tr)
    X_val, y_val, _ = data.prep.make_windows(va)
    log.info("Train windows %s | val windows %s (engines %d/%d)", X_tr.shape, X_val.shape,
             len(data.train_units), len(data.val_units))

    models_dir = project_path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    model = build_from_config(cfg, data.prep.n_features)
    model.summary(print_fn=log.info)
    t0 = time.time()
    tcfg = cfg["training"]
    history = fit_model(model, X_tr, y_tr, X_val, y_val, tcfg["epochs"], tcfg["batch_size"], tcfg["patience"],
                        checkpoint_path=models_dir / "lstm_v1_best.keras")
    elapsed = time.time() - t0

    model.save(models_dir / "lstm_v1.keras")
    save_json(data.prep.to_dict(), models_dir / "preprocessor.json")
    results = project_path(cfg["paths"]["results_dir"])
    history.to_csv(results / "metrics" / "training_history.csv", index_label="epoch")
    info = {"epochs_run": len(history), "train_seconds": round(elapsed, 1),
            "best_val_rmse": float(history["val_rmse"].min()), "n_train_windows": int(len(X_tr)),
            "n_val_windows": int(len(X_val)), "features": data.prep.features,
            "train_units": data.train_units.tolist(), "val_units": data.val_units.tolist()}
    save_json(info, results / "metrics" / "training_info.json")
    log.info("Training finished in %.1fs, best val RMSE %.2f", elapsed, info["best_val_rmse"])
    return info
