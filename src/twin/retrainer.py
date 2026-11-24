"""Adaptive retraining: warm-start fine-tuning on post-drift data plus a replay buffer of reference data
(to limit catastrophic forgetting), with an acceptance check before the new version is deployed."""
from __future__ import annotations

import keras
import numpy as np

from src.training.train import fit_model
from src.utils.common import get_logger

log = get_logger(__name__)


def fine_tune(base: keras.Model, X_new, y_new, X_replay, y_replay, X_val, y_val, epochs: int = 25,
              replay_fraction: float = 0.3, learning_rate: float = 5e-4, batch_size: int = 256,
              seed: int = 42) -> tuple[keras.Model, dict]:
    rng = np.random.default_rng(seed)
    n_rep = min(len(X_replay), int(len(X_new) * replay_fraction / (1 - replay_fraction)))
    idx = rng.choice(len(X_replay), n_rep, replace=False)
    X = np.concatenate([X_new, X_replay[idx]])
    y = np.concatenate([y_new, y_replay[idx]])
    model = keras.models.clone_model(base)
    model.set_weights(base.get_weights())
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse",
                  metrics=[keras.metrics.RootMeanSquaredError(name="rmse"), keras.metrics.MeanAbsoluteError(name="mae")])
    log.info("Fine-tuning on %d new + %d replay windows", len(X_new), n_rep)
    hist = fit_model(model, X, y, X_val, y_val, epochs, batch_size, patience=6, verbose=0)
    return model, {"epochs_run": len(hist), "best_val_rmse": float(hist["val_rmse"].min()),
                   "n_new": int(len(X_new)), "n_replay": int(n_rep)}


def accept(new_rmse: float, old_rmse: float, min_improvement: float = 0.02) -> bool:
    """Deploy the candidate only if it improves post-drift validation RMSE by at least 2 %."""
    return new_rmse < old_rmse * (1 - min_improvement)
