"""Regression metrics for RUL prediction."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Asymmetric PHM08 scoring function: late predictions (d > 0) are penalised more than early ones."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "nasa_score": nasa_score(y_true, y_pred),
        "mean_error": float(np.mean(y_pred - y_true)),
        "n": int(len(y_true)),
    }
