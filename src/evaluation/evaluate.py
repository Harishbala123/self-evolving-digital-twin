"""Evaluate a trained RUL model: train / validation / test metrics, prediction CSVs and graphs."""
from __future__ import annotations

import keras
import numpy as np
import pandas as pd

from src.data.preprocess import CMAPSSPreprocessor
from src.evaluation.metrics import regression_report
from src.evaluation.plots import plot_engine_trajectories, plot_loss_curves, plot_pred_vs_actual, plot_residuals
from src.training.train import prepare_data
from src.utils.common import get_logger, load_json, project_path, save_json

log = get_logger(__name__)


def predict(model: keras.Model, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    return np.clip(model.predict(X, batch_size=batch_size, verbose=0).ravel(), 0, None)


def evaluate(cfg: dict, model_path=None) -> dict:
    models_dir = project_path(cfg["paths"]["models_dir"])
    res = project_path(cfg["paths"]["results_dir"])
    model = keras.models.load_model(model_path or models_dir / "lstm_v1.keras")
    prep = CMAPSSPreprocessor.from_dict(load_json(models_dir / "preprocessor.json"))
    data = prepare_data(cfg, prep=prep)

    tr = data.train_df[data.train_df.unit.isin(data.train_units)]
    va = data.train_df[data.train_df.unit.isin(data.val_units)]
    X_tr, y_tr, _ = prep.make_windows(tr)
    X_va, y_va, _ = prep.make_windows(va)
    X_te, y_te, u_te = prep.make_last_windows(data.test_df)
    X_all, y_all, u_all = prep.make_windows(data.test_df)

    p_tr, p_va, p_te, p_all = (predict(model, X) for X in (X_tr, X_va, X_te, X_all))
    metrics = {
        "train_all_windows": regression_report(y_tr, p_tr),
        "validation_all_windows": regression_report(y_va, p_va),
        "test_last_window": regression_report(y_te, p_te),      # standard C-MAPSS protocol (100 engines)
        "test_all_windows": regression_report(y_all, p_all),
    }
    save_json(metrics, res / "metrics" / "evaluation_metrics.json")
    rows = [{"split": k, **v} for k, v in metrics.items()]
    pd.DataFrame(rows).to_csv(res / "metrics" / "evaluation_metrics.csv", index=False)
    pd.DataFrame({"unit": u_te, "actual_rul": y_te, "predicted_rul": p_te, "error": p_te - y_te}).to_csv(
        res / "metrics" / "test_predictions.csv", index=False)

    # Per-cycle trajectories on the test engines
    cycles = data.test_df.groupby("unit")["cycle"].apply(lambda s: s.to_numpy()[prep.window - 1:] if len(s) >= prep.window
                                                         else s.to_numpy()[-1:])
    traj = pd.DataFrame({"unit": u_all, "RUL": y_all, "pred": p_all})
    traj["cycle"] = np.concatenate([cycles[u] for u in np.unique(u_all)])

    g = res / "graphs"
    hist_path = res / "metrics" / "training_history.csv"
    if hist_path.exists():
        plot_loss_curves(pd.read_csv(hist_path, index_col="epoch"), g / "loss_curves.png")
    plot_pred_vs_actual(y_te, p_te, g / "test_pred_vs_actual.png",
                        f"FD001 test set: RMSE {metrics['test_last_window']['rmse']:.2f}")
    plot_residuals(y_te, p_te, g / "test_residuals.png")
    plot_engine_trajectories(traj, g / "test_engine_trajectories.png", [24, 31, 34, 81])
    for k, v in metrics.items():
        log.info("%-24s RMSE %.2f  MAE %.2f  R2 %.3f  score %.1f", k, v["rmse"], v["mae"], v["r2"], v["nasa_score"])
    return metrics
