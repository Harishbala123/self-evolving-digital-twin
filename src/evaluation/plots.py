"""Graphs for RUL model evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.plotting import INK_2, SERIES, plt, save


def plot_loss_curves(history: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in [(axes[0], "loss", "MSE loss"), (axes[1], "rmse", "RMSE (cycles)")]:
        ax.plot(history.index + 1, history[key], label="Training", color=SERIES[0])
        ax.plot(history.index + 1, history[f"val_{key}"], label="Validation", color=SERIES[1])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend()
    axes[0].set_title("Training vs validation loss")
    axes[1].set_title("Training vs validation RMSE")
    return save(fig, path)


def plot_pred_vs_actual(y_true, y_pred, path: Path, title: str) -> Path:
    order = np.argsort(y_true)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(np.arange(len(order)), np.asarray(y_true)[order], label="Actual RUL", color=SERIES[0])
    ax.plot(np.arange(len(order)), np.asarray(y_pred)[order], "o", ms=4, label="Predicted RUL", color=SERIES[1])
    ax.set_xlabel("Test engines (sorted by actual RUL)")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(title)
    ax.legend()
    ax = axes[1]
    lim = [0, max(np.max(y_true), np.max(y_pred)) * 1.05]
    ax.plot(lim, lim, "--", color=INK_2, lw=1, label="Perfect prediction")
    ax.scatter(y_true, y_pred, s=18, color=SERIES[0], alpha=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("Predicted vs actual")
    ax.legend()
    return save(fig, path)


def plot_residuals(y_true, y_pred, path: Path) -> Path:
    res = np.asarray(y_pred) - np.asarray(y_true)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(res, bins=25, color=SERIES[0], edgecolor="white")
    axes[0].axvline(0, color=INK_2, lw=1, ls="--")
    axes[0].set_xlabel("Residual = predicted − actual (cycles)")
    axes[0].set_ylabel("Engines")
    axes[0].set_title(f"Residual distribution (mean {res.mean():+.2f}, std {res.std():.2f})")
    axes[1].scatter(y_true, res, s=18, color=SERIES[0])
    axes[1].axhline(0, color=INK_2, lw=1, ls="--")
    axes[1].set_xlabel("Actual RUL")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual vs actual RUL (positive = late, riskier)")
    return save(fig, path)


def plot_engine_trajectories(traj: pd.DataFrame, path: Path, units: list[int]) -> Path:
    """traj columns: unit, cycle, RUL, pred."""
    n = len(units)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6), sharey=True)
    for ax, u in zip(np.atleast_1d(axes), units):
        g = traj[traj.unit == u]
        ax.plot(g.cycle, g.RUL, label="Actual (capped)", color=SERIES[0])
        ax.plot(g.cycle, g.pred, label="Predicted", color=SERIES[1])
        ax.set_title(f"Test engine {u}")
        ax.set_xlabel("Cycle")
    np.atleast_1d(axes)[0].set_ylabel("RUL (cycles)")
    np.atleast_1d(axes)[0].legend()
    return save(fig, path)
