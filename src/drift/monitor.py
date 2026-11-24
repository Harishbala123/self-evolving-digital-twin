"""Unsupervised input-drift monitoring for the digital twin.

The twin keeps an *expected-sensor model*: for each selected (normalised, smoothed) sensor it learns
E[sensor | RUL] from healthy-to-failure training trajectories (cubic polynomial in RUL). At run time the
twin's own RUL estimate gives the expected sensor vector; the standardised residuals are aggregated into a
score  s_t = mean_j z_{t,j}^2.  Under the training distribution s_t is stationary; a sensor bias or a new
fault mode raises it. A Page-Hinkley test on s_t (per engine) raises the alarm. Its threshold is calibrated
on clean validation engines to a target false-alarm rate, never on the drifted data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.drift.detectors import PageHinkley


@dataclass
class ExpectedSensorModel:
    degree: int = 3
    rul_cap: float = 125.0
    coefs: dict[str, list[float]] = field(default_factory=dict)
    sigma: dict[str, float] = field(default_factory=dict)

    def fit(self, tdf: pd.DataFrame, features: list[str]) -> "ExpectedSensorModel":
        x = tdf["RUL"].to_numpy(float) / self.rul_cap
        for f in features:
            c = np.polyfit(x, tdf[f].to_numpy(float), self.degree)
            r = tdf[f].to_numpy(float) - np.polyval(c, x)
            self.coefs[f], self.sigma[f] = c.tolist(), float(r.std() or 1.0)
        return self

    def residual_z(self, feats: pd.DataFrame | np.ndarray, rul: np.ndarray) -> np.ndarray:
        x = np.asarray(rul, float) / self.rul_cap
        F = feats[list(self.coefs)].to_numpy(float) if isinstance(feats, pd.DataFrame) else np.asarray(feats, float)
        out = np.empty_like(F)
        for j, f in enumerate(self.coefs):
            out[:, j] = (F[:, j] - np.polyval(self.coefs[f], x)) / self.sigma[f]
        return out

    def score(self, feats, rul) -> np.ndarray:
        return np.mean(self.residual_z(feats, rul) ** 2, axis=1)

    def to_dict(self) -> dict:
        return {"degree": self.degree, "rul_cap": self.rul_cap, "coefs": self.coefs, "sigma": self.sigma}

    @classmethod
    def from_dict(cls, d: dict) -> "ExpectedSensorModel":
        return cls(**d)


def run_page_hinkley(scores: np.ndarray, reference_mean: float, delta: float, threshold: float,
                     min_instances: int = 5) -> int | None:
    """Index of the first alarm in a score sequence, or None."""
    ph = PageHinkley(delta=delta, threshold=threshold, min_instances=min_instances, reference_mean=reference_mean)
    for i, s in enumerate(scores):
        if ph.update(float(s)):
            return i
    return None


def calibrate_threshold(clean_scores: list[np.ndarray], reference_mean: float, delta: float,
                        target_false_alarm_rate: float = 0.05, grid=None) -> float:
    """Smallest PH threshold whose per-engine false-alarm rate on clean engines is <= target."""
    grid = grid if grid is not None else np.arange(2.0, 200.0, 1.0)
    for lam in grid:
        fa = np.mean([run_page_hinkley(s, reference_mean, delta, lam) is not None for s in clean_scores])
        if fa <= target_false_alarm_rate:
            return float(lam)
    return float(grid[-1])
