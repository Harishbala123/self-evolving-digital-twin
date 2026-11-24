"""Concept-drift detectors implemented from their original papers (no black-box dependency).

* PageHinkley — sequential change detection for an increase in the mean
  (Page 1954; Hinkley 1971; formulation as in Gama et al., 2013).
* ADWIN — ADaptive WINdowing (Bifet & Gavaldà, 2007): keeps a variable-length window and drops
  its older part whenever two sub-windows have significantly different means.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PageHinkley:
    """Detects an upward shift in the mean of a stream.

    If `reference_mean` is given (e.g. calibrated on clean data), deviations are measured against it;
    otherwise against the running mean of the stream, as in the classic test.
    """

    delta: float = 0.5          # tolerated magnitude of change
    threshold: float = 25.0     # lambda: alarm when cumulative excess exceeds this
    min_instances: int = 5
    reference_mean: float | None = None
    n: int = 0
    mean: float = 0.0
    cum: float = 0.0
    cum_min: float = 0.0
    drift_detected: bool = False

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        ref = self.mean if self.reference_mean is None else self.reference_mean
        self.cum += x - ref - self.delta
        self.cum_min = min(self.cum_min, self.cum)
        self.drift_detected = self.n >= self.min_instances and (self.cum - self.cum_min) > self.threshold
        return self.drift_detected

    @property
    def statistic(self) -> float:
        return self.cum - self.cum_min

    def reset(self) -> None:
        self.n, self.mean, self.cum, self.cum_min, self.drift_detected = 0, 0.0, 0.0, 0.0, False


@dataclass
class ADWIN:
    """ADWIN with the variance-aware bound of ADWIN2.

    For simplicity the window is stored explicitly (bounded by `max_window`) instead of in exponential
    histograms; every cut point is checked with prefix sums, which is O(window) per check.
    """

    delta: float = 0.002
    max_window: int = 5000
    min_window: int = 10
    clock: int = 1               # check for change every `clock` insertions
    window: deque = field(default_factory=deque)
    drift_detected: bool = False
    n_detections: int = 0
    _t: int = 0

    def update(self, x: float) -> bool:
        self.window.append(float(x))
        if len(self.window) > self.max_window:
            self.window.popleft()
        self._t += 1
        self.drift_detected = False
        if self._t % self.clock == 0 and len(self.window) >= 2 * self.min_window:
            while self._detect_and_cut():
                self.drift_detected = True
        if self.drift_detected:
            self.n_detections += 1
        return self.drift_detected

    def _detect_and_cut(self) -> bool:
        w = np.fromiter(self.window, float)
        n = len(w)
        if n < 2 * self.min_window:
            return False
        var = float(np.var(w))
        cs = np.cumsum(w)
        total = cs[-1]
        log_term = math.log(2.0 / (self.delta / math.log(max(n, 2))))
        i = np.arange(self.min_window, n - self.min_window + 1)      # candidate cut points (vectorised)
        n0, n1 = i.astype(float), (n - i).astype(float)
        mu0, mu1 = cs[i - 1] / n0, (total - cs[i - 1]) / n1
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        eps = np.sqrt(2.0 / m * var * log_term) + 2.0 / (3.0 * m) * log_term
        hits = np.nonzero(np.abs(mu0 - mu1) > eps)[0]
        if hits.size == 0:
            return False
        cut = int(i[hits[0]])
        for _ in range(cut):              # drop the older sub-window
            self.window.popleft()
        return True

    @property
    def width(self) -> int:
        return len(self.window)

    @property
    def estimation(self) -> float:
        return float(np.mean(self.window)) if self.window else 0.0
