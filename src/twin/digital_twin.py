"""Software digital twin of a turbofan engine fleet.

EngineTwin   — virtual state of one engine: ingests sensor readings cycle by cycle, estimates RUL and a
               health index, tracks the health trend and monitors its inputs for drift.
FleetTwin    — hosts all engine twins, the active model version, delayed-label error monitoring (ADWIN)
               and the retraining trigger policy; logs every event.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import keras
import numpy as np
import pandas as pd

from src.data.preprocess import CMAPSSPreprocessor
from src.drift.detectors import ADWIN, PageHinkley
from src.drift.monitor import ExpectedSensorModel


def health_state(health_index: float) -> str:
    if health_index > 0.6:
        return "Healthy"
    if health_index > 0.25:
        return "Degrading"
    return "Critical"


@dataclass
class EngineTwin:
    unit: int
    model: keras.Model
    prep: CMAPSSPreprocessor
    esm: ExpectedSensorModel
    ph_ref_mean: float
    ph_delta: float
    ph_threshold: float
    model_version: str = "v1"
    raw: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    drift_cycle: int | None = None

    def __post_init__(self):
        self.ph = PageHinkley(delta=self.ph_delta, threshold=self.ph_threshold, reference_mean=self.ph_ref_mean)

    def ingest(self, reading: dict) -> dict:
        """Online update with one cycle of sensor data (a dict with unit, cycle, settings, s1..s21)."""
        self.raw.append(reading)
        df = pd.DataFrame(self.raw)
        tdf = self.prep.transform(df)
        X, _, _ = self.prep.make_last_windows(tdf)
        rul = float(np.clip(self.model.predict(X, verbose=0).ravel()[0], 0, None))
        return self._update_state(int(reading["cycle"]), rul, tdf.iloc[[-1]])

    def replay(self, engine_df: pd.DataFrame) -> pd.DataFrame:
        """Replay a recorded trajectory. Vectorised but causally identical to calling `ingest` per cycle:
        smoothing is causal and every window only contains past cycles."""
        tdf = self.prep.transform(engine_df)
        X, _, _ = self.prep.make_windows(tdf.assign(RUL=0.0))
        preds = np.clip(self.model.predict(X, verbose=0, batch_size=1024).ravel(), 0, None)
        n, w = len(tdf), self.prep.window
        # cycles before a full window exists use the first full-window prediction (padding semantics)
        rul = np.concatenate([np.full(max(0, n - len(preds)), preds[0]), preds])[-n:]
        scores = self.esm.score(tdf, rul)
        self.raw.extend(engine_df.to_dict("records"))
        for c, r, s in zip(tdf.cycle.to_numpy(), rul, scores):
            self._update_state(int(c), float(r), score=float(s))
        return self.history_frame()

    def _update_state(self, cycle: int, rul: float, feat_row: pd.DataFrame | None = None,
                      score: float | None = None) -> dict:
        if score is None:
            score = float(self.esm.score(feat_row, np.array([rul]))[0])
        alarm = self.ph.update(score)
        if alarm and self.drift_cycle is None:
            self.drift_cycle = cycle
        hi = min(rul / self.prep.rul_cap, 1.0)
        state = {"unit": self.unit, "cycle": cycle, "predicted_rul": rul, "health_index": hi,
                 "health_state": health_state(hi), "drift_score": score, "ph_statistic": self.ph.statistic,
                 "drift_alarm": self.drift_cycle is not None, "model_version": self.model_version}
        self.history.append(state)
        return state

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


@dataclass
class FleetTwin:
    model: keras.Model
    model_version: str
    prep: CMAPSSPreprocessor
    esm: ExpectedSensorModel
    ph_ref_mean: float
    ph_delta: float
    ph_threshold: float
    engines_for_trigger: int = 3
    adwin_delta: float = 0.002
    alarm_window: int = 10       # only alarms among the last N engines count toward the trigger
    events: list[dict] = field(default_factory=list)
    alarmed_engines: list[int] = field(default_factory=list)
    twins: dict[str, EngineTwin] = field(default_factory=dict)
    step: int = 0

    def __post_init__(self):
        self.error_monitor = ADWIN(delta=self.adwin_delta, min_window=5)
        self.error_drift = False

    def log(self, kind: str, **kw) -> None:
        self.events.append({"step": self.step, "event": kind, "model_version": self.model_version, **kw})

    def process_engine(self, key: str, unit: int, engine_df: pd.DataFrame, true_rul: np.ndarray | None = None,
                       injected_at: int | None = None) -> pd.DataFrame:
        """Replay one engine through its twin. If true RUL becomes available (run-to-failure/after
        maintenance), the per-cycle absolute errors are fed to ADWIN as delayed labels."""
        self.step += 1
        twin = EngineTwin(unit, self.model, self.prep, self.esm, self.ph_ref_mean, self.ph_delta, self.ph_threshold,
                          self.model_version)
        hist = twin.replay(engine_df)
        hist["engine_key"] = key
        if twin.drift_cycle is not None:
            delay = None if injected_at is None else twin.drift_cycle - injected_at
            self.alarmed_engines.append(self.step)
            self.log("input_drift_alarm", engine=key, alarm_cycle=twin.drift_cycle, injected_at=injected_at,
                     detection_delay_cycles=delay)
        if true_rul is not None:
            # Delayed labels: once the engine's true RUL is known, its trajectory RMSE enters the fleet-level
            # error stream. ADWIN flags a significant change in the fleet's mean error.
            hist["true_rul"] = true_rul
            hist["abs_error"] = np.abs(hist["predicted_rul"] - true_rul)
            engine_rmse = float(np.sqrt(np.mean(hist["abs_error"] ** 2)))
            if self.error_monitor.update(engine_rmse):
                self.error_drift = True
                self.log("error_drift_alarm_adwin", engine=key, engine_rmse=engine_rmse,
                         adwin_mean_rmse=self.error_monitor.estimation)
        self.twins[key] = twin
        return hist

    def retrain_needed(self) -> str | None:
        """Trigger policy: input-drift alarms on >= k engines, or an ADWIN change in fleet error."""
        recent = [s for s in self.alarmed_engines if s > self.step - self.alarm_window]
        if len(recent) >= self.engines_for_trigger:
            return f"input drift alarms on {len(recent)} of the last {self.alarm_window} engines"
        if self.error_drift:
            return "ADWIN detected a change in fleet prediction error"
        return None

    def deploy(self, model: keras.Model, version: str, esm: ExpectedSensorModel, ph_ref_mean: float,
               ph_threshold: float) -> None:
        self.log("model_deployed", new_version=version)
        self.model, self.model_version, self.esm = model, version, esm
        self.ph_ref_mean, self.ph_threshold = ph_ref_mean, ph_threshold
        self.alarmed_engines = []
        self.error_monitor = ADWIN(delta=self.adwin_delta, min_window=5)
        self.error_drift = False
