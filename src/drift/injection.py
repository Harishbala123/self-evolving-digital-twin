"""Simulated changes in incoming sensor distributions.

Two families of change are used in the experiments:
1. Sensor calibration drift (simulated): a bias added to selected sensors from a given cycle onwards,
   either as a step or as a linear ramp. Magnitude is expressed in training standard deviations.
2. Fault-mode change (real data): engines from C-MAPSS FD003 (HPC + fan degradation) arriving at a twin
   trained on FD001 (HPC degradation only). No synthetic change is applied for this scenario.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sensor_stds(train_df: pd.DataFrame, sensors: list[str]) -> dict[str, float]:
    return {s: float(train_df[s].std()) for s in sensors}


def inject_sensor_offset(df: pd.DataFrame, sensors: list[str], stds: dict[str, float], magnitude: float,
                         start_cycle: int, mode: str = "step", ramp_cycles: int = 20) -> pd.DataFrame:
    """Return a copy of a raw C-MAPSS frame with a bias of `magnitude` x std added to `sensors`
    for every cycle >= start_cycle. `mode='ramp'` grows the bias linearly over `ramp_cycles`."""
    out = df.copy()
    after = (out["cycle"] - start_cycle).to_numpy()
    if mode == "step":
        factor = (after >= 0).astype(float)
    elif mode == "ramp":
        factor = np.clip((after + 1) / ramp_cycles, 0, 1) * (after >= 0)
    else:
        raise ValueError(f"unknown mode {mode}")
    for s in sensors:
        out[s] = out[s] + factor * magnitude * stds[s]
    return out
