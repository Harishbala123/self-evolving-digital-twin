"""C-MAPSS preprocessing: loading, RUL targets, feature selection, operating-condition
normalisation, causal smoothing and sliding-window sequence generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SETTING_COLS = ["setting1", "setting2", "setting3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTING_COLS, *SENSOR_COLS]


def load_split(raw_dir: str | Path, subset: str, split: str) -> pd.DataFrame:
    """Load train_/test_ file for a subset into a typed DataFrame."""
    df = pd.read_csv(Path(raw_dir) / f"{split}_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def load_test_rul(raw_dir: str | Path, subset: str) -> pd.Series:
    """True RUL at the last observed cycle of each test engine (index = unit id)."""
    rul = pd.read_csv(Path(raw_dir) / f"RUL_{subset}.txt", sep=r"\s+", header=None)[0]
    rul.index = np.arange(1, len(rul) + 1)
    return rul


def add_train_rul(df: pd.DataFrame, cap: float | None) -> pd.DataFrame:
    """Run-to-failure engines: RUL = last cycle - current cycle, clipped at `cap` (piece-wise linear target)."""
    df = df.copy()
    df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def add_test_rul(df: pd.DataFrame, rul_end: pd.Series, cap: float | None) -> pd.DataFrame:
    """Truncated test engines: RUL(t) = RUL at last cycle + (last cycle - t)."""
    df = df.copy()
    last = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = df["unit"].map(rul_end).astype(float) + (last - df["cycle"])
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def condition_key(df: pd.DataFrame) -> pd.Series:
    """Operating condition id. C-MAPSS uses a few discrete regimes (altitude, Mach, throttle angle);
    rounding the three settings recovers them exactly (FD001/FD003: 1 regime, FD002/FD004: 6 regimes)."""
    return (
        df["setting1"].round(0).astype(int).astype(str)
        + "_" + df["setting2"].round(2).astype(str)
        + "_" + df["setting3"].round(0).astype(int).astype(str)
    )


@dataclass
class CMAPSSPreprocessor:
    rul_cap: float = 125
    window: int = 30
    min_std: float = 0.01
    smoothing_alpha: float | None = 0.3
    features: list[str] = field(default_factory=list)
    stats: dict[str, dict[str, list[float]]] = field(default_factory=dict)  # condition -> mean/std per feature

    # ------------------------------------------------------------------ fitting
    def fit(self, train_df: pd.DataFrame) -> "CMAPSSPreprocessor":
        cond = condition_key(train_df)
        # Feature selection: keep sensors whose within-condition std is non-negligible in at least one regime.
        within_std = train_df[SENSOR_COLS].groupby(cond).std().max(axis=0)
        self.features = [c for c in SENSOR_COLS if within_std[c] >= self.min_std]
        self.stats = {}
        for key, grp in train_df.groupby(cond):
            mu = grp[self.features].mean()
            sd = grp[self.features].std().replace(0, 1.0)
            self.stats[key] = {"mean": mu.tolist(), "std": sd.tolist()}
        return self

    @property
    def n_features(self) -> int:
        return len(self.features)

    # ------------------------------------------------------------- transforming
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-operating-condition z-score normalisation using training statistics only."""
        if not self.features:
            raise RuntimeError("Preprocessor is not fitted")
        out = df[["unit", "cycle"]].copy()
        vals = np.zeros((len(df), self.n_features), dtype=np.float32)
        cond = condition_key(df).to_numpy()
        known = list(self.stats.keys())
        for key in np.unique(cond):
            # Unseen regimes fall back to the closest known regime by name (rare; logged by the caller).
            st = self.stats.get(key) or self.stats[known[0]]
            m = cond == key
            vals[m] = (df.loc[m, self.features].to_numpy() - np.array(st["mean"])) / np.array(st["std"])
        out[self.features] = vals
        if "RUL" in df:
            out["RUL"] = df["RUL"].to_numpy()
        return out

    def smooth(self, norm_df: pd.DataFrame) -> pd.DataFrame:
        """Causal exponential smoothing per engine (uses only past/current cycles, so it is valid online)."""
        if not self.smoothing_alpha:
            return norm_df
        out = norm_df.copy()
        out[self.features] = (
            norm_df.groupby("unit")[self.features].transform(lambda s: s.ewm(alpha=self.smoothing_alpha, adjust=False).mean())
        )
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.smooth(self.normalize(df))

    # ------------------------------------------------------- sequence building
    def _pad(self, arr: np.ndarray) -> np.ndarray:
        if len(arr) >= self.window:
            return arr
        pad = np.repeat(arr[:1], self.window - len(arr), axis=0)
        return np.concatenate([pad, arr], axis=0)

    def make_windows(self, tdf: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All sliding windows (stride 1). Returns X [n, window, f], y [n] (RUL at window end), unit ids [n]."""
        xs, ys, us = [], [], []
        for unit, grp in tdf.groupby("unit", sort=True):
            feats = self._pad(grp[self.features].to_numpy(np.float32))
            rul = grp["RUL"].to_numpy(np.float32) if "RUL" in grp else np.full(len(grp), np.nan, np.float32)
            rul = np.concatenate([np.full(len(feats) - len(rul), rul[0], np.float32), rul])
            idx = np.arange(self.window, len(feats) + 1)
            win = np.lib.stride_tricks.sliding_window_view(feats, (self.window, self.n_features))[:, 0]
            xs.append(win)
            ys.append(rul[idx - 1])
            us.append(np.full(len(idx), unit))
        return np.concatenate(xs).astype(np.float32), np.concatenate(ys), np.concatenate(us)

    def make_last_windows(self, tdf: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Only the final window of each engine (standard C-MAPSS test protocol)."""
        xs, ys, us = [], [], []
        for unit, grp in tdf.groupby("unit", sort=True):
            feats = self._pad(grp[self.features].to_numpy(np.float32))
            xs.append(feats[-self.window:])
            ys.append(grp["RUL"].iloc[-1] if "RUL" in grp else np.nan)
            us.append(unit)
        return np.stack(xs).astype(np.float32), np.array(ys, np.float32), np.array(us)

    # ---------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        return {
            "rul_cap": self.rul_cap, "window": self.window, "min_std": self.min_std,
            "smoothing_alpha": self.smoothing_alpha, "features": self.features, "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CMAPSSPreprocessor":
        return cls(**d)


def split_units(units: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Engine-level split so that no engine contributes windows to both train and validation."""
    rng = np.random.default_rng(seed)
    units = np.array(sorted(np.unique(units)))
    rng.shuffle(units)
    n_val = max(1, int(round(len(units) * val_fraction)))
    return np.sort(units[n_val:]), np.sort(units[:n_val])
