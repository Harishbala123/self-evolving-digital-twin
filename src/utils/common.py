"""Shared helpers: project paths, config loading, seeding and logging."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str | os.PathLike) -> Path:
    """Return an absolute path inside the project (works on Windows and Colab)."""
    return PROJECT_ROOT.joinpath(*parts)


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    path = Path(path) if path else project_path("configs", "config.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow (if it is installed) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:  # tests that do not need TF
        pass


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_json(obj: Any, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")
