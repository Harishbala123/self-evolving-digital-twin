"""Minimal model registry: every model version, why it was created, and its metrics."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.utils.common import load_json, save_json


class ModelRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = load_json(self.path) if self.path.exists() else {"active": None, "versions": []}

    def register(self, model_path: str | Path, trigger: str, parent: str | None, metrics: dict,
                 notes: str = "", activate: bool = True) -> str:
        version = f"v{len(self.data['versions']) + 1}"
        self.data["versions"].append({
            "version": version, "model_path": str(Path(model_path).name), "parent": parent,
            "trigger": trigger, "created": datetime.now().isoformat(timespec="seconds"),
            "metrics": metrics, "notes": notes,
        })
        if activate:
            self.data["active"] = version
        self.save()
        return version

    def get(self, version: str) -> dict:
        return next(v for v in self.data["versions"] if v["version"] == version)

    @property
    def active(self) -> str | None:
        return self.data["active"]

    def save(self) -> None:
        save_json(self.data, self.path)
