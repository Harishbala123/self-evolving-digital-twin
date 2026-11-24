"""Download + validate NASA C-MAPSS.  Usage: python scripts/download_data.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.download import download_cmapss, validate_cmapss  # noqa: E402
from src.utils.common import load_config, project_path  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    raw = download_cmapss(cfg["data"]["url"], project_path(cfg["data"]["raw_dir"]))
    validate_cmapss(raw)
