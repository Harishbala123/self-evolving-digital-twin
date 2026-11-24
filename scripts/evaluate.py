"""Evaluate the trained model and write metrics + graphs.  Usage: python scripts/evaluate.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.evaluate import evaluate  # noqa: E402
from src.utils.common import load_config  # noqa: E402

if __name__ == "__main__":
    evaluate(load_config())
