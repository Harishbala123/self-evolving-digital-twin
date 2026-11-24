"""Train the base LSTM RUL model.  Usage: python scripts/train.py [--epochs N]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train import train  # noqa: E402
from src.utils.common import load_config  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    train(cfg)
