"""Download and validate the NASA C-MAPSS turbofan degradation dataset.

Source: NASA Prognostics Center of Excellence (PCoE) data repository,
"Turbofan Engine Degradation Simulation Data Set" (Saxena et al., 2008).
The archive is mirrored by NASA on the public PHM datasets S3 bucket.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.utils.common import get_logger

log = get_logger(__name__)

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
# Number of engines (train, test) published by NASA for each subset, used to validate the download.
EXPECTED_UNITS = {"FD001": (100, 100), "FD002": (260, 259), "FD003": (100, 100), "FD004": (249, 248)}
N_COLUMNS = 26  # unit, cycle, 3 operational settings, 21 sensors


def _expected_files() -> list[str]:
    files = []
    for s in SUBSETS:
        files += [f"train_{s}.txt", f"test_{s}.txt", f"RUL_{s}.txt"]
    return files


def is_downloaded(raw_dir: Path) -> bool:
    return all((raw_dir / f).exists() for f in _expected_files())


def _extract_cmapss(zip_bytes: bytes, raw_dir: Path) -> None:
    """The NASA archive contains a nested CMAPSSData.zip; extract the .txt files from whichever level holds them."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        txt = [n for n in names if Path(n).name in _expected_files()]
        if txt:
            for n in txt:
                (raw_dir / Path(n).name).write_bytes(zf.read(n))
            return
        nested = [n for n in names if n.lower().endswith(".zip")]
        if not nested:
            raise RuntimeError(f"No C-MAPSS files in archive: {names[:10]}")
        for n in nested:
            _extract_cmapss(zf.read(n), raw_dir)


def download_cmapss(url: str, raw_dir: str | Path, force: bool = False) -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if is_downloaded(raw_dir) and not force:
        log.info("C-MAPSS already present in %s, skipping download", raw_dir)
        return raw_dir
    log.info("Downloading C-MAPSS from %s", url)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    log.info("Downloaded %.1f MB, extracting", len(resp.content) / 1e6)
    _extract_cmapss(resp.content, raw_dir)
    validate_cmapss(raw_dir)
    return raw_dir


def validate_cmapss(raw_dir: str | Path) -> dict[str, dict[str, int]]:
    """Check presence, column count, engine counts and missing values for every subset."""
    raw_dir = Path(raw_dir)
    report = {}
    for s in SUBSETS:
        tr = pd.read_csv(raw_dir / f"train_{s}.txt", sep=r"\s+", header=None)
        te = pd.read_csv(raw_dir / f"test_{s}.txt", sep=r"\s+", header=None)
        rul = pd.read_csv(raw_dir / f"RUL_{s}.txt", sep=r"\s+", header=None)
        if tr.shape[1] != N_COLUMNS or te.shape[1] != N_COLUMNS:
            raise ValueError(f"{s}: unexpected column count {tr.shape[1]}/{te.shape[1]}")
        n_tr, n_te = tr[0].nunique(), te[0].nunique()
        if (n_tr, n_te) != EXPECTED_UNITS[s]:
            raise ValueError(f"{s}: engine counts {(n_tr, n_te)} != {EXPECTED_UNITS[s]}")
        if len(rul) != n_te:
            raise ValueError(f"{s}: RUL file has {len(rul)} rows for {n_te} test engines")
        if tr.isna().any().any() or te.isna().any().any():
            raise ValueError(f"{s}: missing values found")
        report[s] = {"train_rows": len(tr), "test_rows": len(te), "train_engines": n_tr, "test_engines": n_te}
        log.info("%s OK: %s", s, report[s])
    return report
