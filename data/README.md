# Data: NASA C-MAPSS

The data is **not committed**. `python scripts/download_data.py` downloads and validates it into `data/raw/`.

| Item | Value |
|---|---|
| Dataset | Turbofan Engine Degradation Simulation Data Set (C-MAPSS) |
| Publisher | NASA Ames Prognostics Center of Excellence (PCoE) |
| Source | NASA PCoE data repository, file `6. Turbofan Engine Degradation Simulation Data Set.zip` (mirrored at `https://phm-datasets.s3.amazonaws.com/NASA/...`) |
| Licence | U.S. Government work, public domain. Cite the reference below. |
| Citation | A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", *Proc. 1st Int. Conf. on Prognostics and Health Management (PHM08)*, Denver, 2008. |

Files used (each has 26 space-separated columns: `unit, cycle, setting1-3, s1-s21`):

| Subset | Train engines | Test engines | Operating conditions | Fault modes | Used for |
|---|---|---|---|---|---|
| FD001 | 100 | 100 | 1 | HPC degradation | base model |
| FD003 | 100 | 100 | 1 | HPC + fan degradation | real concept-drift scenario |
| FD002 / FD004 | 260 / 249 | 259 / 248 | 6 | 1 / 2 | supported by the preprocessing (per-condition normalisation), not used in the reported results |

The download script checks the column count, engine counts, RUL-file lengths and missing values for all four subsets.
