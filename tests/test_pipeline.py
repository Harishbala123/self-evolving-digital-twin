import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (SENSOR_COLS, SETTING_COLS, CMAPSSPreprocessor, add_test_rul, add_train_rul,
                                 split_units)
from src.drift.detectors import ADWIN, PageHinkley
from src.drift.injection import inject_sensor_offset
from src.drift.monitor import ExpectedSensorModel, calibrate_threshold, run_page_hinkley
from src.evaluation.metrics import nasa_score, regression_report


def synthetic_fleet(n_units=6, seed=0, min_len=40, max_len=80):
    """Small C-MAPSS-shaped frame: sensors drift linearly toward failure, plus noise; s1 is constant."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        T = rng.integers(min_len, max_len)
        for c in range(1, T + 1):
            life = c / T
            r = {"unit": u, "cycle": c, "setting1": 0.0, "setting2": 0.0, "setting3": 100.0}
            for j, s in enumerate(SENSOR_COLS):
                r[s] = 1.0 if s == "s1" else 10 * j + 3 * life * (1 if j % 2 else -1) + rng.normal(0, 0.3)
            rows.append(r)
    return pd.DataFrame(rows)


def test_train_rul_is_capped_and_reaches_zero():
    df = add_train_rul(synthetic_fleet(), cap=30)
    assert df.RUL.max() == 30
    assert (df.groupby("unit").RUL.last() == 0).all()


def test_test_rul_counts_down_to_given_value():
    df = synthetic_fleet(3)
    end = pd.Series([5, 10, 15], index=[1, 2, 3])
    out = add_test_rul(df, end, cap=None)
    assert out.groupby("unit").RUL.last().tolist() == [5, 10, 15]
    g = out[out.unit == 1].RUL.to_numpy()
    assert np.all(np.diff(g) == -1)


def test_feature_selection_drops_constant_sensor_and_windows_shape():
    df = add_train_rul(synthetic_fleet(), cap=125)
    prep = CMAPSSPreprocessor(window=10, smoothing_alpha=0.3).fit(df)
    assert "s1" not in prep.features and len(prep.features) == 20
    X, y, u = prep.make_windows(prep.transform(df))
    expected = sum(len(g) - 10 + 1 for _, g in df.groupby("unit"))
    assert X.shape == (expected, 10, 20) and len(y) == expected == len(u)
    Xl, yl, ul = prep.make_last_windows(prep.transform(df))
    assert Xl.shape == (6, 10, 20) and np.all(yl == 0)


def test_short_engine_is_padded():
    df = add_train_rul(synthetic_fleet(2, min_len=5, max_len=6), cap=125)
    prep = CMAPSSPreprocessor(window=10).fit(df)
    X, _, _ = prep.make_last_windows(prep.transform(df))
    assert X.shape[1] == 10


def test_smoothing_is_causal():
    df = add_train_rul(synthetic_fleet(1), cap=125)
    prep = CMAPSSPreprocessor(window=5).fit(df)
    full = prep.transform(df)
    part = prep.transform(df.iloc[:20])
    np.testing.assert_allclose(full.iloc[:20][prep.features].to_numpy(), part[prep.features].to_numpy(), atol=1e-5)


def test_preprocessor_roundtrip():
    df = add_train_rul(synthetic_fleet(), cap=125)
    prep = CMAPSSPreprocessor(window=10).fit(df)
    prep2 = CMAPSSPreprocessor.from_dict(prep.to_dict())
    pd.testing.assert_frame_equal(prep.transform(df), prep2.transform(df))


def test_unit_split_is_disjoint():
    tr, va = split_units(np.arange(1, 101), 0.2, 42)
    assert len(va) == 20 and len(set(tr) & set(va)) == 0 and len(tr) + len(va) == 100


def test_page_hinkley_detects_shift_not_noise():
    rng = np.random.default_rng(1)
    clean = rng.normal(0, 1, 300)
    assert run_page_hinkley(clean, 0.0, 0.5, 15) is None
    shifted = np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 1, 100)])
    idx = run_page_hinkley(shifted, 0.0, 0.5, 15)
    assert idx is not None and 100 <= idx < 115


def test_page_hinkley_running_mean_mode():
    ph = PageHinkley(delta=0.1, threshold=10)
    for x in np.r_[np.zeros(50), np.full(50, 2.0)]:
        if ph.update(x):
            break
    assert ph.drift_detected and ph.n > 50


def test_adwin_detects_mean_change():
    rng = np.random.default_rng(2)
    ad = ADWIN(delta=0.002)
    hits = [i for i, x in enumerate(np.r_[rng.normal(0, 1, 500), rng.normal(2, 1, 500)]) if ad.update(x)]
    assert hits and 500 <= hits[0] < 600
    assert ad.width < 600            # old window part was dropped


def test_adwin_quiet_on_stationary_stream():
    rng = np.random.default_rng(3)
    ad = ADWIN(delta=0.002)
    assert not any(ad.update(x) for x in rng.normal(0, 1, 1000))


def test_offset_injection_only_after_start():
    df = synthetic_fleet(2)
    out = inject_sensor_offset(df, ["s3"], {"s3": 2.0}, 1.5, start_cycle=10)
    diff = out.s3 - df.s3
    assert np.allclose(diff[df.cycle < 10], 0) and np.allclose(diff[df.cycle >= 10], 3.0)
    ramp = inject_sensor_offset(df, ["s3"], {"s3": 1.0}, 1.0, start_cycle=10, mode="ramp", ramp_cycles=5)
    d = (ramp.s3 - df.s3)[df.unit == 1].to_numpy()
    assert np.allclose(d[8:14], [0, 0.2, 0.4, 0.6, 0.8, 1.0])


def test_expected_sensor_model_flags_bias():
    df = add_train_rul(synthetic_fleet(8), cap=125)
    prep = CMAPSSPreprocessor(window=5, smoothing_alpha=None).fit(df)
    tdf = prep.transform(df)
    esm = ExpectedSensorModel(rul_cap=125).fit(tdf, prep.features)
    base = esm.score(tdf, tdf.RUL.to_numpy()).mean()
    biased = tdf.copy()
    biased[prep.features[:5]] += 5.0
    assert esm.score(biased, biased.RUL.to_numpy()).mean() > base + 1


def test_threshold_calibration_controls_false_alarms():
    rng = np.random.default_rng(4)
    clean = [rng.normal(1, 0.3, 150) for _ in range(40)]
    lam = calibrate_threshold(clean, 1.0, 0.1, 0.05)
    fa = np.mean([run_page_hinkley(s, 1.0, 0.1, lam) is not None for s in clean])
    assert fa <= 0.05


def test_metrics():
    assert nasa_score([10, 10], [10, 10]) == 0
    assert nasa_score([10], [20]) > nasa_score([20], [10])     # late predictions are penalised more
    r = regression_report([1, 2, 3], [1, 2, 4])
    assert r["rmse"] == pytest.approx(np.sqrt(1 / 3))
