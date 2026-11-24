"""Tests that need TensorFlow: model build/fit and online-vs-replay equivalence of the digital twin."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.data.preprocess import CMAPSSPreprocessor, add_train_rul  # noqa: E402
from src.drift.monitor import ExpectedSensorModel  # noqa: E402
from src.models.lstm import build_lstm  # noqa: E402
from src.twin.digital_twin import EngineTwin, FleetTwin, health_state  # noqa: E402
from src.twin.registry import ModelRegistry  # noqa: E402
from tests.test_pipeline import synthetic_fleet  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    df = add_train_rul(synthetic_fleet(6), cap=125)
    prep = CMAPSSPreprocessor(window=10, smoothing_alpha=0.3).fit(df)
    tdf = prep.transform(df)
    X, y, _ = prep.make_windows(tdf)
    model = build_lstm(10, prep.n_features, (8, 4), 4, 0.0, 1e-2)
    model.fit(X, y, epochs=2, batch_size=64, verbose=0)
    esm = ExpectedSensorModel(rul_cap=125).fit(tdf, prep.features)
    return df, prep, model, esm


def test_model_output_shape(setup):
    df, prep, model, _ = setup
    X, _, _ = prep.make_windows(prep.transform(df))
    assert model.predict(X[:7], verbose=0).shape == (7, 1)


def test_online_ingest_matches_replay(setup):
    df, prep, model, esm = setup
    eng = df[df.unit == 1].drop(columns="RUL")
    online = EngineTwin(1, model, prep, esm, 1.0, 0.5, 50.0)
    for rec in eng.to_dict("records")[:25]:
        online.ingest(rec)
    replay = EngineTwin(1, model, prep, esm, 1.0, 0.5, 50.0)
    replay.replay(eng.iloc[:25])
    a, b = online.history_frame(), replay.history_frame()
    np.testing.assert_allclose(a.predicted_rul.to_numpy()[9:], b.predicted_rul.to_numpy()[9:], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(a.drift_score.to_numpy()[9:], b.drift_score.to_numpy()[9:], rtol=1e-3, atol=1e-3)


def test_fleet_trigger_and_events(setup):
    df, prep, model, esm = setup
    fleet = FleetTwin(model, "v1", prep, esm, ph_ref_mean=0.0, ph_delta=0.0, ph_threshold=0.5, engines_for_trigger=2)
    for u in (1, 2):
        g = df[df.unit == u]
        fleet.process_engine(f"e{u}", u, g.drop(columns="RUL"), g.RUL.to_numpy())
    assert fleet.retrain_needed()                       # threshold so low that every engine alarms
    assert any(e["event"] == "input_drift_alarm" for e in fleet.events)
    fleet.deploy(model, "v2", esm, 1.0, 100.0)
    assert fleet.model_version == "v2" and fleet.retrain_needed() is None


def test_registry(tmp_path):
    reg = ModelRegistry(tmp_path / "r.json")
    v1 = reg.register("m1.keras", "init", None, {"rmse": 1.0})
    v2 = reg.register("m2.keras", "drift", v1, {"rmse": 0.5}, activate=False)
    assert (v1, v2, reg.active) == ("v1", "v2", "v1")
    assert ModelRegistry(tmp_path / "r.json").get("v2")["parent"] == "v1"


def test_health_state():
    assert [health_state(x) for x in (0.9, 0.4, 0.1)] == ["Healthy", "Degrading", "Critical"]
