"""Interactive demo of the self-evolving digital twin.

Run:  streamlit run app.py        (or double-click run_demo.bat)
Pick an engine, optionally inject sensor drift, and watch the twin estimate RUL, track health and raise drift alarms.
"""
from __future__ import annotations

import json

import keras
import numpy as np
import pandas as pd
import streamlit as st

from src.data.download import download_cmapss, is_downloaded
from src.data.preprocess import CMAPSSPreprocessor
from src.drift.experiments import load_subset
from src.drift.injection import inject_sensor_offset, sensor_stds
from src.drift.monitor import ExpectedSensorModel
from src.evaluation.metrics import regression_report
from src.twin.digital_twin import EngineTwin
from src.utils.common import load_config, load_json, project_path
from src.utils.plotting import CRITICAL, GOOD, INK_2, SERIES, WARNING, plt

st.set_page_config(page_title="Digital Twin Demo", layout="wide")
cfg = load_config()
MODELS = {
    "v1: base model (FD001)": "lstm_v1.keras",
    "v2: retrained after sensor-bias drift": "lstm_v2_sensor_bias.keras",
    "v3: retrained after new fault mode (FD003)": "lstm_v3_fd003.keras",
}


@st.cache_resource
def load_model(fname: str):
    return keras.models.load_model(project_path("models", fname))


@st.cache_resource
def load_twin_parts():
    prep = CMAPSSPreprocessor.from_dict(load_json(project_path("models", "preprocessor.json")))
    mon = load_json(project_path("models", "monitor_v1.json"))
    return prep, ExpectedSensorModel.from_dict(mon["esm"]), mon


@st.cache_data
def load_data(subset: str):
    return load_subset(cfg, subset)


raw_dir = project_path(cfg["data"]["raw_dir"])
st.title("Self-Evolving Digital Twin: Turbofan Predictive Maintenance")
st.caption("LSTM Remaining-Useful-Life forecasting on NASA C-MAPSS with Page-Hinkley drift detection. "
           "Every number below is computed live by the same code that produced the README results.")

if not is_downloaded(raw_dir):
    st.warning("The NASA C-MAPSS dataset is not downloaded yet (~12 MB).")
    if st.button("Download dataset"):
        with st.spinner("Downloading and validating C-MAPSS..."):
            download_cmapss(cfg["data"]["url"], raw_dir)
        st.rerun()
    st.stop()

prep, esm, mon = load_twin_parts()
tab_demo, tab_results, tab_how = st.tabs(["Live twin", "Experiment results", "How it works"])

# ---------------------------------------------------------------- live demo
with tab_demo:
    c = st.sidebar
    c.header("Scenario")
    subset_label = c.radio("Engine fleet", ["FD001 test engines (training distribution)",
                                            "FD003 test engines (new fault mode = real drift)"])
    subset = "FD001" if subset_label.startswith("FD001") else "FD003"
    train_df, test_df = load_data(cfg["data"]["subset"])[0], load_data(subset)[1]
    units = sorted(test_df.unit.unique())
    unit = c.selectbox("Engine", units, index=units.index(31) if 31 in units else 0)
    model_label = c.selectbox("Model version", list(MODELS))
    inject = c.checkbox("Inject sensor-calibration drift", value=False)
    eng = test_df[test_df.unit == unit].reset_index(drop=True)
    n = int(eng.cycle.max())
    start, mag = None, 0.0
    if inject:
        start = c.slider("Drift starts at cycle", 1, max(2, n - 10), min(40, max(1, n - 10)))
        mag = c.slider("Bias size (training std)", 0.5, 5.0, float(cfg["drift"]["offset_sigma"]), 0.5)
        sensors = cfg["drift"]["offset_sensors"]
        stds = sensor_stds(train_df, sensors)
        eng = inject_sensor_offset(eng, sensors, stds, mag, start)
        c.caption(f"Bias added to sensors {', '.join(sensors)}")

    model = load_model(MODELS[model_label])
    twin = EngineTwin(int(unit), model, prep, esm, mon["ref_mean"], mon["delta"], mon["threshold"],
                      model_label.split(":")[0])
    hist = twin.replay(eng.drop(columns="RUL"))
    hist["true_rul"] = eng.RUL.to_numpy()
    last = hist.iloc[-1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted RUL (last cycle)", f"{last.predicted_rul:.0f} cycles",
              f"{last.predicted_rul - last.true_rul:+.0f} vs actual", delta_color="off")
    m2.metric("Actual RUL (capped at 125)", f"{last.true_rul:.0f} cycles")
    m3.metric("Health state", last.health_state)
    if twin.drift_cycle is None:
        m4.metric("Drift alarm", "none")
    else:
        delay = f"{twin.drift_cycle - start} cycles after onset" if start else "no drift injected"
        m4.metric("Drift alarm", f"cycle {twin.drift_cycle}", delay, delta_color="off")

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax = axes[0]
    cap = prep.rul_cap
    ax.axhspan(0.6 * cap, cap * 1.15, color=GOOD, alpha=0.08)
    ax.axhspan(0.25 * cap, 0.6 * cap, color=WARNING, alpha=0.10)
    ax.axhspan(0, 0.25 * cap, color=CRITICAL, alpha=0.08)
    ax.plot(hist.cycle, hist.true_rul, color=SERIES[0], label="Actual RUL (capped)")
    ax.plot(hist.cycle, hist.predicted_rul, color=SERIES[1], label="Twin estimate")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"Engine {unit} ({subset}): health bands Healthy / Degrading / Critical")
    ax.legend(loc="lower left")
    ax = axes[1]
    ax.plot(hist.cycle, hist.drift_score, color=SERIES[2], label="Drift score (mean z²)")
    if start:
        ax.axvline(start, color=INK_2, ls="--", lw=1, label="Drift injected")
    if twin.drift_cycle:
        ax.axvline(twin.drift_cycle, color=CRITICAL, lw=1.5, label="Page-Hinkley alarm")
    ax.set_xlabel("Operational cycle")
    ax.set_ylabel("Drift score")
    ax.legend(loc="upper left")
    st.pyplot(fig)

    rep = regression_report(hist.true_rul, hist.predicted_rul)
    st.write(f"Trajectory RMSE for this engine: **{rep['rmse']:.1f} cycles** (MAE {rep['mae']:.1f}).")
    with st.expander("Twin state log (every cycle)"):
        st.dataframe(hist[["cycle", "predicted_rul", "true_rul", "health_index", "health_state", "drift_score",
                           "ph_statistic", "drift_alarm", "model_version"]].round(2))
    st.info("Try this: engine 31, model v1, inject drift at cycle 40. The alarm fires a few cycles later and v1 "
            "misreads the bias as degradation (RUL collapses). Switch to model v2, which was retrained on post-drift "
            "data: its end-of-life estimate recovers. Set the drift start to cycle 1 (an engine that entered service "
            "after the recalibration, the case v2 was trained for) to see v2 track the whole life. The drift monitor "
            "stays calibrated for v1, so it still flags the bias as a change from the original data.")

# ---------------------------------------------------------------- results
with tab_results:
    res = project_path("results")
    st.subheader("RUL model (FD001)")
    st.dataframe(pd.read_csv(res / "metrics" / "evaluation_metrics.csv").round(3))
    c1, c2 = st.columns(2)
    c1.image(str(res / "graphs" / "test_pred_vs_actual.png"))
    c2.image(str(res / "graphs" / "loss_curves.png"))
    st.subheader("Drift detection and retraining")
    s = json.loads((res / "metrics" / "drift_summary.json").read_text())
    st.dataframe(pd.DataFrame({k: v for k, v in s.items() if k not in ("retraining", "monitor")}).T,
)
    st.dataframe(pd.read_csv(res / "metrics" / "retraining_before_after.csv")[
        ["scenario", "eval_set", "model", "rmse", "mae", "r2"]].round(2))
    c1, c2 = st.columns(2)
    c1.image(str(res / "graphs" / "drift_detection_delay.png"))
    c2.image(str(res / "graphs" / "retraining_before_after.png"))
    st.subheader("Fleet digital-twin replay")
    st.image(str(res / "graphs" / "twin_fleet_timeline.png"))
    st.dataframe(pd.read_csv(res / "reports" / "twin_events.csv"))

# ---------------------------------------------------------------- explanation
with tab_how:
    st.markdown("""
**Pipeline:** raw sensor readings → 14 informative sensors → per-condition z-scores → causal smoothing →
30-cycle windows → stacked LSTM (64→32 units) → Remaining Useful Life.

**Digital twin:** each engine has a virtual state, updated every cycle: RUL estimate, health index (RUL/125),
health state, and a *drift score*. The drift score compares the sensors with what the twin expects at the
current health level (a cubic model of each sensor vs RUL, learned from training engines).

**Drift detection:** a Page-Hinkley test on the drift score raises the alarm. Its parameters were tuned on
validation engines only (≤ 5 % false alarms). ADWIN watches the fleet's prediction error once true RUL becomes known.

**Self-evolution:** alarms on ≥ 3 of the last 10 engines trigger fine-tuning on post-drift data plus a replay
buffer. The candidate is deployed only if validation RMSE improves; every version is recorded in `models/registry.json`.
""")
    st.json(load_json(project_path("models", "registry.json")), expanded=False)
