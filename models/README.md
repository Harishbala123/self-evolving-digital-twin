# Saved models

All models are small (< 1 MB), so they are committed directly.

| File | Description |
|---|---|
| `lstm_v1.keras` | Base LSTM RUL model trained on FD001 (`scripts/train.py`) |
| `preprocessor.json` | Fitted preprocessing state: selected sensors, per-condition mean/std, window, RUL cap, smoothing |
| `monitor_v1.json` | Expected-sensor drift monitor + Page-Hinkley parameters tuned on validation engines |
| `lstm_v2_sensor_bias.keras` | v1 fine-tuned after the sensor-bias drift alarm (`scripts/run_drift_experiment.py`) |
| `lstm_v3_fd003.keras` | v1 fine-tuned after the new-fault-mode (FD003) drift alarm |
| `registry.json` | Model registry: version lineage, trigger reason, metrics |
| `twin_demo/lstm_v2.keras`, `twin_demo/lstm_v3.keras` | Versions created live by the fleet digital-twin demo |

Load a model:
```python
import keras, json
from src.data.preprocess import CMAPSSPreprocessor
model = keras.models.load_model("models/lstm_v1.keras")
prep = CMAPSSPreprocessor.from_dict(json.load(open("models/preprocessor.json")))
```
