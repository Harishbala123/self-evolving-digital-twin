"""Stacked-LSTM regressor for Remaining Useful Life."""
from __future__ import annotations

import keras
from keras import layers


def build_lstm(window: int, n_features: int, lstm_units=(64, 32), dense_units: int = 16,
               dropout: float = 0.2, learning_rate: float = 1e-3) -> keras.Model:
    inputs = keras.Input(shape=(window, n_features), name="sensor_window")
    x = inputs
    for i, units in enumerate(lstm_units):
        x = layers.LSTM(units, return_sequences=i < len(lstm_units) - 1, name=f"lstm_{i + 1}")(x)
        x = layers.Dropout(dropout, name=f"dropout_{i + 1}")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense")(x)
    outputs = layers.Dense(1, name="rul")(x)
    model = keras.Model(inputs, outputs, name="rul_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse"), keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def build_from_config(cfg: dict, n_features: int) -> keras.Model:
    m = cfg["model"]
    return build_lstm(cfg["preprocessing"]["window"], n_features, tuple(m["lstm_units"]), m["dense_units"],
                      m["dropout"], m["learning_rate"])
