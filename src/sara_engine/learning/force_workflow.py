# Directory Path: src/sara_engine/learning/force_workflow.py
# English Title: FORCE Workflow Helpers
# Purpose/Content: Shared helpers for loading scalar time-series data and running FORCE training or inference with spiking reservoirs.

import json
import math
from typing import Any, Dict, List, Sequence, Tuple

from ..encoders.time_series import TimeSeriesCurrentEncoder
from ..models.liquid_reservoir import LiquidReservoir


def build_sine_series(length: int, frequency: float, phase: float = 0.0) -> List[float]:
    return [math.sin(frequency * step + phase) for step in range(length)]


def load_series(series_path: str) -> List[float]:
    with open(series_path, "r", encoding="utf-8") as handle:
        raw_text = handle.read().strip()

    if not raw_text:
        raise ValueError("Time-series file is empty.")

    if series_path.endswith(".json"):
        json_values = json.loads(raw_text)
        if not isinstance(json_values, list):
            raise ValueError("JSON time-series must be a list of numbers.")
        return [float(value) for value in json_values]

    normalized = raw_text.replace(",", "\n")
    values: List[float] = []
    for line in normalized.splitlines():
        stripped = line.strip()
        if stripped:
            values.append(float(stripped))
    return values


def split_series(series: Sequence[float], test_ratio: float) -> Tuple[List[float], List[float]]:
    if len(series) < 8:
        raise ValueError("Time-series must contain at least 8 values.")

    clamped_ratio = min(0.5, max(0.05, test_ratio))
    split_index = int(len(series) * (1.0 - clamped_ratio))
    split_index = max(4, min(len(series) - 4, split_index))
    return list(series[:split_index]), list(series[split_index:])


def evaluate_force_sequence(
    reservoir: LiquidReservoir,
    encoder: TimeSeriesCurrentEncoder,
    signal: Sequence[float],
    reset_readout: bool = False,
) -> Dict[str, Any]:
    reservoir.reset_dynamic_state(reset_readout=reset_readout)
    predictions: List[float] = []
    mae_sum = 0.0
    mse_sum = 0.0

    for index in range(len(signal) - 1):
        previous_value = signal[index - 1] if index > 0 else signal[index]
        prediction = reservoir.predict_force(
            encoder.encode(signal[index], previous_value, reservoir.n)
        )[0]
        target = signal[index + 1]
        error = prediction - target
        predictions.append(prediction)
        mae_sum += abs(error)
        mse_sum += error * error

    count = max(1, len(predictions))
    return {
        "predictions": predictions,
        "mae": mae_sum / count,
        "rmse": math.sqrt(mse_sum / count),
        "targets": list(signal[1:]),
    }


def train_force_sequence(
    reservoir: LiquidReservoir,
    encoder: TimeSeriesCurrentEncoder,
    signal: Sequence[float],
    epochs: int,
    report_every: int,
) -> List[Dict[str, float]]:
    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        reservoir.reset_dynamic_state(reset_readout=False)
        for index in range(len(signal) - 1):
            previous_value = signal[index - 1] if index > 0 else signal[index]
            reservoir.train_force(
                encoder.encode(signal[index], previous_value, reservoir.n),
                [signal[index + 1]],
            )

        if (epoch + 1) % report_every == 0 or epoch == epochs - 1:
            train_metrics = evaluate_force_sequence(reservoir, encoder, signal)
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_mae": train_metrics["mae"],
                    "train_rmse": train_metrics["rmse"],
                }
            )
            print(
                f"Epoch {epoch + 1:03d} | "
                f"train_mae={train_metrics['mae']:.4f} "
                f"train_rmse={train_metrics['rmse']:.4f}"
            )
    return history
