import math
import os
import random
import sys
from typing import List, Sequence, Tuple

_FILE_INFO = {
    "//": "Directory Path: examples/learning/demo_force_time_series_prediction.py",
    "//": "English Title: FORCE Time-Series Prediction Demo",
    "//": "Purpose/Content: Demonstrates stable next-step prediction on a sine wave using a spiking liquid reservoir with an online FORCE/RLS readout.",
}


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from sara_engine.encoders import TimeSeriesCurrentEncoder
from sara_engine.models.liquid_reservoir import LiquidReservoir


def build_signal(length: int, frequency: float = 0.08) -> List[float]:
    return [math.sin(frequency * step) for step in range(length)]


def evaluate_sequence(
    reservoir: LiquidReservoir,
    encoder: TimeSeriesCurrentEncoder,
    signal: Sequence[float],
    reset_readout: bool = False,
) -> Tuple[List[float], float]:
    reservoir.reset_dynamic_state(reset_readout=reset_readout)
    predictions: List[float] = []
    error_sum = 0.0

    for index in range(len(signal) - 1):
        previous_value = signal[index - 1] if index > 0 else signal[index]
        prediction = reservoir.predict_force(
            encoder.encode(
                signal[index],
                previous_value,
                reservoir.n,
            )
        )[0]
        target = signal[index + 1]
        predictions.append(prediction)
        error_sum += abs(prediction - target)

    mean_absolute_error = error_sum / max(1, len(predictions))
    return predictions, mean_absolute_error


def train_sequence(
    reservoir: LiquidReservoir,
    encoder: TimeSeriesCurrentEncoder,
    signal: Sequence[float],
    epochs: int,
) -> None:
    for epoch in range(epochs):
        reservoir.reset_dynamic_state(reset_readout=False)
        for index in range(len(signal) - 1):
            previous_value = signal[index - 1] if index > 0 else signal[index]
            reservoir.train_force(
                encoder.encode(
                    signal[index],
                    previous_value,
                    reservoir.n,
                ),
                [signal[index + 1]],
            )
        if (epoch + 1) % 10 == 0:
            _, mae = evaluate_sequence(reservoir, encoder, signal)
            print(f"Epoch {epoch + 1:02d} | mean_absolute_error={mae:.4f}")


def main() -> None:
    random.seed(7)

    print("=" * 64)
    print("FORCE Time-Series Prediction Demo")
    print("=" * 64)

    train_signal = build_signal(length=220, frequency=0.09)
    test_signal = build_signal(length=120, frequency=0.09)
    encoder = TimeSeriesCurrentEncoder()

    reservoir = LiquidReservoir(
        n_neurons=48,
        p_connect=0.12,
        readout_decay=0.92,
        enable_force_readout=True,
        force_output_dim=1,
        force_alpha=1.0,
        force_forgetting_factor=0.999,
    )

    _, baseline_mae = evaluate_sequence(
        reservoir,
        encoder,
        test_signal,
        reset_readout=True,
    )
    print(f"Baseline test MAE: {baseline_mae:.4f}")

    train_sequence(reservoir, encoder, train_signal, epochs=30)

    predictions, test_mae = evaluate_sequence(reservoir, encoder, test_signal)
    print(f"Final test MAE: {test_mae:.4f}")

    print("\nSample predictions:")
    for index in range(8):
        current_value = test_signal[index]
        target = test_signal[index + 1]
        prediction = predictions[index]
        print(
            f"step={index:02d} input={current_value:+.4f} "
            f"target={target:+.4f} prediction={prediction:+.4f}"
        )


if __name__ == "__main__":
    main()
