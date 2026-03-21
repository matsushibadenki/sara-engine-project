# Directory Path: examples/learning/demo_unified_snn.py
# English Title: Unified SNN Minimal Demo
# Purpose/Content: Minimal end-to-end demo showing UnifiedSNNModel with reservoir + JEPA + FORCE readout.

import argparse
import math
import os
import random
import sys
from typing import List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.encoders import TimeSeriesCurrentEncoder
from sara_engine.models.liquid_reservoir import LiquidReservoir
from sara_engine.models.spiking_jepa import SpikingJEPA
from sara_engine.models.unified_snn import UnifiedSNNModel


def build_signal(length: int, frequency: float = 0.12) -> List[float]:
    return [math.sin(frequency * step) for step in range(length)]


def spikes_from_value(value: float, vocab_size: int = 64) -> List[int]:
    """Map a scalar value to a small set of spike indices."""
    center = int((value + 1.0) * 0.5 * (vocab_size - 1))
    center = max(0, min(vocab_size - 1, center))
    spread = 2
    return [max(0, min(vocab_size - 1, center + offset)) for offset in range(-spread, spread + 1)]


def run_epoch(
    model: UnifiedSNNModel,
    encoder: TimeSeriesCurrentEncoder,
    signal: List[float],
    vocab_size: int,
    log_every: int,
) -> float:
    total_mae = 0.0
    for step in range(len(signal) - 1):
        previous_value = signal[step - 1] if step > 0 else signal[step]
        currents = encoder.encode(signal[step], previous_value, model.reservoir.n)
        target_readout = [signal[step + 1]]
        target_future_spikes = spikes_from_value(signal[step + 1], vocab_size=vocab_size)

        output = model.step(
            external_currents=currents,
            target_readout=target_readout,
            target_future_spikes=target_future_spikes,
            learning=True,
        )

        prediction = output["readout"][0] if output["readout"] else 0.0
        total_mae += abs(prediction - target_readout[0])

        if log_every > 0 and step % log_every == 0:
            print(
                f"Step {step:02d} | target={target_readout[0]:+.4f} "
                f"pred={prediction:+.4f} | spikes={len(output['spikes'])}"
            )

    return total_mae / max(1, len(signal) - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified SNN Minimal Demo")
    parser.add_argument("--length", type=int, default=80, help="Signal length.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--frequency", type=float, default=0.12, help="Signal frequency.")
    parser.add_argument("--log-every", type=int, default=10, help="Log interval per step.")
    args = parser.parse_args()

    random.seed(7)
    print("=" * 60)
    print("Unified SNN Minimal Demo")
    print("=" * 60)

    signal = build_signal(length=max(8, args.length), frequency=args.frequency)
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
    jepa = SpikingJEPA(
        layer_configs=[{"embed_dim": 64, "hidden_dim": 96}],
        ema_decay=0.95,
        learning_rate=0.1,
        time_scales={"base": 1},
    )
    model = UnifiedSNNModel(reservoir=reservoir, jepa=jepa)

    vocab_size = 64
    epochs = max(1, args.epochs)
    for epoch in range(epochs):
        mean_mae = run_epoch(
            model,
            encoder,
            signal,
            vocab_size=vocab_size,
            log_every=args.log_every,
        )
        print(f"Epoch {epoch + 1:02d}/{epochs} | mean absolute error: {mean_mae:.4f}")


if __name__ == "__main__":
    main()
