# Directory Path: scripts/train/train_force_cli.py
# English Title: FORCE Reservoir Training CLI
# Purpose/Content: Trains a spiking liquid reservoir with an online FORCE/RLS readout on reusable time-series inputs and saves managed artifacts under workspace/ and models/.

import argparse
import os
import sys
from typing import Any, Dict


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from sara_engine.encoders.time_series import TimeSeriesCurrentEncoder
from sara_engine.learning.force_io import export_force_artifact
from sara_engine.learning.force_workflow import (
    build_sine_series,
    evaluate_force_sequence,
    load_series,
    split_series,
    train_force_sequence,
)
from sara_engine.models.liquid_reservoir import LiquidReservoir
from sara_engine.utils.project_paths import (
    model_path,
    workspace_path,
)

# Backward-compatible aliases for tests and downstream script reuse.
evaluate_sequence = evaluate_force_sequence
train_sequence = train_force_sequence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a spiking liquid reservoir with FORCE learning on a time-series."
    )
    parser.add_argument(
        "--series-path",
        default="",
        help="Optional path to a JSON or text time-series file. If omitted, a sine wave is generated.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--report-every", type=int, default=10, help="Epoch interval for progress reports.")
    parser.add_argument("--length", type=int, default=320, help="Generated sine series length when --series-path is omitted.")
    parser.add_argument("--frequency", type=float, default=0.09, help="Generated sine frequency.")
    parser.add_argument("--test-ratio", type=float, default=0.3, help="Holdout ratio for evaluation.")
    parser.add_argument("--neurons", type=int, default=48, help="Number of reservoir neurons.")
    parser.add_argument("--connectivity", type=float, default=0.12, help="Reservoir connection probability.")
    parser.add_argument("--readout-decay", type=float, default=0.92, help="Low-pass decay for reservoir state collection.")
    parser.add_argument("--force-alpha", type=float, default=1.0, help="Initial RLS inverse-correlation scale.")
    parser.add_argument("--forgetting-factor", type=float, default=0.999, help="FORCE forgetting factor.")
    parser.add_argument(
        "--report-path",
        default=workspace_path("force_training", "latest_report.json"),
        help="Managed output path for training metrics JSON.",
    )
    parser.add_argument(
        "--model-dir",
        default=model_path("force_time_series"),
        help="Managed output directory for exported FORCE artifacts.",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("FORCE Reservoir Training CLI")
    print("=" * 64)

    if args.series_path:
        series = load_series(args.series_path)
        series_name = os.path.basename(args.series_path)
    else:
        series = build_sine_series(length=max(16, args.length), frequency=args.frequency)
        series_name = "generated_sine"

    train_signal, test_signal = split_series(series, test_ratio=args.test_ratio)
    encoder = TimeSeriesCurrentEncoder()
    reservoir = LiquidReservoir(
        n_neurons=max(8, args.neurons),
        p_connect=max(0.01, min(0.9, args.connectivity)),
        readout_decay=max(0.0, min(0.999, args.readout_decay)),
        enable_force_readout=True,
        force_output_dim=1,
        force_alpha=max(1e-6, args.force_alpha),
        force_forgetting_factor=max(1e-6, min(1.0, args.forgetting_factor)),
    )

    baseline_metrics = evaluate_force_sequence(
        reservoir,
        encoder,
        test_signal,
        reset_readout=True,
    )
    print(
        f"Baseline | test_mae={baseline_metrics['mae']:.4f} "
        f"test_rmse={baseline_metrics['rmse']:.4f}"
    )

    history = train_force_sequence(
        reservoir,
        encoder,
        train_signal,
        epochs=max(1, args.epochs),
        report_every=max(1, args.report_every),
    )
    final_metrics = evaluate_force_sequence(reservoir, encoder, test_signal)
    print(
        f"Final    | test_mae={final_metrics['mae']:.4f} "
        f"test_rmse={final_metrics['rmse']:.4f}"
    )

    sample_count = min(8, len(final_metrics["predictions"]))
    if sample_count > 0:
        print("\nSample predictions:")
        for index in range(sample_count):
            print(
                f"step={index:02d} "
                f"target={final_metrics['targets'][index]:+.4f} "
                f"prediction={final_metrics['predictions'][index]:+.4f}"
            )

    metadata = {
        "series_name": series_name,
        "series_length": len(series),
        "train_length": len(train_signal),
        "test_length": len(test_signal),
        "epochs": max(1, args.epochs),
        "baseline_test_mae": baseline_metrics["mae"],
        "baseline_test_rmse": baseline_metrics["rmse"],
        "final_test_mae": final_metrics["mae"],
        "final_test_rmse": final_metrics["rmse"],
        "history": history,
    }
    artifact_path, report_path = export_force_artifact(
        artifact_path=os.path.join(args.model_dir, "force_run.json"),
        report_path=args.report_path,
        reservoir=reservoir,
        encoder=encoder,
        metadata=metadata,
    )
    print(f"\nSaved report: {report_path}")
    print(f"Saved artifact: {artifact_path}")


if __name__ == "__main__":
    main()
