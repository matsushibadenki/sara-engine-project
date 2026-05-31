# Directory Path: scripts/train/resume_force_cli.py
# English Title: FORCE Reservoir Resume Training CLI
# Purpose/Content: Restores a saved force_run.json artifact, continues FORCE learning on new or generated time-series data, and saves updated artifacts under managed directories.

import argparse
import os
import sys
from typing import Any, Dict


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from sara_engine.learning.force_io import export_force_artifact, load_force_artifact
from sara_engine.learning.force_workflow import (
    build_sine_series,
    evaluate_force_sequence,
    load_series,
    split_series,
    train_force_sequence,
)
from sara_engine.utils.project_paths import model_path, workspace_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume FORCE learning from a saved force_run.json artifact."
    )
    parser.add_argument(
        "--artifact-path",
        default=model_path("force_time_series", "force_run.json"),
        help="Path to an existing force_run.json artifact.",
    )
    parser.add_argument(
        "--series-path",
        default="",
        help="Optional path to a JSON or text time-series file. If omitted, a sine wave is generated.",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Number of additional training epochs.")
    parser.add_argument("--report-every", type=int, default=4, help="Epoch interval for progress reports.")
    parser.add_argument("--length", type=int, default=320, help="Generated sine series length when --series-path is omitted.")
    parser.add_argument("--frequency", type=float, default=0.09, help="Generated sine frequency.")
    parser.add_argument("--test-ratio", type=float, default=0.3, help="Holdout ratio for evaluation.")
    parser.add_argument(
        "--report-path",
        default=workspace_path("force_training", "resume_report.json"),
        help="Managed output path for resumed training metrics JSON.",
    )
    parser.add_argument(
        "--model-dir",
        default=model_path("force_time_series_resumed"),
        help="Managed output directory for the resumed artifact.",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("FORCE Reservoir Resume Training CLI")
    print("=" * 64)

    reservoir, encoder, metadata = load_force_artifact(args.artifact_path)
    original_epochs = int(metadata.get("epochs", 0))
    print(f"Loaded artifact: {args.artifact_path}")
    print(f"Previous epochs: {original_epochs}")
    print(f"Original series: {metadata.get('series_name', 'unknown')}")

    if args.series_path:
        series = load_series(args.series_path)
        series_name = os.path.basename(args.series_path)
    else:
        series = build_sine_series(length=max(16, args.length), frequency=args.frequency)
        series_name = "generated_sine"

    train_signal, test_signal = split_series(series, test_ratio=args.test_ratio)

    baseline_metrics = evaluate_force_sequence(
        reservoir,
        encoder,
        test_signal,
        reset_readout=False,
    )
    print(
        f"Resume baseline | test_mae={baseline_metrics['mae']:.4f} "
        f"test_rmse={baseline_metrics['rmse']:.4f}"
    )

    history = train_force_sequence(
        reservoir,
        encoder,
        train_signal,
        epochs=max(1, args.epochs),
        report_every=max(1, args.report_every),
    )
    final_metrics = evaluate_force_sequence(
        reservoir,
        encoder,
        test_signal,
        reset_readout=False,
    )
    print(
        f"Resumed final   | test_mae={final_metrics['mae']:.4f} "
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

    previous_history = metadata.get("history", [])
    if not isinstance(previous_history, list):
        previous_history = []

    resumed_metadata: Dict[str, Any] = {
        "series_name": series_name,
        "source_artifact": args.artifact_path,
        "series_length": len(series),
        "train_length": len(train_signal),
        "test_length": len(test_signal),
        "epochs": original_epochs + max(1, args.epochs),
        "resume_epochs": max(1, args.epochs),
        "baseline_test_mae": baseline_metrics["mae"],
        "baseline_test_rmse": baseline_metrics["rmse"],
        "final_test_mae": final_metrics["mae"],
        "final_test_rmse": final_metrics["rmse"],
        "history": previous_history + history,
    }

    artifact_path, report_path = export_force_artifact(
        artifact_path=os.path.join(args.model_dir, "force_run.json"),
        report_path=args.report_path,
        reservoir=reservoir,
        encoder=encoder,
        metadata=resumed_metadata,
    )
    print(f"\nSaved resumed report: {report_path}")
    print(f"Saved resumed artifact: {artifact_path}")


if __name__ == "__main__":
    main()
