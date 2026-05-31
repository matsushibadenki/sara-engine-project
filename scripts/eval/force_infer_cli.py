# Directory Path: scripts/eval/force_infer_cli.py
# English Title: FORCE Reservoir Inference CLI
# Purpose/Content: Loads a saved force_run.json artifact, runs inference on a time-series, and stores evaluation metrics under workspace/.

import argparse
import json
import os
import sys
from typing import Any, Dict, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from sara_engine.learning.force_io import load_force_artifact
from sara_engine.learning.force_workflow import (
    build_sine_series,
    evaluate_force_sequence,
    load_series,
)
from sara_engine.utils.project_paths import ensure_parent_directory, model_path, workspace_path


def run_inference(
    signal: Sequence[float],
    artifact_path: str,
) -> Dict[str, Any]:
    reservoir, encoder, metadata = load_force_artifact(artifact_path)
    metrics = evaluate_force_sequence(
        reservoir,
        encoder,
        signal,
        reset_readout=False,
    )
    return {
        "metadata": metadata,
        "predictions": metrics["predictions"],
        "targets": metrics["targets"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
    }


def resolve_artifact_path(
    artifact_path: str,
    model_dir: str,
    artifact_variant: str,
) -> str:
    if artifact_path:
        return artifact_path

    if artifact_variant == "auto":
        for candidate in (
            "best_force_run.json",
            "latest_force_run.json",
            "force_run.json",
        ):
            candidate_path = os.path.join(model_dir, candidate)
            if os.path.exists(candidate_path):
                return candidate_path
        return os.path.join(model_dir, "force_run.json")

    variant_to_name = {
        "standard": "force_run.json",
        "latest": "latest_force_run.json",
        "best": "best_force_run.json",
    }
    filename = variant_to_name.get(artifact_variant, "force_run.json")
    return os.path.join(model_dir, filename)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with a saved FORCE reservoir artifact."
    )
    parser.add_argument(
        "--artifact-path",
        default="",
        help="Explicit path to a saved FORCE artifact. Overrides --model-dir and --artifact-variant.",
    )
    parser.add_argument(
        "--model-dir",
        default=model_path("force_time_series"),
        help="Model directory used when resolving --artifact-variant automatically.",
    )
    parser.add_argument(
        "--artifact-variant",
        choices=("auto", "standard", "latest", "best"),
        default="auto",
        help="Which artifact file to load from --model-dir when --artifact-path is omitted.",
    )
    parser.add_argument(
        "--series-path",
        default="",
        help="Optional path to a JSON or text time-series file. If omitted, a sine wave is generated.",
    )
    parser.add_argument("--length", type=int, default=120, help="Generated sine series length when --series-path is omitted.")
    parser.add_argument("--frequency", type=float, default=0.09, help="Generated sine frequency.")
    parser.add_argument(
        "--report-path",
        default=workspace_path("force_inference", "latest_report.json"),
        help="Managed output path for inference metrics JSON.",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("FORCE Reservoir Inference CLI")
    print("=" * 64)

    if args.series_path:
        signal = load_series(args.series_path)
        signal_name = os.path.basename(args.series_path)
    else:
        signal = build_sine_series(length=max(16, args.length), frequency=args.frequency)
        signal_name = "generated_sine"

    resolved_artifact_path = resolve_artifact_path(
        artifact_path=args.artifact_path,
        model_dir=args.model_dir,
        artifact_variant=args.artifact_variant,
    )
    results = run_inference(signal, artifact_path=resolved_artifact_path)
    trained_on = results["metadata"].get("series_name", "unknown")
    print(f"Artifact: {resolved_artifact_path}")
    print(f"Trained on: {trained_on}")
    print(f"Inference series: {signal_name}")
    print(f"MAE={results['mae']:.4f} RMSE={results['rmse']:.4f}")

    sample_count = min(8, len(results["predictions"]))
    if sample_count > 0:
        print("\nSample predictions:")
        for index in range(sample_count):
            print(
                f"step={index:02d} "
                f"target={results['targets'][index]:+.4f} "
                f"prediction={results['predictions'][index]:+.4f}"
            )

    report_path = ensure_parent_directory(args.report_path)
    report_payload = {
        "artifact_path": resolved_artifact_path,
        "signal_name": signal_name,
        "mae": results["mae"],
        "rmse": results["rmse"],
        "metadata": results["metadata"],
        "predictions_preview": results["predictions"][:16],
        "targets_preview": results["targets"][:16],
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
