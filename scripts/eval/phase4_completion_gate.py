# Directory Path: scripts/eval/phase4_completion_gate.py
# English Title: Phase 4 Completion Gate
# Purpose/Content: Validates that Phase 4 scale-out and continual-learning criteria are satisfied using managed benchmark artifacts.

import argparse
import json
import os
import sys
from typing import Any, Dict, List


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


DEFAULT_PHASE3_REPORT_PATH = os.path.join(PROJECT_ROOT, "workspace", "evaluation", "phase3_accuracy_suite.json")
DEFAULT_PHASE4_REPORT_PATH = os.path.join(PROJECT_ROOT, "workspace", "evaluation", "phase4_scale_continual_benchmark.json")
PHASE4_REQUIRED_METRICS = [
    "structural_plasticity_stability",
    "hippocampal_transfer_integrity",
    "scale_out_retention_integrity",
    "continual_drift_recovery_integrity",
]
PHASE4_QUALITY_THRESHOLDS = {
    "structural_synapse_ratio_min": 0.45,
    "structural_synapse_ratio_max": 1.60,
    "hippocampal_after_top_score_min": 0.10,
    "hippocampal_score_retention_ratio_min": 0.85,
    "scale_out_retention_rate_min": 0.99,
    "scale_out_average_query_ms_max": 30.0,
}


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Report is not a JSON object.")
    return payload


def validate_phase4_completion(phase3_report: Dict[str, Any], phase4_report: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    phase3_completion = phase3_report.get("phase3_completion", {})
    if not isinstance(phase3_completion, dict) or not bool(phase3_completion.get("passed", False)):
        errors.append("Phase 3 completion gate is not passed.")

    if str(phase4_report.get("evaluator_name", "")) != "Phase4ScaleContinualBenchmark":
        errors.append("Phase 4 benchmark report has an unexpected evaluator name.")
    if not bool(phase4_report.get("passed", False)):
        errors.append("Phase 4 benchmark did not pass.")

    metrics = phase4_report.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    for metric_name in PHASE4_REQUIRED_METRICS:
        metric_value = float(metrics.get(metric_name, 0.0) or 0.0)
        if metric_value < 1.0:
            errors.append(
                f"Phase 4 required metric '{metric_name}' did not satisfy the minimum threshold (value={metric_value:.3f}, required>=1.000)."
            )
    threshold_results = phase4_report.get("threshold_results", {})
    if isinstance(threshold_results, dict):
        failed_thresholds = sorted(name for name, passed in threshold_results.items() if not bool(passed))
        if failed_thresholds:
            errors.append("Phase 4 threshold_results contains failed checks: " + ", ".join(failed_thresholds))

    quality_metrics = phase4_report.get("quality_metrics", {})
    if "quality_metrics" not in phase4_report or not isinstance(quality_metrics, dict):
        errors.append("Phase 4 benchmark report is missing quality_metrics.")
        quality_metrics = {}
    structural_ratio = float(quality_metrics.get("structural_synapse_ratio", 0.0) or 0.0)
    if not (
        PHASE4_QUALITY_THRESHOLDS["structural_synapse_ratio_min"]
        <= structural_ratio
        <= PHASE4_QUALITY_THRESHOLDS["structural_synapse_ratio_max"]
    ):
        errors.append(
            "Phase 4 structural synapse ratio is outside the stability window "
            f"(value={structural_ratio:.3f}, required="
            f"{PHASE4_QUALITY_THRESHOLDS['structural_synapse_ratio_min']:.2f}-"
            f"{PHASE4_QUALITY_THRESHOLDS['structural_synapse_ratio_max']:.2f})."
        )
    if float(quality_metrics.get("structural_per_context_non_empty", 0.0) or 0.0) < 1.0:
        errors.append("Phase 4 structural plasticity has an empty context compartment.")
    hippocampal_score = float(quality_metrics.get("hippocampal_after_top_score", 0.0) or 0.0)
    if hippocampal_score < PHASE4_QUALITY_THRESHOLDS["hippocampal_after_top_score_min"]:
        errors.append(
            f"Phase 4 hippocampal transfer score is too low (value={hippocampal_score:.3f}, required>=0.100)."
        )
    hippocampal_retention = float(quality_metrics.get("hippocampal_score_retention_ratio", 0.0) or 0.0)
    if hippocampal_retention < PHASE4_QUALITY_THRESHOLDS["hippocampal_score_retention_ratio_min"]:
        errors.append(
            f"Phase 4 hippocampal transfer retention ratio is too low (value={hippocampal_retention:.3f}, required>=0.850)."
        )
    retention_rate = float(quality_metrics.get("scale_out_retention_rate", 0.0) or 0.0)
    if retention_rate < PHASE4_QUALITY_THRESHOLDS["scale_out_retention_rate_min"]:
        errors.append(
            f"Phase 4 scale-out retention rate is too low (value={retention_rate:.3f}, required>=0.990)."
        )
    query_ms = float(quality_metrics.get("scale_out_average_query_ms", 999999.0) or 999999.0)
    if query_ms > PHASE4_QUALITY_THRESHOLDS["scale_out_average_query_ms_max"]:
        errors.append(
            f"Phase 4 scale-out average query latency is too high (value={query_ms:.3f}ms, required<=30.000ms)."
        )
    if float(quality_metrics.get("continual_baseline_recovered", 0.0) or 0.0) < 1.0:
        errors.append("Phase 4 continual recovery did not restore the baseline anchor prediction.")
    if float(quality_metrics.get("continual_drift_observed", 0.0) or 0.0) < 1.0:
        errors.append("Phase 4 continual drift scenario was not observed before recovery.")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 4 completion gate from managed Phase 3/Phase 4 reports.")
    parser.add_argument(
        "--phase3-report-path",
        default=DEFAULT_PHASE3_REPORT_PATH,
        help="Managed path to the phase3 accuracy suite report.",
    )
    parser.add_argument(
        "--phase4-report-path",
        default=DEFAULT_PHASE4_REPORT_PATH,
        help="Managed path to the phase4 scale-out benchmark report.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.phase3_report_path):
        print(f"Phase 4 completion gate failed: Phase 3 report not found at {args.phase3_report_path}")
        return 1
    if not os.path.exists(args.phase4_report_path):
        print(f"Phase 4 completion gate failed: Phase 4 report not found at {args.phase4_report_path}")
        return 1

    try:
        phase3_report = _load_json(args.phase3_report_path)
        phase4_report = _load_json(args.phase4_report_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Phase 4 completion gate failed: {exc}")
        return 1

    errors = validate_phase4_completion(phase3_report, phase4_report)
    if errors:
        print("Phase 4 completion gate failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    overall_score = float(phase4_report.get("overall_score", 0.0) or 0.0)
    print("Phase 4 completion gate passed.")
    print(f"phase4_overall_score={overall_score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
