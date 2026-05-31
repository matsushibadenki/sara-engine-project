# Directory Path: scripts/eval/phase5_entry_gate.py
# English Title: Phase 5 Entry Gate
# Purpose/Content: Validates the Phase 5 predictive-coding entry benchmark report before deeper H-JEPA work.

import argparse
import importlib.util
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

def _load_module_from_path(module_name: str, path: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from path: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_project_paths_helpers() -> tuple[Any, Any]:
    module_path = os.path.join(PROJECT_ROOT, "src", "sara_engine", "utils", "project_paths.py")
    module = _load_module_from_path("sara_project_paths", module_path)
    ensure_parent = getattr(module, "ensure_parent_directory", None)
    workspace = getattr(module, "workspace_path", None)
    if not callable(ensure_parent) or not callable(workspace):
        raise RuntimeError("project_paths helper is missing required callables.")
    return ensure_parent, workspace


def _load_phase5_required_metrics() -> List[str]:
    module_path = os.path.join(PROJECT_ROOT, "src", "sara_engine", "evaluation", "phase5_contract.py")
    module = _load_module_from_path("sara_eval_phase5_contract", module_path)
    return list(getattr(module, "PHASE5_ENTRY_METRIC_NAMES"))


ensure_parent_directory, workspace_path = _load_project_paths_helpers()
PHASE5_REQUIRED_METRICS = _load_phase5_required_metrics()


DEFAULT_PHASE5_REPORT_PATH = workspace_path("evaluation", "phase5_predictive_coding_benchmark.json")
DEFAULT_OUTPUT_REPORT_PATH = workspace_path("evaluation", "phase5_entry_gate_report.json")
DEFAULT_OUTPUT_SUMMARY_PATH = workspace_path("evaluation", "phase5_entry_gate_summary.txt")


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Report is not a JSON object.")
    return payload


def validate_phase5_entry(report: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if str(report.get("suite_name", "")) != "Phase5PredictiveCodingBenchmark":
        errors.append("Phase 5 predictive coding report has an unexpected suite name.")
    if not bool(report.get("passed", False)):
        errors.append("Phase 5 predictive coding benchmark did not pass.")

    metrics = report.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
        errors.append("Phase 5 predictive coding report is missing metrics.")
    for metric_name in PHASE5_REQUIRED_METRICS:
        metric_value = float(metrics.get(metric_name, 0.0) or 0.0)
        if metric_value < 1.0:
            errors.append(
                f"Phase 5 required metric '{metric_name}' did not satisfy the minimum threshold "
                f"(value={metric_value:.3f}, required>=1.000)."
            )

    threshold_results = report.get("threshold_results", {})
    if isinstance(threshold_results, dict):
        failed_thresholds = sorted(name for name, passed in threshold_results.items() if not bool(passed))
        if failed_thresholds:
            errors.append("Phase 5 threshold_results contains failed checks: " + ", ".join(failed_thresholds))
    else:
        errors.append("Phase 5 predictive coding report is missing threshold_results.")

    details = report.get("details", {})
    if not isinstance(details, dict):
        details = {}
    primary = details.get("primary_transition", {})
    multi_step = details.get("multi_step_trace", {})
    branch_comparison = details.get("branch_comparison", {})
    if not isinstance(primary, dict) or not bool(primary.get("trace_complete", False)):
        errors.append("Phase 5 primary latent transition trace is incomplete.")
    if not isinstance(multi_step, dict) or not bool(multi_step.get("trace_complete", False)):
        errors.append("Phase 5 multi-step latent transition trace is incomplete.")
    if not isinstance(branch_comparison, dict) or not bool(branch_comparison.get("separable", False)):
        errors.append("Phase 5 counterfactual latent transition is not separable.")
    return errors


def build_phase5_entry_gate_report(report: Dict[str, Any]) -> Dict[str, Any]:
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    threshold_results = (
        report.get("threshold_results", {})
        if isinstance(report.get("threshold_results"), dict)
        else {}
    )
    details = report.get("details", {}) if isinstance(report.get("details"), dict) else {}
    primary = details.get("primary_transition", {}) if isinstance(details.get("primary_transition"), dict) else {}
    multi_step = details.get("multi_step_trace", {}) if isinstance(details.get("multi_step_trace"), dict) else {}
    branch_comparison = (
        details.get("branch_comparison", {})
        if isinstance(details.get("branch_comparison"), dict)
        else {}
    )

    checks: Dict[str, Dict[str, Any]] = {
        "suite_name": {
            "passed": str(report.get("suite_name", "")) == "Phase5PredictiveCodingBenchmark",
            "details": {"suite_name": str(report.get("suite_name", ""))},
        },
        "benchmark_passed": {
            "passed": bool(report.get("passed", False)),
            "details": {"passed": bool(report.get("passed", False))},
        },
        "metrics_present": {
            "passed": isinstance(report.get("metrics", {}), dict),
            "details": {"metric_count": len(metrics)},
        },
        "thresholds_present": {
            "passed": isinstance(report.get("threshold_results", {}), dict),
            "details": {"threshold_count": len(threshold_results)},
        },
        "primary_trace_complete": {
            "passed": bool(primary.get("trace_complete", False)),
            "details": {"trace_complete": bool(primary.get("trace_complete", False))},
        },
        "counterfactual_branch_separable": {
            "passed": bool(branch_comparison.get("separable", False)),
            "details": {"separable": bool(branch_comparison.get("separable", False))},
        },
        "multi_step_trace_complete": {
            "passed": bool(multi_step.get("trace_complete", False)),
            "details": {
                "trace_complete": bool(multi_step.get("trace_complete", False)),
                "step_count": int(multi_step.get("step_count", 0) or 0),
            },
        },
    }
    for metric_name in PHASE5_REQUIRED_METRICS:
        metric_value = float(metrics.get(metric_name, 0.0) or 0.0)
        threshold_passed = bool(threshold_results.get(metric_name, False))
        checks[f"metric.{metric_name}"] = {
            "passed": metric_value >= 1.0,
            "details": {"value": metric_value, "required_min": 1.0},
        }
        checks[f"threshold.{metric_name}"] = {
            "passed": threshold_passed,
            "details": {"threshold_passed": threshold_passed},
        }

    failed_checks = [name for name, check in checks.items() if not bool(check.get("passed", False))]
    errors = validate_phase5_entry(report)
    return {
        "suite_name": "Phase5EntryGate",
        "passed": len(failed_checks) == 0 and not errors,
        "failed_checks": failed_checks,
        "error_count": len(errors),
        "errors": errors,
        "check_count": len(checks),
        "pass_count": len(checks) - len(failed_checks),
        "phase5_overall_score": float(report.get("overall_score", 0.0) or 0.0),
        "checks": checks,
    }


def format_phase5_entry_gate_summary(gate_report: Dict[str, Any]) -> str:
    checks = gate_report.get("checks", {}) if isinstance(gate_report.get("checks"), dict) else {}
    lines = [
        "SARA Engine Phase 5 Entry Gate Summary",
        f"- gate_status: {'PASS' if bool(gate_report.get('passed', False)) else 'FAIL'}",
        f"- pass_count: {int(gate_report.get('pass_count', 0))}/{int(gate_report.get('check_count', 0))}",
        f"- failed_check_count: {len(gate_report.get('failed_checks', [])) if isinstance(gate_report.get('failed_checks', []), list) else 0}",
        f"- phase5_overall_score: {float(gate_report.get('phase5_overall_score', 0.0) or 0.0):.3f}",
    ]
    for check_name, check_data in checks.items():
        if isinstance(check_data, dict):
            lines.append(f"- {check_name}: {'PASS' if bool(check_data.get('passed', False)) else 'FAIL'}")
    for failed in gate_report.get("failed_checks", []) if isinstance(gate_report.get("failed_checks", []), list) else []:
        lines.append(f"  failed: {failed}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 5 entry gate from the managed predictive-coding report.")
    parser.add_argument(
        "--report-path",
        default=DEFAULT_PHASE5_REPORT_PATH,
        help="Managed path to the phase5 predictive coding benchmark report.",
    )
    parser.add_argument(
        "--output-report-path",
        default=DEFAULT_OUTPUT_REPORT_PATH,
        help="Managed output path for the Phase 5 entry gate report.",
    )
    parser.add_argument(
        "--output-summary-path",
        default=DEFAULT_OUTPUT_SUMMARY_PATH,
        help="Managed output path for the Phase 5 entry gate summary.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.report_path):
        print(f"Phase 5 entry gate failed: report not found at {args.report_path}")
        return 1

    try:
        report = _load_json(args.report_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Phase 5 entry gate failed: {exc}")
        return 1

    gate_report = build_phase5_entry_gate_report(report)
    output_report_path = ensure_parent_directory(args.output_report_path)
    output_summary_path = ensure_parent_directory(args.output_summary_path)
    with open(output_report_path, "w", encoding="utf-8") as handle:
        json.dump(gate_report, handle, indent=2, ensure_ascii=False)
    with open(output_summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_phase5_entry_gate_summary(gate_report))

    errors = validate_phase5_entry(report)
    if errors:
        print("Phase 5 entry gate failed:")
        for error in errors:
            print(f"- {error}")
        print(f"Saved report: {output_report_path}")
        print(f"Saved summary: {output_summary_path}")
        return 1

    overall_score = float(report.get("overall_score", 0.0) or 0.0)
    print("Phase 5 entry gate passed.")
    print(f"phase5_overall_score={overall_score:.3f}")
    print(f"Saved report: {output_report_path}")
    print(f"Saved summary: {output_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
