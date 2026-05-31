# Directory Path: scripts/eval/phase5_completion_gate.py
# English Title: Phase 5 Completion Gate
# Purpose/Content: Validates that Phase 5 predictive-coding completion criteria are satisfied using managed Phase 4/Phase 5 artifacts.

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
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "workspace", "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(PROJECT_ROOT, "workspace", "cache"))

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
PHASE5_ENTRY_METRIC_NAMES = _load_phase5_required_metrics()


DEFAULT_PHASE4_REPORT_PATH = workspace_path("evaluation", "phase4_scale_continual_benchmark.json")
DEFAULT_PHASE5_REPORT_PATH = workspace_path("evaluation", "phase5_predictive_coding_benchmark.json")
DEFAULT_PHASE5_ENTRY_GATE_REPORT_PATH = workspace_path("evaluation", "phase5_entry_gate_report.json")
DEFAULT_SPARSE_DIFFUSION_BLOCK_REPORT_PATH = workspace_path("evaluation", "sparse_diffusion_block_readiness.json")
DEFAULT_OUTPUT_REPORT_PATH = workspace_path("evaluation", "phase5_completion_gate_report.json")
DEFAULT_OUTPUT_SUMMARY_PATH = workspace_path("evaluation", "phase5_completion_gate_summary.txt")

PHASE5_REQUIRED_METRICS = PHASE5_ENTRY_METRIC_NAMES
SPARSE_DIFFUSION_REQUIRED_METRICS = {
    "sparse_diffusion_partition_integrity": 1.0,
    "sparse_diffusion_independent_block_integrity": 1.0,
    "sparse_diffusion_denoise_accuracy": 1.0,
    "sparse_diffusion_event_cost_advantage": 2.0,
    "sparse_diffusion_block_ablation_integrity": 1.0,
    "sparse_diffusion_single_pass_recurrent_integrity": 1.0,
    "sparse_diffusion_policy_compatibility": 1.0,
}


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Report is not a JSON object.")
    return payload


def validate_phase5_completion(
    phase4_report: Dict[str, Any],
    phase5_report: Dict[str, Any],
    phase5_entry_gate_report: Dict[str, Any],
    sparse_diffusion_block_report: Dict[str, Any],
) -> List[str]:
    errors: List[str] = []

    if str(phase4_report.get("evaluator_name", "")) != "Phase4ScaleContinualBenchmark":
        errors.append("Phase 4 benchmark report has an unexpected evaluator name.")
    if not bool(phase4_report.get("passed", False)):
        errors.append("Phase 4 completion prerequisite is not passed.")

    if str(phase5_report.get("suite_name", "")) != "Phase5PredictiveCodingBenchmark":
        errors.append("Phase 5 predictive coding report has an unexpected suite name.")
    if not bool(phase5_report.get("passed", False)):
        errors.append("Phase 5 predictive coding benchmark did not pass.")
    if float(phase5_report.get("overall_score", 0.0) or 0.0) < 1.0:
        errors.append(
            "Phase 5 overall score is below completion threshold "
            f"(value={float(phase5_report.get('overall_score', 0.0) or 0.0):.3f}, required>=1.000)."
        )

    metrics = phase5_report.get("metrics", {})
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

    threshold_results = phase5_report.get("threshold_results", {})
    if isinstance(threshold_results, dict):
        failed_thresholds = sorted(name for name, passed in threshold_results.items() if not bool(passed))
        if failed_thresholds:
            errors.append("Phase 5 threshold_results contains failed checks: " + ", ".join(failed_thresholds))
    else:
        errors.append("Phase 5 predictive coding report is missing threshold_results.")

    details = phase5_report.get("details", {})
    if not isinstance(details, dict):
        details = {}
    primary = details.get("primary_transition", {}) if isinstance(details.get("primary_transition"), dict) else {}
    multi_step = details.get("multi_step_trace", {}) if isinstance(details.get("multi_step_trace"), dict) else {}
    branch_comparison = (
        details.get("branch_comparison", {})
        if isinstance(details.get("branch_comparison"), dict)
        else {}
    )

    if not bool(primary.get("trace_complete", False)):
        errors.append("Phase 5 primary latent transition trace is incomplete.")
    if float(primary.get("alignment_ratio", 0.0) or 0.0) < 1.0:
        errors.append(
            "Phase 5 primary alignment ratio is below completion threshold "
            f"(value={float(primary.get('alignment_ratio', 0.0) or 0.0):.3f}, required>=1.000)."
        )
    prediction_error_count = (
        len(primary.get("prediction_error_ids", []))
        if isinstance(primary.get("prediction_error_ids"), list)
        else 0
    )
    if prediction_error_count <= 0:
        errors.append("Phase 5 primary trace has no observable prediction_error_ids.")
    if not bool(primary.get("correction_coverage", False)):
        errors.append("Phase 5 primary trace does not cover prediction errors with corrections.")
    if not bool(primary.get("anti_collapse_diversity", False)):
        errors.append("Phase 5 primary trace failed anti-collapse diversity check.")

    if not bool(multi_step.get("trace_complete", False)):
        errors.append("Phase 5 multi-step latent transition trace is incomplete.")
    step_count = int(multi_step.get("step_count", 0) or 0)
    if step_count < 2:
        errors.append(f"Phase 5 multi-step trace is too short (value={step_count}, required>=2).")
    complete_steps = int(multi_step.get("complete_steps", 0) or 0)
    if complete_steps != step_count:
        errors.append(
            "Phase 5 multi-step trace contains incomplete steps "
            f"(complete_steps={complete_steps}, step_count={step_count})."
        )
    coverage_steps = int(multi_step.get("correction_coverage_steps", 0) or 0)
    if coverage_steps != step_count:
        errors.append(
            "Phase 5 multi-step trace correction coverage is incomplete "
            f"(coverage_steps={coverage_steps}, step_count={step_count})."
        )
    total_errors = int(multi_step.get("total_prediction_errors", 0) or 0)
    total_corrections = int(multi_step.get("total_corrections", 0) or 0)
    if total_corrections < total_errors:
        errors.append(
            "Phase 5 multi-step corrections are insufficient "
            f"(total_corrections={total_corrections}, total_prediction_errors={total_errors})."
        )

    if not bool(branch_comparison.get("separable", False)):
        errors.append("Phase 5 counterfactual latent transition is not separable.")

    macro_step_reduction = float(details.get("macro_step_reduction", 0.0) or 0.0)
    if macro_step_reduction < 2.0:
        errors.append(
            "Phase 5 macro step reduction is below completion threshold "
            f"(value={macro_step_reduction:.3f}, required>=2.000)."
        )
    macro_cost_reduction = float(details.get("macro_cost_reduction", 0.0) or 0.0)
    if macro_cost_reduction < 0.30:
        errors.append(
            "Phase 5 macro cost reduction is below completion threshold "
            f"(value={macro_cost_reduction:.3f}, required>=0.300)."
        )
    subgoal_coverage_ratio = float(details.get("subgoal_coverage_ratio", 0.0) or 0.0)
    if subgoal_coverage_ratio < 1.0:
        errors.append(
            "Phase 5 subgoal coverage ratio is below completion threshold "
            f"(value={subgoal_coverage_ratio:.3f}, required>=1.000)."
        )
    micro_es = details.get("micro_es_refinement", {}) if isinstance(details.get("micro_es_refinement"), dict) else {}
    if not bool(micro_es.get("low_rank_trace_complete", False)):
        errors.append("Phase 5 micro-ES low-rank refinement trace is incomplete.")
    micro_es_fitness_improvement = float(micro_es.get("fitness_improvement", 0.0) or 0.0)
    if micro_es_fitness_improvement <= 0.05:
        errors.append(
            "Phase 5 micro-ES fitness improvement is below completion threshold "
            f"(value={micro_es_fitness_improvement:.3f}, required>0.050)."
        )
    micro_es_event_cost_reduction = float(micro_es.get("event_cost_reduction", 0.0) or 0.0)
    if micro_es_event_cost_reduction < 0.04:
        errors.append(
            "Phase 5 micro-ES event cost reduction is below completion threshold "
            f"(value={micro_es_event_cost_reduction:.3f}, required>=0.040)."
        )
    micro_es_population_cost = float(micro_es.get("population_event_cost_proxy", 0.0) or 0.0)
    micro_es_event_budget = float(micro_es.get("event_budget", 0.0) or 0.0)
    if micro_es_population_cost > micro_es_event_budget:
        errors.append(
            "Phase 5 micro-ES population event cost exceeds its event budget "
            f"(value={micro_es_population_cost:.3f}, budget={micro_es_event_budget:.3f})."
        )

    if str(phase5_entry_gate_report.get("suite_name", "")) != "Phase5EntryGate":
        errors.append("Phase 5 entry gate report has an unexpected suite name.")
    if not bool(phase5_entry_gate_report.get("passed", False)):
        errors.append("Phase 5 entry gate is not passed.")
    failed_checks = (
        phase5_entry_gate_report.get("failed_checks", [])
        if isinstance(phase5_entry_gate_report.get("failed_checks"), list)
        else []
    )
    if failed_checks:
        errors.append("Phase 5 entry gate report contains failed checks: " + ", ".join(str(item) for item in failed_checks))

    if str(sparse_diffusion_block_report.get("suite_name", "")) != "SparseDiffusionBlockReadiness":
        errors.append("Sparse diffusion block readiness report has an unexpected suite name.")
    if not bool(sparse_diffusion_block_report.get("passed", False)):
        errors.append("Sparse diffusion block readiness did not pass.")
    sparse_metrics = (
        sparse_diffusion_block_report.get("metrics", {})
        if isinstance(sparse_diffusion_block_report.get("metrics"), dict)
        else {}
    )
    for metric_name, required_min in SPARSE_DIFFUSION_REQUIRED_METRICS.items():
        metric_value = float(sparse_metrics.get(metric_name, 0.0) or 0.0)
        if metric_value < required_min:
            errors.append(
                f"Sparse diffusion block metric '{metric_name}' did not satisfy the minimum threshold "
                f"(value={metric_value:.3f}, required>={required_min:.3f})."
            )

    return errors


def build_phase5_completion_gate_report(
    phase4_report: Dict[str, Any],
    phase5_report: Dict[str, Any],
    phase5_entry_gate_report: Dict[str, Any],
    sparse_diffusion_block_report: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = phase5_report.get("metrics", {}) if isinstance(phase5_report.get("metrics"), dict) else {}
    threshold_results = (
        phase5_report.get("threshold_results", {})
        if isinstance(phase5_report.get("threshold_results"), dict)
        else {}
    )
    details = phase5_report.get("details", {}) if isinstance(phase5_report.get("details"), dict) else {}
    primary = details.get("primary_transition", {}) if isinstance(details.get("primary_transition"), dict) else {}
    multi_step = details.get("multi_step_trace", {}) if isinstance(details.get("multi_step_trace"), dict) else {}
    branch_comparison = (
        details.get("branch_comparison", {})
        if isinstance(details.get("branch_comparison"), dict)
        else {}
    )
    micro_es = details.get("micro_es_refinement", {}) if isinstance(details.get("micro_es_refinement"), dict) else {}
    micro_es_population_cost = float(micro_es.get("population_event_cost_proxy", 0.0) or 0.0)
    micro_es_event_budget = float(micro_es.get("event_budget", 0.0) or 0.0)
    sparse_metrics = (
        sparse_diffusion_block_report.get("metrics", {})
        if isinstance(sparse_diffusion_block_report.get("metrics"), dict)
        else {}
    )

    checks: Dict[str, Dict[str, Any]] = {
        "phase4_prerequisite_passed": {
            "passed": bool(phase4_report.get("passed", False)),
            "details": {"passed": bool(phase4_report.get("passed", False))},
        },
        "phase5_suite_name": {
            "passed": str(phase5_report.get("suite_name", "")) == "Phase5PredictiveCodingBenchmark",
            "details": {"suite_name": str(phase5_report.get("suite_name", ""))},
        },
        "phase5_benchmark_passed": {
            "passed": bool(phase5_report.get("passed", False)),
            "details": {"passed": bool(phase5_report.get("passed", False))},
        },
        "phase5_overall_score": {
            "passed": float(phase5_report.get("overall_score", 0.0) or 0.0) >= 1.0,
            "details": {"value": float(phase5_report.get("overall_score", 0.0) or 0.0), "required_min": 1.0},
        },
        "phase5_entry_gate_passed": {
            "passed": bool(phase5_entry_gate_report.get("passed", False)),
            "details": {"passed": bool(phase5_entry_gate_report.get("passed", False))},
        },
        "primary_trace_complete": {
            "passed": bool(primary.get("trace_complete", False)),
            "details": {"trace_complete": bool(primary.get("trace_complete", False))},
        },
        "primary_alignment_ratio": {
            "passed": float(primary.get("alignment_ratio", 0.0) or 0.0) >= 1.0,
            "details": {"value": float(primary.get("alignment_ratio", 0.0) or 0.0), "required_min": 1.0},
        },
        "primary_prediction_error_observed": {
            "passed": isinstance(primary.get("prediction_error_ids"), list) and len(primary.get("prediction_error_ids", [])) > 0,
            "details": {"count": len(primary.get("prediction_error_ids", [])) if isinstance(primary.get("prediction_error_ids"), list) else 0},
        },
        "primary_correction_coverage": {
            "passed": bool(primary.get("correction_coverage", False)),
            "details": {"correction_coverage": bool(primary.get("correction_coverage", False))},
        },
        "primary_anti_collapse_diversity": {
            "passed": bool(primary.get("anti_collapse_diversity", False)),
            "details": {"anti_collapse_diversity": bool(primary.get("anti_collapse_diversity", False))},
        },
        "counterfactual_branch_separable": {
            "passed": bool(branch_comparison.get("separable", False)),
            "details": {"separable": bool(branch_comparison.get("separable", False))},
        },
        "multi_step_trace_complete": {
            "passed": bool(multi_step.get("trace_complete", False)),
            "details": {"trace_complete": bool(multi_step.get("trace_complete", False))},
        },
        "multi_step_step_count": {
            "passed": int(multi_step.get("step_count", 0) or 0) >= 2,
            "details": {"value": int(multi_step.get("step_count", 0) or 0), "required_min": 2},
        },
        "multi_step_complete_steps": {
            "passed": int(multi_step.get("complete_steps", 0) or 0) == int(multi_step.get("step_count", 0) or 0),
            "details": {
                "complete_steps": int(multi_step.get("complete_steps", 0) or 0),
                "step_count": int(multi_step.get("step_count", 0) or 0),
            },
        },
        "multi_step_correction_coverage_steps": {
            "passed": int(multi_step.get("correction_coverage_steps", 0) or 0) == int(multi_step.get("step_count", 0) or 0),
            "details": {
                "coverage_steps": int(multi_step.get("correction_coverage_steps", 0) or 0),
                "step_count": int(multi_step.get("step_count", 0) or 0),
            },
        },
        "multi_step_total_corrections": {
            "passed": int(multi_step.get("total_corrections", 0) or 0) >= int(multi_step.get("total_prediction_errors", 0) or 0),
            "details": {
                "total_corrections": int(multi_step.get("total_corrections", 0) or 0),
                "total_prediction_errors": int(multi_step.get("total_prediction_errors", 0) or 0),
            },
        },
        "macro_step_reduction": {
            "passed": float(details.get("macro_step_reduction", 0.0) or 0.0) >= 2.0,
            "details": {"value": float(details.get("macro_step_reduction", 0.0) or 0.0), "required_min": 2.0},
        },
        "macro_cost_reduction": {
            "passed": float(details.get("macro_cost_reduction", 0.0) or 0.0) >= 0.30,
            "details": {"value": float(details.get("macro_cost_reduction", 0.0) or 0.0), "required_min": 0.30},
        },
        "subgoal_coverage_ratio": {
            "passed": float(details.get("subgoal_coverage_ratio", 0.0) or 0.0) >= 1.0,
            "details": {"value": float(details.get("subgoal_coverage_ratio", 0.0) or 0.0), "required_min": 1.0},
        },
        "micro_es_low_rank_trace_complete": {
            "passed": bool(micro_es.get("low_rank_trace_complete", False)),
            "details": {"low_rank_trace_complete": bool(micro_es.get("low_rank_trace_complete", False))},
        },
        "micro_es_fitness_improvement": {
            "passed": float(micro_es.get("fitness_improvement", 0.0) or 0.0) > 0.05,
            "details": {"value": float(micro_es.get("fitness_improvement", 0.0) or 0.0), "required_gt": 0.05},
        },
        "micro_es_event_cost_reduction": {
            "passed": float(micro_es.get("event_cost_reduction", 0.0) or 0.0) >= 0.04,
            "details": {"value": float(micro_es.get("event_cost_reduction", 0.0) or 0.0), "required_min": 0.04},
        },
        "micro_es_population_event_budget": {
            "passed": micro_es_population_cost <= micro_es_event_budget,
            "details": {"value": micro_es_population_cost, "event_budget": micro_es_event_budget},
        },
        "sparse_diffusion_block_readiness_passed": {
            "passed": bool(sparse_diffusion_block_report.get("passed", False)),
            "details": {
                "passed": bool(sparse_diffusion_block_report.get("passed", False)),
                "overall_score": float(sparse_diffusion_block_report.get("overall_score", 0.0) or 0.0),
                "block_count": int(sparse_diffusion_block_report.get("block_count", 0) or 0),
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

    for metric_name, required_min in SPARSE_DIFFUSION_REQUIRED_METRICS.items():
        metric_value = float(sparse_metrics.get(metric_name, 0.0) or 0.0)
        checks[f"sparse_diffusion.{metric_name}"] = {
            "passed": metric_value >= required_min,
            "details": {"value": metric_value, "required_min": required_min},
        }

    failed_checks = [name for name, check in checks.items() if not bool(check.get("passed", False))]
    errors = validate_phase5_completion(phase4_report, phase5_report, phase5_entry_gate_report, sparse_diffusion_block_report)
    return {
        "suite_name": "Phase5CompletionGate",
        "passed": len(failed_checks) == 0 and not errors,
        "failed_checks": failed_checks,
        "error_count": len(errors),
        "errors": errors,
        "check_count": len(checks),
        "pass_count": len(checks) - len(failed_checks),
        "phase5_overall_score": float(phase5_report.get("overall_score", 0.0) or 0.0),
        "checks": checks,
    }


def format_phase5_completion_gate_summary(gate_report: Dict[str, Any]) -> str:
    checks = gate_report.get("checks", {}) if isinstance(gate_report.get("checks"), dict) else {}
    detail_check_names = [
        "macro_step_reduction",
        "macro_cost_reduction",
        "subgoal_coverage_ratio",
        "micro_es_fitness_improvement",
        "micro_es_event_cost_reduction",
        "micro_es_population_event_budget",
        "sparse_diffusion_block_readiness_passed",
    ]
    lines = [
        "SARA Engine Phase 5 Completion Gate Summary",
        f"- gate_status: {'PASS' if bool(gate_report.get('passed', False)) else 'FAIL'}",
        f"- pass_count: {int(gate_report.get('pass_count', 0))}/{int(gate_report.get('check_count', 0))}",
        f"- failed_check_count: {len(gate_report.get('failed_checks', [])) if isinstance(gate_report.get('failed_checks', []), list) else 0}",
        f"- phase5_overall_score: {float(gate_report.get('phase5_overall_score', 0.0) or 0.0):.3f}",
    ]
    for check_name, check_data in checks.items():
        if isinstance(check_data, dict):
            lines.append(f"- {check_name}: {'PASS' if bool(check_data.get('passed', False)) else 'FAIL'}")
    for check_name in detail_check_names:
        check_data = checks.get(check_name, {})
        if not isinstance(check_data, dict) or not isinstance(check_data.get("details"), dict):
            continue
        detail = check_data["details"]
        if "value" in detail:
            line = f"- {check_name}_value: {float(detail.get('value', 0.0) or 0.0):.3f}"
            if "required_min" in detail:
                line += f" required_min={float(detail.get('required_min', 0.0) or 0.0):.3f}"
            if "required_gt" in detail:
                line += f" required_gt={float(detail.get('required_gt', 0.0) or 0.0):.3f}"
            if "event_budget" in detail:
                line += f" event_budget={float(detail.get('event_budget', 0.0) or 0.0):.3f}"
            lines.append(line)
    for failed in gate_report.get("failed_checks", []) if isinstance(gate_report.get("failed_checks", []), list) else []:
        lines.append(f"  failed: {failed}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 5 completion gate from managed Phase 4/Phase 5 reports.")
    parser.add_argument(
        "--phase4-report-path",
        default=DEFAULT_PHASE4_REPORT_PATH,
        help="Managed path to the phase4 scale-out benchmark report.",
    )
    parser.add_argument(
        "--phase5-report-path",
        default=DEFAULT_PHASE5_REPORT_PATH,
        help="Managed path to the phase5 predictive coding benchmark report.",
    )
    parser.add_argument(
        "--phase5-entry-gate-report-path",
        default=DEFAULT_PHASE5_ENTRY_GATE_REPORT_PATH,
        help="Managed path to the phase5 entry gate report.",
    )
    parser.add_argument(
        "--sparse-diffusion-block-report-path",
        default=DEFAULT_SPARSE_DIFFUSION_BLOCK_REPORT_PATH,
        help="Managed path to the sparse diffusion block readiness report.",
    )
    parser.add_argument(
        "--output-report-path",
        default=DEFAULT_OUTPUT_REPORT_PATH,
        help="Managed output path for the phase5 completion gate report.",
    )
    parser.add_argument(
        "--output-summary-path",
        default=DEFAULT_OUTPUT_SUMMARY_PATH,
        help="Managed output path for the phase5 completion gate summary.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.phase4_report_path):
        print(f"Phase 5 completion gate failed: Phase 4 report not found at {args.phase4_report_path}")
        return 1
    if not os.path.exists(args.phase5_report_path):
        print(f"Phase 5 completion gate failed: Phase 5 report not found at {args.phase5_report_path}")
        return 1
    if not os.path.exists(args.phase5_entry_gate_report_path):
        print(f"Phase 5 completion gate failed: Phase 5 entry gate report not found at {args.phase5_entry_gate_report_path}")
        return 1
    if not os.path.exists(args.sparse_diffusion_block_report_path):
        print(f"Phase 5 completion gate failed: sparse diffusion block report not found at {args.sparse_diffusion_block_report_path}")
        return 1

    try:
        phase4_report = _load_json(args.phase4_report_path)
        phase5_report = _load_json(args.phase5_report_path)
        phase5_entry_gate_report = _load_json(args.phase5_entry_gate_report_path)
        sparse_diffusion_block_report = _load_json(args.sparse_diffusion_block_report_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Phase 5 completion gate failed: {exc}")
        return 1

    gate_report = build_phase5_completion_gate_report(
        phase4_report,
        phase5_report,
        phase5_entry_gate_report,
        sparse_diffusion_block_report,
    )
    output_report_path = ensure_parent_directory(args.output_report_path)
    output_summary_path = ensure_parent_directory(args.output_summary_path)
    with open(output_report_path, "w", encoding="utf-8") as handle:
        json.dump(gate_report, handle, indent=2, ensure_ascii=False)
    with open(output_summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_phase5_completion_gate_summary(gate_report))

    errors = validate_phase5_completion(phase4_report, phase5_report, phase5_entry_gate_report, sparse_diffusion_block_report)
    if errors:
        print("Phase 5 completion gate failed:")
        for error in errors:
            print(f"- {error}")
        print(f"Saved report: {output_report_path}")
        print(f"Saved summary: {output_summary_path}")
        return 1

    overall_score = float(phase5_report.get("overall_score", 0.0) or 0.0)
    print("Phase 5 completion gate passed.")
    print(f"phase5_overall_score={overall_score:.3f}")
    print(f"Saved report: {output_report_path}")
    print(f"Saved summary: {output_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
