# Directory Path: scripts/eval/release_gate.py
# English Title: Release Gate Validator
# Purpose/Content: Validates release readiness from soak reports and packaging metadata, then exits non-zero when required gates are not satisfied.

import argparse
import importlib.util
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
MPL_CACHE_PATH = os.path.join(PROJECT_ROOT, "workspace", "matplotlib")
os.makedirs(MPL_CACHE_PATH, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE_PATH)
XDG_CACHE_PATH = os.path.join(PROJECT_ROOT, "workspace", "cache")
os.makedirs(XDG_CACHE_PATH, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", XDG_CACHE_PATH)
DEFAULT_REPORT_PATH = os.path.join(PROJECT_ROOT, "workspace", "release", "release_soak_report.json")
DEFAULT_ACCURACY_REPORT_PATH = os.path.join(
    PROJECT_ROOT, "workspace", "evaluation", "phase3_accuracy_suite.json"
)
DEFAULT_EXTERNAL_VALIDITY_REPORT_PATH = os.path.join(
    PROJECT_ROOT, "workspace", "evaluation", "real_data_external_validity.json"
)
DEFAULT_PHASE5_COMPLETION_GATE_REPORT_PATH = os.path.join(
    PROJECT_ROOT, "workspace", "evaluation", "phase5_completion_gate_report.json"
)

def _load_module_from_path(module_name: str, path: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from path: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_contract_constants() -> tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str], List[str]]:
    evaluation_dir = os.path.join(PROJECT_ROOT, "src", "sara_engine", "evaluation")
    stage_b_module = _load_module_from_path(
        "sara_eval_stage_b_contract",
        os.path.join(evaluation_dir, "stage_b_contract.py"),
    )
    stage_c_module = _load_module_from_path(
        "sara_eval_stage_c_contract",
        os.path.join(evaluation_dir, "stage_c_contract.py"),
    )
    stage_d_module = _load_module_from_path(
        "sara_eval_stage_d_contract",
        os.path.join(evaluation_dir, "stage_d_contract.py"),
    )
    stage_e_module = _load_module_from_path(
        "sara_eval_stage_e_contract",
        os.path.join(evaluation_dir, "stage_e_contract.py"),
    )
    phase5_module = _load_module_from_path(
        "sara_eval_phase5_contract",
        os.path.join(evaluation_dir, "phase5_contract.py"),
    )

    return (
        dict(getattr(stage_b_module, "STAGE_B_REQUIRED_MINIMUM_CHECKS")),
        dict(getattr(stage_c_module, "STAGE_C_REQUIRED_MINIMUM_CHECKS")),
        dict(getattr(stage_d_module, "STAGE_D_REQUIRED_MINIMUM_CHECKS")),
        dict(getattr(stage_e_module, "STAGE_E_REQUIRED_MINIMUM_CHECKS")),
        list(getattr(phase5_module, "PHASE5_ENTRY_METRIC_NAMES")),
    )


def _load_project_paths_helpers() -> tuple[Any, Any]:
    module_path = os.path.join(PROJECT_ROOT, "src", "sara_engine", "utils", "project_paths.py")
    module = _load_module_from_path("sara_project_paths", module_path)
    ensure_parent = getattr(module, "ensure_parent_directory", None)
    workspace = getattr(module, "workspace_path", None)
    if not callable(ensure_parent) or not callable(workspace):
        raise RuntimeError("project_paths helper is missing required callables.")
    return ensure_parent, workspace


(
    STAGE_B_REQUIRED_MINIMUM_CHECKS,
    STAGE_C_REQUIRED_MINIMUM_CHECKS,
    STAGE_D_REQUIRED_MINIMUM_CHECKS,
    STAGE_E_REQUIRED_MINIMUM_CHECKS,
    PHASE5_ENTRY_METRIC_NAMES,
) = _load_contract_constants()
ensure_parent_directory, workspace_path = _load_project_paths_helpers()


DEFAULT_REPAIR_LOG_PATH = os.path.join(PROJECT_ROOT, "workspace", "release", "release_repair_execution_log.json")
DEFAULT_REPAIR_PLAN_PATH = os.path.join(PROJECT_ROOT, "workspace", "release", "release_gate_repair_plan.json")


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _int_value(container: Dict[str, Any], key: str, default: int) -> int:
    try:
        return int(container.get(key, default))
    except (TypeError, ValueError):
        return default


def validate_release_report(
    report: Dict[str, object],
    *,
    skip_embedded_accuracy: bool = False,
) -> List[str]:
    errors: List[str] = []

    agent = report.get("agent", {})
    inference = report.get("inference", {})
    criteria = report.get("criteria", {})
    embedded_accuracy = report.get("accuracy")
    release_metadata = report.get("release_metadata")
    release_checklist = report.get("release_checklist")

    min_agent_turns = 24
    min_inference_iterations = 32
    min_pattern_count = 1
    min_duration_seconds = 5.0

    if isinstance(criteria, dict):
        min_agent_turns = _int_value(criteria, "min_agent_turns", min_agent_turns)
        min_inference_iterations = _int_value(criteria, "min_inference_iterations", min_inference_iterations)
        min_pattern_count = _int_value(criteria, "min_pattern_count", min_pattern_count)
        min_duration_raw = criteria.get("min_duration_seconds", min_duration_seconds)
        try:
            if isinstance(min_duration_raw, (int, float, str)):
                min_duration_seconds = float(min_duration_raw)
        except (TypeError, ValueError):
            min_duration_seconds = 5.0

    actual_duration_raw = report.get("duration_seconds", 0.0)
    try:
        if isinstance(actual_duration_raw, (int, float, str)):
            actual_duration_seconds = float(actual_duration_raw)
        else:
            actual_duration_seconds = 0.0
    except (TypeError, ValueError):
        actual_duration_seconds = 0.0

    if actual_duration_seconds < min_duration_seconds:
        errors.append(f"Soak duration is below the minimum required window ({min_duration_seconds} seconds).")

    if not isinstance(agent, dict) or not agent.get("history_bounded", False):
        errors.append("Agent history is not bounded.")
    if not isinstance(agent, dict) or int(agent.get("issue_count", 0)) > 0:
        errors.append("Agent soak recorded runtime issues.")
    if not isinstance(agent, dict) or _int_value(agent, "turns", 0) < min_agent_turns:
        errors.append(f"Agent soak did not reach the minimum turn count ({min_agent_turns}).")
    if not isinstance(inference, dict) or not inference.get("roundtrip_ok", False):
        errors.append("Inference memory round-trip failed.")
    if not isinstance(inference, dict) or not inference.get("tuple_keys_only", False):
        errors.append("Inference memory uses non-tuple keys.")
    if not isinstance(inference, dict) or _int_value(inference, "iterations", 0) < min_inference_iterations:
        errors.append(
            f"Inference soak did not reach the minimum iteration count ({min_inference_iterations})."
        )
    if isinstance(inference, dict) and _int_value(inference, "pattern_count", 0) < min_pattern_count:
        errors.append(
            f"Inference soak did not produce the minimum memory patterns ({min_pattern_count})."
        )

    require_phase3_accuracy = False
    if isinstance(criteria, dict):
        require_phase3_accuracy = bool(criteria.get("require_phase3_accuracy", False))
    if require_phase3_accuracy and not skip_embedded_accuracy:
        if not isinstance(embedded_accuracy, dict):
            errors.append("Release soak report requires embedded Phase 3 accuracy results.")
        else:
            errors.extend(validate_phase3_accuracy_report(embedded_accuracy))

    if isinstance(release_metadata, dict):
        if not bool(release_metadata.get("versions_match", False)):
            errors.append("Embedded release metadata reports mismatched package versions.")
        if not bool(release_metadata.get("has_expected_console_scripts", False)):
            errors.append("Embedded release metadata reports missing console scripts.")
        if not str(release_metadata.get("release_notes_heading", "")).strip():
            errors.append("Embedded release metadata is missing a release notes heading.")

    if isinstance(release_checklist, dict):
        if not bool(release_checklist.get("passed", False)):
            errors.append("Embedded release checklist is not satisfied.")
        if not bool(release_checklist.get("managed_output_paths_ok", False)):
            errors.append("Embedded release checklist reports unmanaged output paths.")
        if not bool(release_checklist.get("release_notes_reviewed", False)):
            errors.append("Embedded release checklist reports missing release notes review state.")
        if not bool(release_checklist.get("report_summary_review_ready", False)):
            errors.append("Embedded release checklist reports missing final review artifacts.")

    return errors


def _float_value(container: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(container.get(key, default))
    except (TypeError, ValueError):
        return default


def _extract_stage_b_metric_value(stage_b_readiness: Dict[str, Any], check_name: str) -> float:
    if not check_name.startswith("metric."):
        return 0.0
    metric_name = check_name[len("metric.") :]
    metrics = stage_b_readiness.get("metrics", {})
    if not isinstance(metrics, dict):
        return 0.0
    return _float_value(metrics, metric_name, 0.0)


def _validate_stage_b_readiness(stage_b_readiness: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(stage_b_readiness, dict):
        return ["Phase 3 accuracy suite is missing Stage B readiness data."]

    if not bool(stage_b_readiness.get("passed", False)):
        errors.append("Phase 3 Stage B readiness criteria are not satisfied.")

    minimum_checks = stage_b_readiness.get("minimum_checks", {})
    if not isinstance(minimum_checks, dict):
        errors.append("Phase 3 Stage B readiness is missing world-model prototype minimum checks.")
        return errors

    missing_minimum_checks = sorted(set(STAGE_B_REQUIRED_MINIMUM_CHECKS).difference(minimum_checks.keys()))
    if missing_minimum_checks:
        errors.append(
            "Phase 3 Stage B readiness is missing required world-model prototype minimum checks: "
            + ", ".join(missing_minimum_checks)
            + "."
        )

    for check_name, description in STAGE_B_REQUIRED_MINIMUM_CHECKS.items():
        if check_name in minimum_checks and not bool(minimum_checks.get(check_name, False)):
            value = _extract_stage_b_metric_value(stage_b_readiness, check_name)
            errors.append(
                "Phase 3 Stage B readiness did not satisfy the world-model prototype minimum for "
                f"{description} ({check_name}, value={value:.3f}, required>=1.000)."
            )

    if not bool(stage_b_readiness.get("minimum_requirements_passed", False)):
        errors.append("Phase 3 Stage B readiness reports unmet world-model prototype minimum requirements.")

    return errors


def _extract_stage_c_metric_value(stage_c_readiness: Dict[str, Any], check_name: str) -> float:
    if not check_name.startswith("metric."):
        return 0.0
    metric_name = check_name[len("metric.") :]
    metrics = stage_c_readiness.get("metrics", {})
    if not isinstance(metrics, dict):
        return 0.0
    return _float_value(metrics, metric_name, 0.0)


def _validate_stage_c_readiness(stage_c_readiness: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(stage_c_readiness, dict):
        return ["Phase 3 accuracy suite is missing Stage C readiness data."]

    if not bool(stage_c_readiness.get("passed", False)):
        errors.append("Phase 3 Stage C readiness criteria are not satisfied.")

    minimum_checks = stage_c_readiness.get("minimum_checks", {})
    if not isinstance(minimum_checks, dict):
        errors.append("Phase 3 Stage C readiness is missing meta-adaptation minimum checks.")
        return errors

    missing_minimum_checks = sorted(set(STAGE_C_REQUIRED_MINIMUM_CHECKS).difference(minimum_checks.keys()))
    if missing_minimum_checks:
        errors.append(
            "Phase 3 Stage C readiness is missing required meta-adaptation minimum checks: "
            + ", ".join(missing_minimum_checks)
            + "."
        )

    for check_name, description in STAGE_C_REQUIRED_MINIMUM_CHECKS.items():
        if check_name in minimum_checks and not bool(minimum_checks.get(check_name, False)):
            value = _extract_stage_c_metric_value(stage_c_readiness, check_name)
            errors.append(
                "Phase 3 Stage C readiness did not satisfy the meta-adaptation minimum for "
                f"{description} ({check_name}, value={value:.3f}, required>=1.000)."
            )

    if not bool(stage_c_readiness.get("minimum_requirements_passed", False)):
        errors.append("Phase 3 Stage C readiness reports unmet meta-adaptation minimum requirements.")

    return errors


def _extract_stage_d_metric_value(stage_d_readiness: Dict[str, Any], check_name: str) -> float:
    if not check_name.startswith("metric."):
        return 0.0
    metric_name = check_name[len("metric.") :]
    metrics = stage_d_readiness.get("metrics", {})
    if not isinstance(metrics, dict):
        return 0.0
    return _float_value(metrics, metric_name, 0.0)


def _validate_stage_d_readiness(stage_d_readiness: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(stage_d_readiness, dict):
        return ["Phase 3 accuracy suite is missing Stage D readiness data."]

    if not bool(stage_d_readiness.get("passed", False)):
        errors.append("Phase 3 Stage D readiness criteria are not satisfied.")

    minimum_checks = stage_d_readiness.get("minimum_checks", {})
    if not isinstance(minimum_checks, dict):
        errors.append("Phase 3 Stage D readiness is missing continual-consolidation minimum checks.")
        return errors

    missing_minimum_checks = sorted(set(STAGE_D_REQUIRED_MINIMUM_CHECKS).difference(minimum_checks.keys()))
    if missing_minimum_checks:
        errors.append(
            "Phase 3 Stage D readiness is missing required continual-consolidation minimum checks: "
            + ", ".join(missing_minimum_checks)
            + "."
        )

    for check_name, description in STAGE_D_REQUIRED_MINIMUM_CHECKS.items():
        if check_name in minimum_checks and not bool(minimum_checks.get(check_name, False)):
            value = _extract_stage_d_metric_value(stage_d_readiness, check_name)
            metric_name = check_name[len("metric.") :] if check_name.startswith("metric.") else check_name
            errors.append(
                "Phase 3 Stage D readiness did not satisfy the continual-consolidation minimum for "
                f"{description} ({check_name}, metric={metric_name}, value={value:.3f}, required>=1.000)."
            )

    if not bool(stage_d_readiness.get("minimum_requirements_passed", False)):
        errors.append("Phase 3 Stage D readiness reports unmet continual-consolidation minimum requirements.")

    return errors


def _extract_stage_e_metric_value(stage_e_readiness: Dict[str, Any], check_name: str) -> float:
    if not check_name.startswith("metric."):
        return 0.0
    metric_name = check_name[len("metric.") :]
    metrics = stage_e_readiness.get("metrics", {})
    if not isinstance(metrics, dict):
        return 0.0
    return _float_value(metrics, metric_name, 0.0)


def _validate_stage_e_readiness(stage_e_readiness: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(stage_e_readiness, dict):
        return ["Phase 3 accuracy suite is missing Stage E readiness data."]

    if not bool(stage_e_readiness.get("passed", False)):
        errors.append("Phase 3 Stage E readiness criteria are not satisfied.")

    minimum_checks = stage_e_readiness.get("minimum_checks", {})
    if not isinstance(minimum_checks, dict):
        errors.append("Phase 3 Stage E readiness is missing modular-cognitive-runtime minimum checks.")
        return errors

    missing_minimum_checks = sorted(set(STAGE_E_REQUIRED_MINIMUM_CHECKS).difference(minimum_checks.keys()))
    if missing_minimum_checks:
        errors.append(
            "Phase 3 Stage E readiness is missing required modular-cognitive-runtime minimum checks: "
            + ", ".join(missing_minimum_checks)
            + "."
        )

    for check_name, description in STAGE_E_REQUIRED_MINIMUM_CHECKS.items():
        if check_name in minimum_checks and not bool(minimum_checks.get(check_name, False)):
            value = _extract_stage_e_metric_value(stage_e_readiness, check_name)
            metric_name = check_name[len("metric.") :] if check_name.startswith("metric.") else check_name
            errors.append(
                "Phase 3 Stage E readiness did not satisfy the modular-cognitive-runtime minimum for "
                f"{description} ({check_name}, metric={metric_name}, value={value:.3f}, required>=1.000)."
            )

    if not bool(stage_e_readiness.get("minimum_requirements_passed", False)):
        errors.append("Phase 3 Stage E readiness reports unmet modular-cognitive-runtime minimum requirements.")

    return errors


def _validate_phase5_entry_readiness(focus_summary: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    phase5_entry = (
        focus_summary.get("phase5_entry_readiness", {})
        if isinstance(focus_summary.get("phase5_entry_readiness"), dict)
        else {}
    )
    if not phase5_entry:
        return ["Phase 3 accuracy suite is missing Phase 5 entry readiness data."]

    if not bool(phase5_entry.get("passed", False)):
        errors.append("Phase 5 entry readiness criteria are not satisfied.")

    metrics = phase5_entry.get("metrics", {})
    if not isinstance(metrics, dict):
        return ["Phase 5 entry readiness is missing predictive-coding metrics."]

    for metric_name in PHASE5_ENTRY_METRIC_NAMES:
        focus_metric_name = f"phase5_predictive_coding.{metric_name}"
        if focus_metric_name not in metrics:
            errors.append(f"Phase 5 entry readiness is missing required metric '{focus_metric_name}'.")
            continue
        value = _float_value(metrics, focus_metric_name, 0.0)
        if value < 1.0:
            errors.append(
                f"Phase 5 entry readiness did not satisfy predictive-coding metric "
                f"{focus_metric_name} (value={value:.3f}, required>=1.000)."
            )

    return errors


def _validate_efficiency_readiness_focus(focus_summary: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    efficiency_readiness = (
        focus_summary.get("efficiency_readiness", {})
        if isinstance(focus_summary.get("efficiency_readiness"), dict)
        else {}
    )
    if not efficiency_readiness:
        return errors

    metrics = efficiency_readiness.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    efficiency_thresholds: Dict[str, tuple[float, str]] = {
        "energy_efficiency.energy_per_success_proxy": (1.0, "energy per success proxy"),
        "energy_efficiency.performance_energy_ratio_proxy": (
            0.20,
            "performance-per-energy ratio proxy",
        ),
        "energy_efficiency.ann_cost_advantage_proxy": (
            8.0,
            "ANN-reference cost advantage proxy",
        ),
        "energy_efficiency.sparse_event_cost_score": (1.0, "sparse-event cost score"),
        "energy_efficiency.brain_efficiency_alignment_proxy": (
            0.85,
            "brain-efficiency alignment proxy",
        ),
        "energy_efficiency.memory_per_success_proxy": (1.0, "memory per success proxy"),
        "energy_efficiency.low_overhead_route_score": (1.0, "low-overhead route score"),
        "energy_efficiency.bounded_latency_score": (0.80, "bounded latency score"),
        "energy_efficiency.stochastic_readout_integrity": (
            1.0,
            "stochastic readout integrity",
        ),
    }

    detailed_failures: List[str] = []
    for metric_name, (threshold, label) in efficiency_thresholds.items():
        if metric_name not in metrics:
            continue
        value = _float_value(metrics, metric_name, 0.0)
        if value >= threshold:
            continue
        detailed_failures.append(
            "Phase 3 efficiency_readiness did not satisfy "
            f"{label} ({metric_name}, value={value:.3f}, required>={threshold:.3f})."
        )

    if detailed_failures:
        errors.extend(detailed_failures)
        return errors

    if not bool(efficiency_readiness.get("passed", False)):
        if not metrics:
            errors.append(
                "Phase 3 efficiency_readiness did not pass and is missing detailed metrics; "
                "expected energy, ANN-cost, and performance-per-energy proxies."
            )
        else:
            errors.append(
                "Phase 3 efficiency_readiness did not pass even though all available efficiency metrics "
                "appear to satisfy thresholds."
            )
    return errors


def validate_phase3_accuracy_report(report: Dict[str, object]) -> List[str]:
    errors: List[str] = []
    if not isinstance(report, dict):
        return ["Phase 3 accuracy report is not a valid object."]

    if report.get("suite_name") != "Phase3AccuracySuite":
        errors.append("Phase 3 accuracy report has an unexpected suite name.")

    if not bool(report.get("passed", False)):
        errors.append("Phase 3 accuracy suite did not pass.")

    overall_score = _float_value(report, "overall_score", 0.0)
    if overall_score <= 0.0:
        errors.append("Phase 3 accuracy suite overall score is missing or invalid.")
    if overall_score < 0.95:
        errors.append("Phase 3 accuracy suite overall score is below the Stage A ACC target (0.95).")

    component_reports = report.get("component_reports", {})
    required_components = {
        "agent_dialogue",
        "sara_inference",
        "spiking_llm",
        "task_switch_adaptation",
        "future_state_consistency",
        "energy_efficiency",
        "continual_consolidation",
        "cognitive_runtime",
        "phase5_predictive_coding",
    }
    if not isinstance(component_reports, dict):
        errors.append("Phase 3 accuracy suite is missing component reports.")
        return errors

    missing_components = sorted(required_components.difference(component_reports.keys()))
    if missing_components:
        errors.append(
            "Phase 3 accuracy suite is missing required components: "
            + ", ".join(missing_components)
            + "."
        )

    for component_name in sorted(required_components.intersection(component_reports.keys())):
        component = component_reports.get(component_name, {})
        if not isinstance(component, dict):
            errors.append(f"Phase 3 component '{component_name}' is not a valid object.")
            continue
        if not bool(component.get("passed", False)):
            errors.append(f"Phase 3 component '{component_name}' did not pass.")

    trend = report.get("trend", {})
    if isinstance(trend, dict) and bool(trend.get("has_previous", False)):
        regression_count = _int_value(
            trend,
            "gate_regression_count",
            _int_value(trend, "regression_count", 0),
        )
        if regression_count > 0:
            errors.append(
                f"Phase 3 accuracy suite detected {regression_count} gate metric regression(s) "
                "versus the previous run."
            )

    focus_summary = report.get("focus_summary", {})
    required_focus = {
        "few_shot",
        "continual",
        "retrieval_hygiene",
        "adaptive_readiness",
        "predictive_readiness",
        "efficiency_readiness",
        "consolidation_readiness",
        "cognitive_runtime_readiness",
        "phase5_entry_readiness",
    }
    if not isinstance(focus_summary, dict):
        errors.append("Phase 3 accuracy suite is missing focus summary data.")
        return errors

    missing_focus = sorted(required_focus.difference(focus_summary.keys()))
    if missing_focus:
        errors.append(
            "Phase 3 accuracy suite is missing focus summaries: "
            + ", ".join(missing_focus)
            + "."
        )

    for focus_name in sorted(required_focus.intersection(focus_summary.keys())):
        focus_report = focus_summary.get(focus_name, {})
        if not isinstance(focus_report, dict):
            errors.append(f"Phase 3 focus summary '{focus_name}' is not a valid object.")
            continue
        if not bool(focus_report.get("passed", False)):
            errors.append(f"Phase 3 focus summary '{focus_name}' did not pass.")

    stage_a_acceptance = report.get("stage_a_acceptance", {})
    if not isinstance(stage_a_acceptance, dict):
        errors.append("Phase 3 accuracy suite is missing Stage A acceptance data.")
    else:
        if not bool(stage_a_acceptance.get("passed", False)):
            errors.append("Phase 3 Stage A acceptance criteria are not satisfied.")

    errors.extend(_validate_stage_b_readiness(report.get("stage_b_readiness")))
    errors.extend(_validate_stage_c_readiness(report.get("stage_c_readiness")))
    errors.extend(_validate_stage_d_readiness(report.get("stage_d_readiness")))
    errors.extend(_validate_stage_e_readiness(report.get("stage_e_readiness")))
    errors.extend(_validate_phase5_entry_readiness(focus_summary))
    errors.extend(_validate_efficiency_readiness_focus(focus_summary))

    return errors


def validate_external_validity_report(report: Dict[str, object]) -> List[str]:
    errors: List[str] = []
    if not isinstance(report, dict):
        return ["Real-data external validity report is not a valid object."]

    if report.get("suite_name") != "RealDataExternalValidity":
        errors.append("Real-data external validity report has an unexpected suite name.")
    if not bool(report.get("passed", False)):
        errors.append("Real-data external validity report did not pass.")

    checks = report.get("checks", {})
    metrics = report.get("metrics", {})
    check_details = report.get("check_details", {})
    thresholds = report.get("thresholds", {})
    if not isinstance(checks, dict):
        errors.append("Real-data external validity report is missing checks.")
        checks = {}
    if not isinstance(metrics, dict):
        errors.append("Real-data external validity report is missing metrics.")
        metrics = {}
    if not isinstance(check_details, dict):
        check_details = {}
    if not isinstance(thresholds, dict):
        thresholds = {}

    required_checks = {
        "real_data_task_count",
        "sparse_accuracy_floor",
        "sparse_matches_dense_accuracy",
        "summary_keyword_coverage_floor",
        "continual_memory_hit_rate_floor",
        "ann_cost_advantage_proxy",
        "performance_energy_ratio_proxy",
        "trend.no_regressions",
    }
    missing_checks = sorted(required_checks.difference(checks.keys()))
    if missing_checks:
        errors.append(
            "Real-data external validity report is missing required checks: "
            + ", ".join(missing_checks)
            + "."
        )
    for check_name in sorted(required_checks.intersection(checks.keys())):
        detail = check_details.get(check_name, {}) if isinstance(check_details.get(check_name), dict) else {}
        passed = bool(detail.get("passed", checks.get(check_name, False)))
        if not passed:
            value = detail.get("value")
            required_min = detail.get("required_min")
            required_max = detail.get("required_max")
            if isinstance(value, (int, float)) and isinstance(required_min, (int, float)):
                errors.append(
                    "Real-data external validity check failed: "
                    f"{check_name} (value={float(value):.3f}, required>={float(required_min):.3f})."
                )
            elif isinstance(value, (int, float)) and isinstance(required_max, (int, float)):
                errors.append(
                    "Real-data external validity check failed: "
                    f"{check_name} (value={float(value):.3f}, required<={float(required_max):.3f})."
                )
            else:
                errors.append(f"Real-data external validity check failed: {check_name}.")

    if check_details:
        return errors

    metric_thresholds = {
        "real_data_qa_accuracy": _float_value(thresholds, "min_real_data_qa_accuracy", 0.80),
        "real_data_summary_keyword_coverage": _float_value(thresholds, "min_summary_keyword_coverage", 0.60),
        "continual_memory_hit_rate": _float_value(thresholds, "min_continual_memory_hit_rate", 0.80),
        "performance_energy_ratio_proxy": _float_value(thresholds, "min_performance_energy_ratio_proxy", 2.0),
        "ann_cost_advantage_proxy": _float_value(thresholds, "min_ann_cost_advantage_proxy", 2.0),
    }
    for metric_name, required in metric_thresholds.items():
        value = _float_value(metrics, metric_name, 0.0) if isinstance(metrics, dict) else 0.0
        if value < required:
            errors.append(
                "Real-data external validity metric did not satisfy release threshold "
                f"({metric_name}, value={value:.3f}, required>={required:.3f})."
            )
    sparse_accuracy = _float_value(metrics, "real_data_qa_accuracy", 0.0) if isinstance(metrics, dict) else 0.0
    dense_accuracy = _float_value(metrics, "ann_proxy_qa_accuracy", 0.0) if isinstance(metrics, dict) else 0.0
    dense_tolerance = _float_value(thresholds, "dense_accuracy_tolerance", 0.05)
    if sparse_accuracy < max(dense_accuracy - dense_tolerance, 0.0):
        errors.append(
            "Real-data external validity sparse retrieval trails ANN proxy beyond release tolerance "
            f"(sara={sparse_accuracy:.3f}, ann_proxy={dense_accuracy:.3f}, tolerance={dense_tolerance:.3f})."
        )
    return errors


def validate_phase5_completion_gate_report(report: Dict[str, object]) -> List[str]:
    errors: List[str] = []
    if not isinstance(report, dict):
        return ["Phase 5 completion gate report is not a valid object."]

    if str(report.get("suite_name", "")) != "Phase5CompletionGate":
        errors.append("Phase 5 completion gate report has an unexpected suite name.")
    if not bool(report.get("passed", False)):
        errors.append("Phase 5 completion gate did not pass.")
    if float(report.get("phase5_overall_score", 0.0) or 0.0) < 1.0:
        errors.append(
            "Phase 5 completion gate overall score is below the required threshold "
            f"(value={float(report.get('phase5_overall_score', 0.0) or 0.0):.3f}, required>=1.000)."
        )

    failed_checks = report.get("failed_checks", [])
    if isinstance(failed_checks, list):
        failed_items = [str(item) for item in failed_checks if str(item).strip()]
        if failed_items:
            errors.append("Phase 5 completion gate has failed checks: " + ", ".join(failed_items))
    else:
        errors.append("Phase 5 completion gate report is missing failed_checks.")

    gate_errors = report.get("errors", [])
    if isinstance(gate_errors, list):
        gate_error_items = [str(item) for item in gate_errors if str(item).strip()]
        if gate_error_items:
            errors.append("Phase 5 completion gate reported errors: " + " | ".join(gate_error_items))
    else:
        errors.append("Phase 5 completion gate report is missing errors.")

    checks = report.get("checks", {})
    if not isinstance(checks, dict):
        errors.append("Phase 5 completion gate report is missing checks.")
        return errors

    required_check_names = {
        "phase5_entry_gate_passed",
        "multi_step_trace_complete",
        "counterfactual_branch_separable",
        "macro_step_reduction",
        "macro_cost_reduction",
        "subgoal_coverage_ratio",
        "micro_es_low_rank_trace_complete",
        "micro_es_fitness_improvement",
        "micro_es_event_cost_reduction",
        "micro_es_population_event_budget",
        "sparse_diffusion_block_readiness_passed",
        "sparse_diffusion.sparse_diffusion_partition_integrity",
        "sparse_diffusion.sparse_diffusion_independent_block_integrity",
        "sparse_diffusion.sparse_diffusion_denoise_accuracy",
        "sparse_diffusion.sparse_diffusion_event_cost_advantage",
        "sparse_diffusion.sparse_diffusion_block_ablation_integrity",
        "sparse_diffusion.sparse_diffusion_single_pass_recurrent_integrity",
        "sparse_diffusion.sparse_diffusion_policy_compatibility",
    }
    required_check_names.update({f"metric.{name}" for name in PHASE5_ENTRY_METRIC_NAMES})
    required_check_names.update({f"threshold.{name}" for name in PHASE5_ENTRY_METRIC_NAMES})

    missing_required_checks = sorted(name for name in required_check_names if name not in checks)
    if missing_required_checks:
        errors.append(
            "Phase 5 completion gate check map is missing required checks: "
            + ", ".join(missing_required_checks)
        )

    failed_required_checks = sorted(
        name
        for name in required_check_names
        if name in checks
        and not (isinstance(checks.get(name), dict) and bool(checks.get(name, {}).get("passed", False)))
    )
    if failed_required_checks:
        errors.append(
            "Phase 5 completion gate check map contains failed checks: "
            + ", ".join(failed_required_checks)
        )

    return errors


def validate_packaging_metadata(project_root: str) -> List[str]:
    errors: List[str] = []
    pyproject = _read_text(os.path.join(project_root, "pyproject.toml"))
    cargo = _read_text(os.path.join(project_root, "Cargo.toml"))

    pyproject_version = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    cargo_version = re.search(r'^version = "([^"]+)"', cargo, re.MULTILINE)

    if pyproject_version is None or cargo_version is None:
        errors.append("Could not read version from pyproject.toml or Cargo.toml.")
    elif pyproject_version.group(1) != cargo_version.group(1):
        errors.append("pyproject.toml and Cargo.toml versions do not match.")

    if 'sara-chat = "sara_engine.cli:chat"' not in pyproject:
        errors.append("Missing sara-chat console script entry.")
    if 'sara-train = "sara_engine.cli:train"' not in pyproject:
        errors.append("Missing sara-train console script entry.")

    return errors


def load_repair_execution_log(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as handle:
            if str(path).lower().endswith(".jsonl"):
                rows = []
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    payload = json.loads(text)
                    if isinstance(payload, dict):
                        rows.append(payload)
                return rows
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []

    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        entries = payload.get("entries", [])
        if isinstance(entries, list):
            return [dict(item) for item in entries if isinstance(item, dict)]
    return []


def suggest_release_gate_recovery_actions(errors: List[str]) -> List[Dict[str, Any]]:
    if not errors:
        return []

    actions: List[Dict[str, Any]] = []
    action_by_title: Dict[str, Dict[str, Any]] = {}
    action_priorities: Dict[str, int] = {}
    seen_titles = set()

    def _add_action(
        title: str,
        command: str,
        reason: str,
        *,
        priority: str,
        expected_effect: str,
        affected_checks: List[str],
    ) -> None:
        normalized_checks = sorted({str(item) for item in affected_checks if str(item).strip()})
        if title in action_by_title:
            existing = action_by_title[title]
            existing_checks = existing.get("affected_checks", [])
            if not isinstance(existing_checks, list):
                existing_checks = []
            existing["affected_checks"] = sorted(
                {
                    str(item)
                    for item in [*existing_checks, *normalized_checks]
                    if str(item).strip()
                }
            )
            return
        if title in seen_titles:
            return
        seen_titles.add(title)
        priority_map = {"high": 0, "medium": 1, "low": 2}
        action_priorities[title] = priority_map.get(priority, 2)
        action = {
            "title": title,
            "command": command,
            "reason": reason,
            "priority": priority,
            "expected_effect": expected_effect,
            "affected_checks": normalized_checks,
        }
        actions.append(action)
        action_by_title[title] = action

    for error in errors:
        text = str(error).lower()
        if "stage b readiness" in text or "world-model prototype minimum" in text:
            _add_action(
                "Re-run Phase 3 Accuracy Suite",
                "python scripts/eval/phase3_accuracy_suite.py",
                "Refresh Stage B world-model metrics and minimum checks.",
                priority="high",
                expected_effect="Restores missing Stage B world-model minimum metrics.",
                affected_checks=[
                    "stage_b.minimum_checks",
                    "stage_b.minimum_requirements_passed",
                ],
            )
        if "stage c readiness" in text or "meta-adaptation minimum" in text:
            _add_action(
                "Re-run Task-Switch Adaptation Benchmark",
                "python scripts/eval/task_switch_adaptation_benchmark.py",
                "Refresh Stage C meta-adaptation metrics and minimum checks.",
                priority="high",
                expected_effect="Restores missing Stage C meta-adaptation minimum metrics.",
                affected_checks=[
                    "stage_c.minimum_checks",
                    "stage_c.minimum_requirements_passed",
                ],
            )
        if "stage d readiness" in text or "continual-consolidation minimum" in text:
            _add_action(
                "Re-run Continual Consolidation Benchmark",
                "python scripts/eval/continual_consolidation_benchmark.py",
                "Refresh Stage D continual-consolidation metrics and minimum checks.",
                priority="high",
                expected_effect="Restores missing Stage D continual-consolidation minimum metrics.",
                affected_checks=[
                    "stage_d.minimum_checks",
                    "stage_d.minimum_requirements_passed",
                ],
            )
        if "stage e readiness" in text or "modular-cognitive-runtime minimum" in text:
            _add_action(
                "Re-run Cognitive Runtime Benchmark",
                "python scripts/eval/cognitive_runtime_benchmark.py",
                "Refresh Stage E modular cognitive runtime metrics and minimum checks.",
                priority="high",
                expected_effect="Restores missing Stage E cognitive runtime minimum metrics.",
                affected_checks=[
                    "stage_e.minimum_checks",
                    "stage_e.minimum_requirements_passed",
                ],
            )
        if "phase 5 entry readiness" in text or "phase5_predictive_coding" in text or "predictive-coding metric" in text:
            phase5_checks = [
                "phase5.entry_readiness",
                "phase5.predictive_coding_metrics",
            ]
            reason = "Refresh Phase 5 Spiking H-JEPA entry metrics."
            expected_effect = "Restores missing Phase 5 predictive-coding entry metrics."
            if "horizon_bucket_stability" in text:
                phase5_checks.append("phase5.horizon_bucket_stability")
                reason = (
                    "Refresh Phase 5 Spiking H-JEPA horizon-bucket traces and transition-bucket consistency."
                )
                expected_effect = (
                    "Recovers horizon_bucket_stability and stabilizes multi-horizon predictive-coding transitions."
                )
            if "macro_action_effectiveness" in text:
                phase5_checks.append("phase5.macro_action_effectiveness")
                reason = (
                    "Refresh macro-action planning traces and macro-vs-micro efficiency comparisons in Phase 5."
                )
                expected_effect = (
                    "Recovers macro_action_effectiveness by restoring macro-step utility and cost-reduction consistency."
                )
            if "subgoal_decomposition_integrity" in text:
                phase5_checks.append("phase5.subgoal_decomposition_integrity")
                reason = (
                    "Refresh subgoal decomposition traces and coverage accounting in Phase 5 predictive coding."
                )
                expected_effect = (
                    "Recovers subgoal_decomposition_integrity by restoring subgoal coverage and decomposition consistency."
                )
            if "micro_es_policy_refinement_integrity" in text:
                phase5_checks.append("phase5.micro_es_policy_refinement_integrity")
                reason = (
                    "Refresh energy-aware micro-ES policy traces in Phase 5 predictive coding."
                )
                expected_effect = (
                    "Recovers micro_es_policy_refinement_integrity by restoring low-rank policy refinement and event-cost checks."
                )
            _add_action(
                "Re-run Phase 5 Predictive Coding Benchmark",
                "python scripts/eval/phase5_predictive_coding_benchmark.py",
                reason,
                priority="high",
                expected_effect=expected_effect,
                affected_checks=phase5_checks,
            )
        if "phase 5 completion gate" in text or "phase5 completion gate" in text:
            _add_action(
                "Re-run Phase 5 Completion Gate",
                "python scripts/eval/phase5_predictive_coding_benchmark.py && python scripts/eval/phase5_entry_gate.py && python scripts/eval/phase5_completion_gate.py",
                "Refresh Phase 5 completion artifact checks, including required metric/threshold, macro-subgoal detail, and micro-ES detail checks.",
                priority="high",
                expected_effect="Recovers missing or failed Phase 5 completion gate required checks before release promotion.",
                affected_checks=[
                    "phase5.completion_gate",
                    "phase5.completion_required_checks",
                    "phase5.macro_action_effectiveness",
                    "phase5.subgoal_decomposition_integrity",
                    "phase5.micro_es_policy_refinement_integrity",
                    "phase5.micro_es_low_rank_trace_complete",
                    "phase5.micro_es_fitness_improvement",
                    "phase5.micro_es_event_cost_reduction",
                    "phase5.micro_es_population_event_budget",
                ],
            )
        if "real-data external validity" in text or "external validity" in text:
            _add_action(
                "Re-run Real-Data External Validity Benchmark",
                "python scripts/eval/real_data_external_validity.py",
                "Refresh real-corpus QA, summary, continual-memory, and ANN-cost advantage evidence.",
                priority="high",
                expected_effect="Restores external validity evidence and ANN-ratio energy checks before release.",
                affected_checks=[
                    "external_validity.report",
                    "external_validity.real_data_qa_accuracy",
                    "external_validity.performance_energy_ratio_proxy",
                    "external_validity.ann_cost_advantage_proxy",
                ],
            )
        if "soak duration" in text or "minimum turn count" in text or "minimum iteration count" in text:
            _add_action(
                "Run Extended Release Soak",
                "python scripts/eval/release_soak.py --profile extended --include-accuracy",
                "Rebuild soak evidence with stronger wall-clock and workload coverage.",
                priority="high",
                expected_effect="Recovers soak threshold failures in duration, turns, and iterations.",
                affected_checks=[
                    "soak.duration_seconds",
                    "soak.agent.turns",
                    "soak.inference.iterations",
                ],
            )
        if "embedded phase 3 accuracy" in text or "accuracy report not found" in text:
            _add_action(
                "Embed Accuracy Into Soak Report",
                "python scripts/eval/release_soak.py --profile release --include-accuracy",
                "Attach Phase 3 report to the release soak artifact.",
                priority="high",
                expected_effect="Restores missing embedded Phase 3 evidence in the soak report.",
                affected_checks=[
                    "release_gate.embedded_accuracy_present",
                    "release_gate.accuracy_required",
                ],
            )
        if "phase 3 component" in text or "focus summary" in text or "acc target" in text:
            _add_action(
                "Inspect Phase 3 Summary",
                "python scripts/eval/phase3_accuracy_suite.py --summary-path workspace/evaluation/phase3_accuracy_summary.txt",
                "Locate failing benchmark/focus metrics before rerunning gate.",
                priority="medium",
                expected_effect="Identifies failing Phase 3 components before another gate run.",
                affected_checks=[
                    "phase3.component_reports",
                    "phase3.focus_summary",
                    "stage_a.acc_target",
                ],
            )
        if (
            "adaptive_readiness" in text
            or "meta_adaptation_parameter_integrity" in text
            or "temporal_self_distillation_stability" in text
            or "temporal self-distillation stability" in text
        ):
            _add_action(
                "Re-run Task-Switch Adaptation Benchmark",
                "python scripts/eval/task_switch_adaptation_benchmark.py",
                "Re-measure adaptation-loop integrity and parameter stabilization before rerunning phase3 suite.",
                priority="high",
                expected_effect="Recovers Stage C adaptation metrics and clarifies adaptation-state drift causes.",
                affected_checks=[
                    "stage_c.adaptive_readiness",
                    "stage_c.meta_adaptation_parameter_integrity",
                    "stage_c.temporal_self_distillation_stability",
                ],
            )
        if (
            "external validity" not in text
            and (
                "efficiency_readiness" in text
                or "performance-per-energy ratio proxy" in text
                or "ann-reference cost advantage proxy" in text
                or "brain-efficiency alignment proxy" in text
                or "energy_per_success_proxy" in text
                or "performance_energy_ratio_proxy" in text
                or "ann_cost_advantage_proxy" in text
                or "sparse_event_cost_score" in text
                or "brain_efficiency_alignment_proxy" in text
            )
        ):
            _add_action(
                "Re-run Energy Efficiency Benchmark",
                "python scripts/eval/energy_efficiency_benchmark.py",
                "Recompute performance-per-energy and ANN-reference cost advantage metrics.",
                priority="high",
                expected_effect="Restores missing or degraded efficiency-readiness metrics including ANN ratio signals.",
                affected_checks=[
                    "focus.efficiency_readiness.passed",
                    "energy_efficiency.performance_energy_ratio_proxy",
                    "energy_efficiency.ann_cost_advantage_proxy",
                    "energy_efficiency.brain_efficiency_alignment_proxy",
                ],
            )
            _add_action(
                "Re-run Phase 3 Accuracy Suite",
                "python scripts/eval/phase3_accuracy_suite.py",
                "Refresh focus summary after efficiency benchmark updates.",
                priority="high",
                expected_effect="Updates efficiency_readiness status and trend in the Phase 3 gate report.",
                affected_checks=[
                    "focus.efficiency_readiness.passed",
                    "phase3.focus_summary",
                ],
            )
        if (
            "consolidation_readiness" in text
            or "replay_recovery_integrity" in text
            or "long_horizon_consolidation_retention" in text
            or "counterfactual_replay_selection_integrity" in text
            or "replay_upgrade_reindex_integrity" in text
            or "memory_health_index_integrity" in text
            or "replay_noise_resilience_integrity" in text
            or "astro_modulation_stability" in text
        ):
            _add_action(
                "Re-run Continual Consolidation Benchmark",
                "python scripts/eval/continual_consolidation_benchmark.py",
                "Re-measure replay recovery and long-horizon consolidation behavior before rerunning phase3 suite.",
                priority="high",
                expected_effect="Recovers Stage D consolidation metrics and clarifies replay-stability drift causes.",
                affected_checks=[
                    "stage_d.consolidation_readiness",
                    "stage_d.replay_recovery_integrity",
                    "stage_d.long_horizon_consolidation_retention",
                    "stage_d.counterfactual_replay_selection_integrity",
                    "stage_d.replay_upgrade_reindex_integrity",
                    "stage_d.memory_health_index_integrity",
                    "stage_d.replay_noise_resilience_integrity",
                    "stage_d.astro_modulation_stability",
                ],
            )
        if "console script" in text or "versions do not match" in text:
            _add_action(
                "Validate Packaging Metadata",
                "python scripts/eval/release_soak.py --profile quick --skip-accuracy-gate",
                "Regenerate metadata checks for version and console-script consistency.",
                priority="high",
                expected_effect="Recovers packaging metadata mismatches and console-script failures.",
                affected_checks=[
                    "packaging.version_match",
                    "packaging.console_scripts",
                ],
            )
        if "release notes" in text or "release checklist" in text:
            _add_action(
                "Rebuild Release Checklist Artifacts",
                "python scripts/eval/release_soak.py --profile release --include-accuracy",
                "Regenerate report/summary/checklist fields from a fresh soak run.",
                priority="medium",
                expected_effect="Repairs missing checklist/release-notes readiness fields.",
                affected_checks=[
                    "checklist.release_notes_reviewed",
                    "checklist.report_summary_review_ready",
                ],
            )

    if not actions:
        _add_action(
            "Re-run Release Gate",
            "python scripts/eval/release_gate.py",
            "Recompute release readiness with the latest managed reports.",
            priority="low",
            expected_effect="Refreshes gate results after manual adjustments.",
            affected_checks=["release_gate.errors"],
        )

    actions.sort(key=lambda action: (action_priorities.get(str(action.get("title", "")), 2), str(action.get("title", ""))))
    return actions


def _infer_failed_checks_from_errors(errors: List[str]) -> List[str]:
    if not errors:
        return []
    inferred = set()
    for error in errors:
        text = str(error).lower()
        matched = False
        if "soak duration" in text:
            inferred.add("soak.duration_seconds")
            matched = True
        if "minimum turn count" in text:
            inferred.add("soak.agent.turns")
            matched = True
        if "minimum iteration count" in text:
            inferred.add("soak.inference.iterations")
            matched = True
        if "round-trip failed" in text:
            inferred.add("soak.inference.roundtrip_ok")
            matched = True
        if "non-tuple keys" in text:
            inferred.add("soak.inference.tuple_keys_only")
            matched = True
        if "minimum memory patterns" in text:
            inferred.add("soak.inference.pattern_count")
            matched = True
        if "embedded phase 3 accuracy" in text:
            inferred.add("release_gate.embedded_accuracy_present")
            matched = True
        if "overall score is below the stage a acc target" in text:
            inferred.add("stage_a.acc_target")
            matched = True
        if "stage b readiness" in text or "world-model prototype minimum" in text:
            inferred.add("stage_b.minimum_checks")
            matched = True
        if "stage c readiness" in text or "meta-adaptation minimum" in text:
            inferred.add("stage_c.minimum_checks")
            matched = True
        if "stage d readiness" in text or "continual-consolidation minimum" in text:
            inferred.add("stage_d.minimum_checks")
            matched = True
        if "stage e readiness" in text or "modular-cognitive-runtime minimum" in text:
            inferred.add("stage_e.minimum_checks")
            matched = True
        if "phase 5 entry readiness" in text or "phase5_predictive_coding" in text or "predictive-coding metric" in text:
            inferred.add("phase5.entry_readiness")
            matched = True
        if "phase 5 completion gate" in text or "phase5 completion gate" in text:
            inferred.add("phase5.completion_gate")
            matched = True
        if "missing required checks" in text and "phase 5 completion gate" in text:
            inferred.add("phase5.completion_required_checks")
            matched = True
        if "real-data external validity" in text or "external validity" in text:
            inferred.add("external_validity.report")
            matched = True
        if "real_data_qa_accuracy" in text:
            inferred.add("external_validity.real_data_qa_accuracy")
            matched = True
        if "continual_memory_hit_rate" in text:
            inferred.add("external_validity.continual_memory_hit_rate")
            matched = True
        if "real_data_summary_keyword_coverage" in text:
            inferred.add("external_validity.summary_keyword_coverage")
            matched = True
        if "real-data external validity" in text and "performance_energy_ratio_proxy" in text:
            inferred.add("external_validity.performance_energy_ratio_proxy")
            matched = True
        if "real-data external validity" in text and "ann_cost_advantage_proxy" in text:
            inferred.add("external_validity.ann_cost_advantage_proxy")
            matched = True
        if "horizon_bucket_stability" in text:
            inferred.add("phase5.horizon_bucket_stability")
            matched = True
        if "macro_action_effectiveness" in text:
            inferred.add("phase5.macro_action_effectiveness")
            matched = True
        if "subgoal_decomposition_integrity" in text:
            inferred.add("phase5.subgoal_decomposition_integrity")
            matched = True
        if "micro_es_policy_refinement_integrity" in text:
            inferred.add("phase5.micro_es_policy_refinement_integrity")
            matched = True
        if "micro_es_low_rank_trace_complete" in text:
            inferred.add("phase5.micro_es_low_rank_trace_complete")
            matched = True
        if "micro_es_fitness_improvement" in text:
            inferred.add("phase5.micro_es_fitness_improvement")
            matched = True
        if "micro_es_event_cost_reduction" in text:
            inferred.add("phase5.micro_es_event_cost_reduction")
            matched = True
        if "micro_es_population_event_budget" in text:
            inferred.add("phase5.micro_es_population_event_budget")
            matched = True
        if "adaptive_readiness" in text:
            inferred.add("stage_c.adaptive_readiness")
            matched = True
        if (
            "external validity" not in text
            and (
                "efficiency_readiness" in text
                or "performance-per-energy ratio proxy" in text
                or "ann-reference cost advantage proxy" in text
                or "brain-efficiency alignment proxy" in text
                or "energy_per_success_proxy" in text
                or "performance_energy_ratio_proxy" in text
                or "ann_cost_advantage_proxy" in text
                or "sparse_event_cost_score" in text
                or "brain_efficiency_alignment_proxy" in text
            )
        ):
            inferred.add("focus.efficiency_readiness.passed")
            matched = True
        if "meta_adaptation_parameter_integrity" in text:
            inferred.add("stage_c.meta_adaptation_parameter_integrity")
            matched = True
        if "temporal_self_distillation_stability" in text or "temporal self-distillation stability" in text:
            inferred.add("stage_c.temporal_self_distillation_stability")
            matched = True
        if "consolidation_readiness" in text:
            inferred.add("stage_d.consolidation_readiness")
            matched = True
        if "replay_recovery_integrity" in text:
            inferred.add("stage_d.replay_recovery_integrity")
            matched = True
        if "long_horizon_consolidation_retention" in text:
            inferred.add("stage_d.long_horizon_consolidation_retention")
            matched = True
        if "counterfactual_replay_selection_integrity" in text:
            inferred.add("stage_d.counterfactual_replay_selection_integrity")
            matched = True
        if "replay_upgrade_reindex_integrity" in text:
            inferred.add("stage_d.replay_upgrade_reindex_integrity")
            matched = True
        if "memory_health_index_integrity" in text:
            inferred.add("stage_d.memory_health_index_integrity")
            matched = True
        if "replay_noise_resilience_integrity" in text:
            inferred.add("stage_d.replay_noise_resilience_integrity")
            matched = True
        if "astro_modulation_stability" in text:
            inferred.add("stage_d.astro_modulation_stability")
            matched = True
        if "common_spike_space_integrity" in text:
            inferred.add("stage_e.common_spike_space_integrity")
            matched = True
        if "temporal_compression_efficiency" in text:
            inferred.add("stage_e.temporal_compression_efficiency")
            matched = True
        if "modality_temporal_budget_integrity" in text:
            inferred.add("stage_e.modality_temporal_budget_integrity")
            matched = True
        if "dendritic_context_gate_stability" in text:
            inferred.add("stage_e.dendritic_context_gate_stability")
            matched = True
        if "spiking_hjepa_latent_transition" in text:
            inferred.add("stage_e.spiking_hjepa_latent_transition")
            matched = True
        if "reverse_reasoning_trace_integrity" in text:
            inferred.add("stage_e.reverse_reasoning_trace_integrity")
            matched = True
        if "module_orchestration_integrity" in text:
            inferred.add("stage_e.module_orchestration_integrity")
            matched = True
        if "counterfactual_lane_integrity" in text:
            inferred.add("stage_e.counterfactual_lane_integrity")
            matched = True
        if "action_trace_observability" in text:
            inferred.add("stage_e.action_trace_observability")
            matched = True
        if "console script" in text:
            inferred.add("packaging.console_scripts")
            matched = True
        if "versions do not match" in text or "mismatched package versions" in text:
            inferred.add("packaging.version_match")
            matched = True
        if "release notes" in text:
            inferred.add("checklist.release_notes_reviewed")
            matched = True
        if "release checklist" in text:
            inferred.add("checklist.report_summary_review_ready")
            matched = True
        if not matched:
            inferred.add("release_gate.unknown_error")
    if not inferred:
        inferred.add("release_gate.errors")
    return sorted(inferred)


def build_release_gate_error_details(errors: List[str]) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    if not errors:
        return details

    minimum_pattern = re.compile(
        r"^Phase 3 Stage (?P<stage>[BCDE]) readiness did not satisfy .* "
        r"\((?P<check>metric\.[^,]+), (?:metric=(?P<metric>[^,]+), )?"
        r"value=(?P<value>[-+]?\d*\.?\d+), required>=(?P<required>[-+]?\d*\.?\d+)\)\.$"
    )
    threshold_pattern = re.compile(
        r"^(?P<metric>[a-z0-9_]+\.[a-z0-9_]+) dropped below threshold\.$",
        re.IGNORECASE,
    )
    phase5_metric_failure_pattern = re.compile(
        r"^Phase 5 entry readiness did not satisfy predictive-coding metric "
        r"(?P<metric>phase5_predictive_coding\.[a-z0-9_]+) "
        r"\(value=(?P<value>[-+]?\d*\.?\d+), required>=(?P<required>[-+]?\d*\.?\d+)\)\.$",
        re.IGNORECASE,
    )
    phase5_completion_missing_checks_pattern = re.compile(
        r"^Phase 5 completion gate check map is missing required checks: (?P<checks>.+)$",
        re.IGNORECASE,
    )
    external_validity_metric_failure_pattern = re.compile(
        r"^Real-data external validity metric did not satisfy release threshold "
        r"\((?P<metric>[a-z0-9_]+), value=(?P<value>[-+]?\d*\.?\d+), required>=(?P<required>[-+]?\d*\.?\d+)\)\.$",
        re.IGNORECASE,
    )
    external_validity_check_failure_pattern = re.compile(
        r"^Real-data external validity check failed: (?P<check>[a-z0-9_.]+)"
        r"(?: \(value=(?P<value>[-+]?\d*\.?\d+), "
        r"required(?P<operator>>=|<=)(?P<required>[-+]?\d*\.?\d+)\))?\.$",
        re.IGNORECASE,
    )
    external_validity_missing_checks_pattern = re.compile(
        r"^Real-data external validity report is missing required checks: (?P<checks>.+)\.$",
        re.IGNORECASE,
    )

    for index, error in enumerate(errors, start=1):
        text = str(error)
        inferred_checks = _infer_failed_checks_from_errors([text])
        payload: Dict[str, Any] = {
            "index": index,
            "error": text,
            "inferred_checks": inferred_checks,
            "category": inferred_checks[0] if inferred_checks else "release_gate.errors",
        }

        minimum_match = minimum_pattern.match(text)
        if minimum_match:
            stage_letter = str(minimum_match.group("stage")).upper()
            check_name = str(minimum_match.group("check"))
            metric_name = minimum_match.group("metric")
            if metric_name is None and check_name.startswith("metric."):
                metric_name = check_name[len("metric.") :]
            payload.update(
                {
                    "type": "minimum_threshold_failure",
                    "stage": f"stage_{stage_letter.lower()}",
                    "check_name": check_name,
                    "metric_name": str(metric_name or ""),
                    "actual_value": float(minimum_match.group("value")),
                    "required_value": float(minimum_match.group("required")),
                }
            )
            details.append(payload)
            continue

        threshold_match = threshold_pattern.match(text)
        if threshold_match:
            payload.update(
                {
                    "type": "metric_threshold_drop",
                    "metric_name": str(threshold_match.group("metric")),
                }
            )
            details.append(payload)
            continue

        phase5_failure_match = phase5_metric_failure_pattern.match(text)
        if phase5_failure_match:
            metric_name = str(phase5_failure_match.group("metric"))
            payload.update(
                {
                    "type": "minimum_threshold_failure",
                    "stage": "phase5",
                    "check_name": f"metric.{metric_name}",
                    "metric_name": metric_name,
                    "actual_value": float(phase5_failure_match.group("value")),
                    "required_value": float(phase5_failure_match.group("required")),
                    "category": inferred_checks[0] if inferred_checks else "phase5.entry_readiness",
                }
            )
            details.append(payload)
            continue

        phase5_completion_missing_checks_match = phase5_completion_missing_checks_pattern.match(text)
        if phase5_completion_missing_checks_match:
            raw_checks = str(phase5_completion_missing_checks_match.group("checks"))
            missing_checks = [item.strip() for item in raw_checks.split(",") if item.strip()]
            payload.update(
                {
                    "type": "missing_required_checks",
                    "stage": "phase5_completion",
                    "check_name": "phase5_completion.required_checks",
                    "missing_checks": missing_checks,
                    "category": inferred_checks[0] if inferred_checks else "phase5.completion_required_checks",
                }
            )
            details.append(payload)
            continue

        external_validity_metric_failure_match = external_validity_metric_failure_pattern.match(text)
        if external_validity_metric_failure_match:
            metric_name = str(external_validity_metric_failure_match.group("metric"))
            payload.update(
                {
                    "type": "minimum_threshold_failure",
                    "stage": "external_validity",
                    "check_name": f"metric.{metric_name}",
                    "metric_name": metric_name,
                    "actual_value": float(external_validity_metric_failure_match.group("value")),
                    "required_value": float(external_validity_metric_failure_match.group("required")),
                    "category": inferred_checks[0] if inferred_checks else "external_validity.report",
                }
            )
            details.append(payload)
            continue

        external_validity_check_failure_match = external_validity_check_failure_pattern.match(text)
        if external_validity_check_failure_match:
            check_name = str(external_validity_check_failure_match.group("check"))
            payload.update(
                {
                    "type": "required_check_failure",
                    "stage": "external_validity",
                    "check_name": check_name,
                    "category": inferred_checks[0] if inferred_checks else "external_validity.report",
                }
            )
            value = external_validity_check_failure_match.group("value")
            required = external_validity_check_failure_match.group("required")
            operator = external_validity_check_failure_match.group("operator")
            if value is not None and required is not None:
                payload["actual_value"] = float(value)
                payload["required_value"] = float(required)
                payload["threshold_operator"] = str(operator or "")
            details.append(payload)
            continue

        external_validity_missing_checks_match = external_validity_missing_checks_pattern.match(text)
        if external_validity_missing_checks_match:
            raw_checks = str(external_validity_missing_checks_match.group("checks"))
            missing_checks = [item.strip() for item in raw_checks.split(",") if item.strip()]
            payload.update(
                {
                    "type": "missing_required_checks",
                    "stage": "external_validity",
                    "check_name": "external_validity.required_checks",
                    "missing_checks": missing_checks,
                    "category": inferred_checks[0] if inferred_checks else "external_validity.report",
                }
            )
            details.append(payload)
            continue

        payload["type"] = "general_error"
        details.append(payload)

    return details


def build_release_gate_error_details_summary(
    error_details: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(error_details, list):
        return {
            "total": 0,
            "by_type": {},
            "by_category": {},
            "by_metric": {},
            "top_types": [],
            "top_categories": [],
            "top_metrics": [],
        }

    by_type: Dict[str, int] = {}
    by_category: Dict[str, int] = {}
    by_metric: Dict[str, int] = {}
    total = 0

    for item in error_details:
        if not isinstance(item, dict):
            continue
        total += 1
        detail_type = str(item.get("type", "general_error"))
        category = str(item.get("category", "release_gate.errors"))
        metric_name = str(item.get("metric_name", "")).strip()

        by_type[detail_type] = int(by_type.get(detail_type, 0)) + 1
        by_category[category] = int(by_category.get(category, 0)) + 1
        if metric_name:
            by_metric[metric_name] = int(by_metric.get(metric_name, 0)) + 1

    sorted_types = sorted(by_type.items(), key=lambda item: (-int(item[1]), str(item[0])))
    sorted_categories = sorted(by_category.items(), key=lambda item: (-int(item[1]), str(item[0])))
    sorted_metrics = sorted(by_metric.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return {
        "total": int(total),
        "by_type": dict(sorted(by_type.items())),
        "by_category": dict(sorted(by_category.items())),
        "by_metric": dict(sorted(by_metric.items())),
        "top_types": [{"name": str(name), "count": int(count)} for name, count in sorted_types[:5]],
        "top_categories": [{"name": str(name), "count": int(count)} for name, count in sorted_categories[:5]],
        "top_metrics": [{"name": str(name), "count": int(count)} for name, count in sorted_metrics[:5]],
    }


def build_release_gate_failure_focus(
    error_details_summary: Dict[str, Any],
    repair_plan: Dict[str, Any],
) -> Dict[str, Any]:
    summary = error_details_summary if isinstance(error_details_summary, dict) else {}
    plan = repair_plan if isinstance(repair_plan, dict) else {}

    top_categories = summary.get("top_categories", [])
    top_metrics = summary.get("top_metrics", [])
    selected_actions = (
        plan.get("selected_actions", [])
        if isinstance(plan.get("selected_actions"), list)
        else []
    )

    primary_category = ""
    primary_metric = ""
    if isinstance(top_categories, list) and top_categories and isinstance(top_categories[0], dict):
        primary_category = str(top_categories[0].get("name", "")).strip()
    if isinstance(top_metrics, list) and top_metrics and isinstance(top_metrics[0], dict):
        primary_metric = str(top_metrics[0].get("name", "")).strip()

    secondary_category = ""
    if (
        isinstance(top_categories, list)
        and len(top_categories) > 1
        and isinstance(top_categories[1], dict)
    ):
        secondary_category = str(top_categories[1].get("name", "")).strip()

    primary_action = {}
    if selected_actions and isinstance(selected_actions[0], dict):
        primary_action = {
            "title": str(selected_actions[0].get("title", "")).strip(),
            "command": str(selected_actions[0].get("command", "")).strip(),
            "priority": str(selected_actions[0].get("priority", "")).strip(),
        }

    total = int(summary.get("total", 0) or 0)
    primary_count = 0
    if isinstance(top_categories, list) and top_categories and isinstance(top_categories[0], dict):
        primary_count = int(top_categories[0].get("count", 0) or 0)
    concentration = (float(primary_count) / float(total)) if total > 0 else 0.0
    action_bonus = 0.15 if primary_action else 0.0
    confidence = min(max(concentration + action_bonus, 0.0), 1.0)

    return {
        "primary_category": primary_category,
        "secondary_category": secondary_category,
        "primary_metric": primary_metric,
        "primary_action": primary_action,
        "confidence": float(round(confidence, 3)),
    }


def _build_fallback_actions_for_uncovered_checks(uncovered_checks: List[str]) -> List[Dict[str, Any]]:
    if not uncovered_checks:
        return []
    checks = sorted({str(item) for item in uncovered_checks if str(item).strip()})
    return [
        {
            "step": 1,
            "title": "Collect Focused Gate Diagnostics",
            "command": "python scripts/eval/release_soak.py --profile release --include-accuracy",
            "priority": "medium",
            "expected_effect": "Refreshes managed diagnostics and checklist artifacts for unresolved checks.",
            "reason": "Unresolved checks remain after the minimal repair plan.",
            "affected_checks": checks,
        },
        {
            "step": 2,
            "title": "Re-run Release Gate With Fresh Artifacts",
            "command": "python scripts/eval/release_gate.py",
            "priority": "low",
            "expected_effect": "Re-evaluates release readiness after fallback diagnostics.",
            "reason": "Confirms whether unresolved checks were recovered by the fallback pass.",
            "affected_checks": checks,
        },
    ]


def build_release_gate_repair_plan(errors: List[str]) -> Dict[str, Any]:
    actions = suggest_release_gate_recovery_actions(errors)
    target_checks = _infer_failed_checks_from_errors(errors)
    if not actions:
        fallback_actions = _build_fallback_actions_for_uncovered_checks(target_checks)
        return {
            "selected_actions": [],
            "covered_checks": [],
            "uncovered_checks": target_checks,
            "fallback_actions": fallback_actions,
            "coverage_ratio": 0.0 if target_checks else 1.0,
            "estimated_steps": 0,
        }

    priority_order = {"high": 0, "medium": 1, "low": 2}
    universe = target_checks if target_checks else ["release_gate.errors"]

    remaining = set(universe)
    selected: List[Dict[str, Any]] = []
    candidate_actions = [dict(action) for action in actions if isinstance(action, dict)]

    while remaining and candidate_actions:
        ranked_candidates = []
        for action in candidate_actions:
            affected = {
                str(check)
                for check in (
                    action.get("affected_checks", [])
                    if isinstance(action.get("affected_checks", []), list)
                    else []
                )
                if str(check).strip()
            }
            uncovered = affected.intersection(remaining)
            ranked_candidates.append(
                (
                    len(uncovered),
                    -priority_order.get(str(action.get("priority", "low")).lower(), 2),
                    str(action.get("title", "")),
                    action,
                    uncovered,
                )
            )

        ranked_candidates.sort(reverse=True)
        best_count, _, _, best_action, best_uncovered = ranked_candidates[0]
        if best_count <= 0:
            break
        selected.append(best_action)
        remaining -= best_uncovered
        candidate_actions = [item for item in candidate_actions if item.get("title") != best_action.get("title")]

    if remaining:
        for action in candidate_actions:
            selected.append(action)
            affected = {
                str(check)
                for check in (
                    action.get("affected_checks", [])
                    if isinstance(action.get("affected_checks", []), list)
                    else []
                )
                if str(check).strip()
            }
            remaining -= affected
            if not remaining:
                break

    selected_steps = []
    covered = set()
    for index, action in enumerate(selected, start=1):
        affected = sorted(
            {
                str(check)
                for check in (
                    action.get("affected_checks", [])
                    if isinstance(action.get("affected_checks", []), list)
                    else []
                )
                if str(check).strip()
            }
        )
        covered.update(affected)
        step_action = dict(action)
        step_action["step"] = index
        step_action["affected_checks"] = affected
        selected_steps.append(step_action)

    coverage_ratio = len(covered.intersection(set(universe))) / max(len(universe), 1)
    uncovered_checks = sorted(set(universe).difference(covered))
    fallback_actions = _build_fallback_actions_for_uncovered_checks(uncovered_checks)
    return {
        "selected_actions": selected_steps,
        "covered_checks": sorted(covered.intersection(set(universe))),
        "uncovered_checks": uncovered_checks,
        "fallback_actions": fallback_actions,
        "coverage_ratio": float(coverage_ratio),
        "estimated_steps": len(selected_steps),
    }


def build_iterative_release_gate_repair_plan(
    errors: List[str],
    execution_log: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    base_plan = build_release_gate_repair_plan(errors)
    target_checks = set(_infer_failed_checks_from_errors(errors))
    remaining_checks = set(target_checks)

    selected_actions = (
        base_plan.get("selected_actions", [])
        if isinstance(base_plan.get("selected_actions"), list)
        else []
    )
    fallback_actions = (
        base_plan.get("fallback_actions", [])
        if isinstance(base_plan.get("fallback_actions"), list)
        else []
    )
    action_catalog = [
        dict(item)
        for item in [*selected_actions, *fallback_actions]
        if isinstance(item, dict)
    ]

    logs = [
        dict(item)
        for item in (execution_log if isinstance(execution_log, list) else [])
        if isinstance(item, dict)
    ]
    success_statuses = {"success", "passed", "done"}
    failure_statuses = {"failed", "error", "timeout"}
    successful_commands = set()
    failed_commands = set()
    executed_steps = 0
    successful_steps = 0
    failed_steps = 0

    for entry in logs:
        command = str(entry.get("command", "")).strip()
        if not command:
            continue
        executed_steps += 1
        status = str(entry.get("status", "")).strip().lower()
        covered_checks = (
            {str(item) for item in entry.get("covered_checks", []) if str(item).strip()}
            if isinstance(entry.get("covered_checks"), list)
            else set()
        )
        if status in success_statuses:
            successful_steps += 1
            successful_commands.add(command)
            remaining_checks -= covered_checks
        elif status in failure_statuses:
            failed_steps += 1
            failed_commands.add(command)
            remaining_checks |= covered_checks

    for action in action_catalog:
        command = str(action.get("command", "")).strip()
        affected_checks = (
            {str(item) for item in action.get("affected_checks", []) if str(item).strip()}
            if isinstance(action.get("affected_checks"), list)
            else set()
        )
        if not command:
            continue
        if command in successful_commands:
            remaining_checks -= affected_checks
        elif command in failed_commands:
            remaining_checks |= affected_checks

    priority_order = {"high": 0, "medium": 1, "low": 2}
    ranked = []
    for action in action_catalog:
        command = str(action.get("command", "")).strip()
        if command in successful_commands:
            continue
        affected_checks = (
            {str(item) for item in action.get("affected_checks", []) if str(item).strip()}
            if isinstance(action.get("affected_checks"), list)
            else set()
        )
        if remaining_checks and not affected_checks.intersection(remaining_checks):
            continue
        ranked.append(
            (
                priority_order.get(str(action.get("priority", "low")).lower(), 2),
                str(action.get("title", "")),
                dict(action),
            )
        )
    ranked.sort()

    next_actions = []
    for index, (_, _, action) in enumerate(ranked, start=1):
        payload = dict(action)
        payload["step"] = index
        next_actions.append(payload)

    completed = bool(not remaining_checks)
    if completed:
        next_actions = []

    coverage_ratio = (
        (len(target_checks) - len(remaining_checks.intersection(target_checks))) / max(len(target_checks), 1)
        if target_checks
        else 1.0
    )
    stalled = bool((not completed) and remaining_checks and not next_actions)
    if not target_checks:
        stop_reason = "no_target_checks"
    elif completed:
        stop_reason = "auto_stopped_completed"
    elif stalled:
        stop_reason = "stalled_no_candidate_actions"
    else:
        stop_reason = "pending_actions"

    if completed:
        next_step_hint = "No further action required. Re-run release_gate only for verification."
    elif next_actions:
        first = next_actions[0]
        next_step_hint = str(first.get("command", "python scripts/eval/release_gate.py")).strip()
    else:
        next_step_hint = "python scripts/eval/release_soak.py --profile release --include-accuracy"

    return {
        "base_plan": base_plan,
        "iteration": executed_steps + 1 if target_checks else 1,
        "executed_steps": executed_steps,
        "successful_steps": successful_steps,
        "failed_steps": failed_steps,
        "remaining_checks": sorted(remaining_checks),
        "next_actions": next_actions,
        "coverage_ratio": float(coverage_ratio),
        "stalled": stalled,
        "completed": completed,
        "auto_stopped": completed,
        "stop_reason": stop_reason,
        "next_step_hint": next_step_hint,
    }


def collect_release_gate_artifacts(
    errors: List[str],
    execution_log: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    recovery_actions = suggest_release_gate_recovery_actions(errors)
    repair_plan = build_release_gate_repair_plan(errors)
    iterative_plan = build_iterative_release_gate_repair_plan(errors, execution_log=execution_log)
    error_details = build_release_gate_error_details(errors)
    error_details_summary = build_release_gate_error_details_summary(error_details)
    failure_focus = build_release_gate_failure_focus(error_details_summary, repair_plan)
    return {
        "recovery_actions": recovery_actions,
        "repair_plan": repair_plan,
        "iterative_repair_plan": iterative_plan,
        "error_details": error_details,
        "error_details_summary": error_details_summary,
        "failure_focus": failure_focus,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate release readiness from soak report and metadata.")
    parser.add_argument(
        "--report-path",
        default=DEFAULT_REPORT_PATH,
        help="Managed path to the release soak report JSON.",
    )
    parser.add_argument(
        "--accuracy-report-path",
        default=DEFAULT_ACCURACY_REPORT_PATH,
        help="Managed path to the Phase 3 accuracy suite JSON.",
    )
    parser.add_argument(
        "--external-validity-report-path",
        default=DEFAULT_EXTERNAL_VALIDITY_REPORT_PATH,
        help="Managed path to the real-data external validity JSON.",
    )
    parser.add_argument(
        "--phase5-completion-gate-report-path",
        default=DEFAULT_PHASE5_COMPLETION_GATE_REPORT_PATH,
        help="Managed path to the Phase 5 completion gate JSON.",
    )
    parser.add_argument(
        "--skip-phase5-completion-gate",
        action="store_true",
        help="Skip validation of the Phase 5 completion gate report.",
    )
    parser.add_argument(
        "--skip-external-validity-gate",
        action="store_true",
        help="Skip validation of the real-data external validity report.",
    )
    parser.add_argument(
        "--skip-accuracy-gate",
        action="store_true",
        help="Skip validation of both standalone and embedded Phase 3 accuracy reports.",
    )
    parser.add_argument(
        "--repair-log-path",
        default=DEFAULT_REPAIR_LOG_PATH,
        help="Managed path to the repair execution log JSON/JSONL.",
    )
    parser.add_argument(
        "--repair-plan-path",
        default=DEFAULT_REPAIR_PLAN_PATH,
        help="Managed output path for recovery/repair planning artifact JSON.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.report_path):
        print(f"Release gate failed: soak report not found at {args.report_path}")
        raise SystemExit(1)

    with open(args.report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    errors = []
    errors.extend(validate_release_report(report, skip_embedded_accuracy=args.skip_accuracy_gate))
    errors.extend(validate_packaging_metadata(PROJECT_ROOT))
    if not args.skip_accuracy_gate:
        if not os.path.exists(args.accuracy_report_path):
            errors.append(
                f"Phase 3 accuracy report not found at {args.accuracy_report_path}"
            )
        else:
            with open(args.accuracy_report_path, "r", encoding="utf-8") as handle:
                accuracy_report = json.load(handle)
            if not isinstance(accuracy_report, dict):
                errors.append(
                    f"Phase 3 accuracy report at {args.accuracy_report_path} is not a JSON object."
                )
            else:
                errors.extend(validate_phase3_accuracy_report(accuracy_report))
    if not args.skip_phase5_completion_gate:
        if not os.path.exists(args.phase5_completion_gate_report_path):
            errors.append(
                f"Phase 5 completion gate report not found at {args.phase5_completion_gate_report_path}"
            )
        else:
            with open(args.phase5_completion_gate_report_path, "r", encoding="utf-8") as handle:
                phase5_completion_report = json.load(handle)
            if not isinstance(phase5_completion_report, dict):
                errors.append(
                    f"Phase 5 completion gate report at {args.phase5_completion_gate_report_path} is not a JSON object."
                )
            else:
                errors.extend(validate_phase5_completion_gate_report(phase5_completion_report))
    if not args.skip_external_validity_gate:
        if not os.path.exists(args.external_validity_report_path):
            errors.append(
                f"Real-data external validity report not found at {args.external_validity_report_path}"
            )
        else:
            with open(args.external_validity_report_path, "r", encoding="utf-8") as handle:
                external_validity_report = json.load(handle)
            if not isinstance(external_validity_report, dict):
                errors.append(
                    f"Real-data external validity report at {args.external_validity_report_path} is not a JSON object."
                )
            else:
                errors.extend(validate_external_validity_report(external_validity_report))

    execution_log = load_repair_execution_log(args.repair_log_path)
    artifacts = collect_release_gate_artifacts(errors, execution_log=execution_log)
    repair_plan_output = {
        "errors": list(errors),
        "repair_log_path": os.path.abspath(args.repair_log_path),
        "execution_log": execution_log,
        "artifacts": artifacts,
    }
    repair_plan_path = ensure_parent_directory(args.repair_plan_path)
    with open(repair_plan_path, "w", encoding="utf-8") as handle:
        json.dump(repair_plan_output, handle, indent=2, ensure_ascii=False)

    if errors:
        print("Release gate failed:")
        for item in errors:
            print(f"- {item}")
        actions = artifacts.get("recovery_actions", [])
        if actions:
            print("Suggested recovery actions:")
            for action in actions:
                affected_checks = action.get("affected_checks", [])
                affected_text = ", ".join(affected_checks) if isinstance(affected_checks, list) else ""
                print(
                    f"- {action.get('title', '')}: {action.get('command', '')} "
                    f"(priority={action.get('priority', '')}, effect={action.get('expected_effect', '')}, "
                    f"affected_checks={affected_text}, reason={action.get('reason', '')})"
                )
        repair_plan = artifacts.get("repair_plan", {})
        selected = repair_plan.get("selected_actions", [])
        if isinstance(selected, list) and selected:
            print("Suggested minimal repair plan:")
            for action in selected:
                checks = action.get("affected_checks", [])
                checks_text = ", ".join(checks) if isinstance(checks, list) else ""
                print(
                    f"- step {int(action.get('step', 0) or 0)}: {action.get('title', '')} -> "
                    f"{action.get('command', '')} (covers={checks_text})"
                )
        fallback_actions = repair_plan.get("fallback_actions", [])
        if isinstance(fallback_actions, list) and fallback_actions:
            print("Fallback plan for uncovered checks:")
            for action in fallback_actions:
                checks = action.get("affected_checks", [])
                checks_text = ", ".join(checks) if isinstance(checks, list) else ""
                print(
                    f"- step {int(action.get('step', 0) or 0)}: {action.get('title', '')} -> "
                    f"{action.get('command', '')} (covers={checks_text})"
                )
        iterative_plan = artifacts.get("iterative_repair_plan", {})
        next_actions = iterative_plan.get("next_actions", [])
        if isinstance(next_actions, list) and next_actions:
            print("Iterative repair loop (next actions):")
            for action in next_actions:
                checks = action.get("affected_checks", [])
                checks_text = ", ".join(checks) if isinstance(checks, list) else ""
                print(
                    f"- step {int(action.get('step', 0) or 0)}: {action.get('title', '')} -> "
                    f"{action.get('command', '')} (covers={checks_text})"
                )
        print(f"Saved repair artifact: {repair_plan_path}")
        raise SystemExit(1)

    iterative_plan = artifacts.get("iterative_repair_plan", {})
    print("Release gate passed.")
    print(f"Saved repair artifact: {repair_plan_path}")
    print(
        "Iterative status: "
        f"completed={bool(iterative_plan.get('completed', False))} "
        f"remaining_checks={len(iterative_plan.get('remaining_checks', [])) if isinstance(iterative_plan.get('remaining_checks', []), list) else 0}"
    )


if __name__ == "__main__":
    main()
