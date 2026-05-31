import json
import time
from typing import Any, Dict, List, Optional

from ..utils.project_paths import ensure_parent_directory


DEFAULT_PHASE3_TREND_TOLERANCE = 0.025


PHASE3_OBSERVED_ONLY_TREND_SUFFIXES = (
    ".average_quality_per_kparam",
    ".average_quality_per_mb",
    ".bounded_latency_score",
    "_observed",
)

PHASE3_OBSERVED_ONLY_TREND_FRAGMENTS = (
    ".manifold_",
    ".causal_route_sparsity",
    ".withheld_trajectory_recall",
)

COGNITIVE_MANIFOLD_TRACE_METRIC_NAMES = (
    "manifold_trace_support_observed",
    "manifold_trace_recall_observed",
    "manifold_trace_scan_budget_observed",
    "manifold_trace_index_scan_reduction_observed",
    "manifold_trace_candidate_guard_observed",
)

CognitiveManifoldTraceMetrics = Dict[str, float]

COGNITIVE_DELTA_MEMORY_METRIC_NAMES = (
    "delta_memory_steering_integrity_observed",
    "delta_memory_counterfactual_isolation_observed",
    "delta_memory_trace_observability_observed",
)

CognitiveDeltaMemoryMetrics = Dict[str, float]

COGNITIVE_LEJEPA_LATENT_HEALTH_METRIC_NAMES = (
    "lejepa_linear_identifiability_proxy_observed",
    "lejepa_latent_whitening_health_observed",
    "lejepa_factor_disentanglement_observed",
    "lejepa_latent_planning_consistency_observed",
    "lejepa_positive_pair_alignment_observed",
)

CognitiveLejepaLatentHealthMetrics = Dict[str, float]

COGNITIVE_LINEAR_SNN_FUSION_METRIC_NAMES = (
    "predictive_spike_entropy_reduction_observed",
    "phase_binding_coincidence_integrity_observed",
    "forward_only_local_update_stability_observed",
) + COGNITIVE_LEJEPA_LATENT_HEALTH_METRIC_NAMES

CognitiveLinearSNNFusionMetrics = Dict[str, float]

COGNITIVE_PLASTIC_SUBMODEL_METRIC_NAMES = (
    "plastic_submodel_registry_integrity_observed",
    "dynamic_submodel_route_integrity_observed",
    "submodel_relearning_trace_integrity_observed",
    "interpretable_submodel_concept_trace_observed",
    "runtime_submodel_route_action_grounding_observed",
    "runtime_submodel_counterfactual_route_separation_observed",
    "runtime_submodel_concept_trace_observed",
    "submodel_intervention_trace_integrity_observed",
    "submodel_ablation_effect_observed",
    "submodel_reactivation_recovery_observed",
    "submodel_credit_assignment_trace_integrity_observed",
    "submodel_credit_selectivity_observed",
    "submodel_credit_state_budget_observed",
    "runtime_submodel_local_credit_assignment_observed",
    "runtime_submodel_feedback_trace_observed",
    "submodel_structural_adaptation_trace_integrity_observed",
    "submodel_structural_growth_bounded_observed",
    "submodel_structural_pruning_observed",
    "submodel_scientific_hypothesis_trace_integrity_observed",
    "submodel_counterexample_revision_observed",
    "submodel_scientific_model_budget_observed",
    "submodel_hypothesis_bank_integrity_observed",
    "submodel_open_ended_selection_observed",
    "submodel_hypothesis_bank_budget_observed",
    "micro_turn_event_budget_observed",
    "foreground_background_context_handoff_observed",
    "interrupt_recovery_trace_observed",
    "simultaneous_stream_route_integrity_observed",
    "time_aligned_backchannel_policy_observed",
    "phase_assigned_submodel_route_observed",
    "uncertainty_bucket_specialization_observed",
    "denoising_correction_trace_integrity_observed",
    "block_independent_local_update_budget_observed",
)

CognitivePlasticSubmodelMetrics = Dict[str, float]

COGNITIVE_STAGE_E_ARCHITECTURE_INTEGRATION_METRIC_NAMES = (
    "micro_turn_event_budget_observed",
    "foreground_background_context_handoff_observed",
    "interrupt_recovery_trace_observed",
    "simultaneous_stream_route_integrity_observed",
    "time_aligned_backchannel_policy_observed",
    "phase_assigned_submodel_route_observed",
    "uncertainty_bucket_specialization_observed",
    "denoising_correction_trace_integrity_observed",
    "block_independent_local_update_budget_observed",
)

CognitiveStageEArchitectureIntegrationMetrics = Dict[str, float]


def phase3_component_metrics(report: Dict[str, Any], component_name: str) -> Dict[str, Any]:
    component_reports = report.get("component_reports", {})
    if not isinstance(component_reports, dict):
        return {}
    component = component_reports.get(str(component_name), {})
    if not isinstance(component, dict):
        return {}
    metrics = component.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def extract_cognitive_manifold_trace_metrics(
    report: Dict[str, Any],
) -> CognitiveManifoldTraceMetrics:
    metrics = phase3_component_metrics(report, "cognitive_runtime")
    return {
        metric_name: float(metrics.get(metric_name, 0.0) or 0.0)
        for metric_name in COGNITIVE_MANIFOLD_TRACE_METRIC_NAMES
    }


def extract_cognitive_delta_memory_metrics(
    report: Dict[str, Any],
) -> CognitiveDeltaMemoryMetrics:
    metrics = phase3_component_metrics(report, "cognitive_runtime")
    return {
        metric_name: float(metrics.get(metric_name, 0.0) or 0.0)
        for metric_name in COGNITIVE_DELTA_MEMORY_METRIC_NAMES
    }


def extract_cognitive_linear_snn_fusion_metrics(
    report: Dict[str, Any],
) -> CognitiveLinearSNNFusionMetrics:
    metrics = phase3_component_metrics(report, "cognitive_runtime")
    return {
        metric_name: float(metrics.get(metric_name, 0.0) or 0.0)
        for metric_name in COGNITIVE_LINEAR_SNN_FUSION_METRIC_NAMES
    }


def extract_cognitive_plastic_submodel_metrics(
    report: Dict[str, Any],
) -> CognitivePlasticSubmodelMetrics:
    metrics = dict(phase3_component_metrics(report, "cognitive_runtime"))
    if isinstance(report, dict):
        focus_summary = report.get("focus_summary", {})
        if isinstance(focus_summary, dict):
            cognitive_focus = focus_summary.get("cognitive_runtime_readiness", {})
            if isinstance(cognitive_focus, dict):
                focus_metrics = cognitive_focus.get("plastic_submodel_observed_metrics", {})
                if isinstance(focus_metrics, dict):
                    metrics.update({
                        str(key).replace("cognitive_runtime.", ""): value
                        for key, value in focus_metrics.items()
                    })
    return {
        metric_name: float(metrics.get(metric_name, 0.0) or 0.0)
        for metric_name in COGNITIVE_PLASTIC_SUBMODEL_METRIC_NAMES
    }


def extract_cognitive_stage_e_architecture_integration_metrics(
    report: Dict[str, Any],
) -> CognitiveStageEArchitectureIntegrationMetrics:
    metrics = dict(phase3_component_metrics(report, "cognitive_runtime"))
    if isinstance(report, dict):
        focus_summary = report.get("focus_summary", {})
        if isinstance(focus_summary, dict):
            cognitive_focus = focus_summary.get("cognitive_runtime_readiness", {})
            if isinstance(cognitive_focus, dict):
                focus_metrics = cognitive_focus.get("plastic_submodel_observed_metrics", {})
                if isinstance(focus_metrics, dict):
                    metrics.update({
                        str(key).replace("cognitive_runtime.", ""): value
                        for key, value in focus_metrics.items()
                    })
    return {
        metric_name: float(metrics.get(metric_name, 0.0) or 0.0)
        for metric_name in COGNITIVE_STAGE_E_ARCHITECTURE_INTEGRATION_METRIC_NAMES
    }


def compact_neuromorphic_profile_trend(
    trend: Dict[str, Any],
    limit: int = 5,
) -> Dict[str, Any]:
    if not isinstance(trend, dict):
        trend = {}
    missing_profiles = [
        str(profile)
        for profile in trend.get("missing_profiles", [])
        if str(profile)
    ] if isinstance(trend.get("missing_profiles", []), list) else []
    regressions = (
        [dict(item) for item in trend.get("regressions", []) if isinstance(item, dict)]
        if isinstance(trend.get("regressions", []), list)
        else []
    )
    policy_changes = (
        [dict(item) for item in trend.get("policy_changes", []) if isinstance(item, dict)]
        if isinstance(trend.get("policy_changes", []), list)
        else []
    )

    regression_details: List[str] = []
    for item in regressions:
        profile = str(item.get("profile", "") or "")
        kind = str(item.get("kind", "") or "")
        check = str(item.get("check", "") or "")
        if not profile or not kind:
            continue
        detail = f"{profile}:{kind}"
        if check:
            detail += f":{check}"
        regression_details.append(detail)
    for profile in missing_profiles:
        regression_details.append(f"{profile}:missing_profile")
    regression_details = list(dict.fromkeys(regression_details))

    policy_change_details: List[str] = []
    for item in policy_changes:
        profile = str(item.get("profile", "") or "")
        previous = str(item.get("previous", "") or "")
        current = str(item.get("current", "") or "")
        if profile and previous and current:
            policy_change_details.append(f"{profile}:{previous}->{current}")
    policy_change_details = list(dict.fromkeys(policy_change_details))

    compact_limit = max(1, int(limit))
    return {
        "missing_profiles": missing_profiles,
        "regression_details": regression_details,
        "policy_change_details": policy_change_details,
        "regression_detail_line": (
            ",".join(regression_details[:compact_limit]) if regression_details else "none"
        ),
        "policy_change_detail_line": (
            ",".join(policy_change_details[:compact_limit]) if policy_change_details else "none"
        ),
    }


def _build_cognitive_observed_metric_trend(
    *,
    schema: str,
    current_metrics: Dict[str, float],
    previous_metrics: Dict[str, float],
    regression_tolerance: float,
) -> Dict[str, Any]:
    regressions: List[Dict[str, Any]] = []
    improvements: List[Dict[str, Any]] = []
    unchanged: List[str] = []
    new_metrics: List[str] = []
    tolerance = float(max(regression_tolerance, 0.0))

    for metric_name, current_value in current_metrics.items():
        if metric_name not in previous_metrics:
            new_metrics.append(metric_name)
            continue
        previous_value = float(previous_metrics.get(metric_name, 0.0) or 0.0)
        delta = float(current_value) - previous_value
        record = {
            "metric": metric_name,
            "previous": previous_value,
            "current": float(current_value),
            "delta": delta,
        }
        if delta < -tolerance:
            regressions.append(record)
        elif delta > tolerance:
            improvements.append(record)
        else:
            unchanged.append(metric_name)

    return {
        "schema": schema,
        "observed_only": True,
        "release_gate_blocking": False,
        "has_previous": bool(previous_metrics),
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
        "unchanged_count": len(unchanged),
        "new_metric_count": len(new_metrics),
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
        "new_metrics": new_metrics,
    }


def build_cognitive_linear_snn_fusion_observed_trend(
    current_report: Dict[str, Any],
    previous_report: Optional[Dict[str, Any]] = None,
    regression_tolerance: float = DEFAULT_PHASE3_TREND_TOLERANCE,
) -> Dict[str, Any]:
    current_metrics = extract_cognitive_linear_snn_fusion_metrics(current_report)
    previous_metrics = (
        extract_cognitive_linear_snn_fusion_metrics(previous_report)
        if isinstance(previous_report, dict)
        else {}
    )
    return _build_cognitive_observed_metric_trend(
        schema="sara-cognitive-linear-snn-fusion-observed-trend-v1",
        current_metrics=current_metrics,
        previous_metrics=previous_metrics,
        regression_tolerance=regression_tolerance,
    )


def build_cognitive_stage_e_architecture_integration_observed_trend(
    current_report: Dict[str, Any],
    previous_report: Optional[Dict[str, Any]] = None,
    regression_tolerance: float = DEFAULT_PHASE3_TREND_TOLERANCE,
) -> Dict[str, Any]:
    current_metrics = extract_cognitive_stage_e_architecture_integration_metrics(current_report)
    previous_metrics = (
        extract_cognitive_stage_e_architecture_integration_metrics(previous_report)
        if isinstance(previous_report, dict)
        else {}
    )
    return _build_cognitive_observed_metric_trend(
        schema="sara-cognitive-stage-e-architecture-integration-observed-trend-v1",
        current_metrics=current_metrics,
        previous_metrics=previous_metrics,
        regression_tolerance=regression_tolerance,
    )


def is_phase3_gate_trend_metric(metric_name: str) -> bool:
    normalized = str(metric_name)
    if any(normalized.endswith(suffix) for suffix in PHASE3_OBSERVED_ONLY_TREND_SUFFIXES):
        return False
    return not any(fragment in normalized for fragment in PHASE3_OBSERVED_ONLY_TREND_FRAGMENTS)


def flatten_phase3_metrics(report: Dict[str, Any]) -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    component_reports = report.get("component_reports", {})
    if not isinstance(component_reports, dict):
        component_reports = {}

    for component_name, component_report in component_reports.items():
        if not isinstance(component_report, dict):
            continue
        metrics = component_report.get("metrics", {})
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                try:
                    flattened[f"{component_name}.{metric_name}"] = float(value)
                except (TypeError, ValueError):
                    continue
        try:
            flattened[f"{component_name}.overall_score"] = float(
                component_report.get("overall_score", 0.0)
            )
        except (TypeError, ValueError):
            pass

    focus_summary = report.get("focus_summary", {})
    if isinstance(focus_summary, dict):
        for focus_name, focus_report in focus_summary.items():
            if not isinstance(focus_report, dict):
                continue
            try:
                flattened[f"focus.{focus_name}.score"] = float(focus_report.get("score", 0.0))
            except (TypeError, ValueError):
                pass
            metrics = focus_report.get("metrics", {})
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    try:
                        flattened[f"focus.{focus_name}.{metric_name}"] = float(value)
                    except (TypeError, ValueError):
                        continue

    try:
        flattened["suite.overall_score"] = float(report.get("overall_score", 0.0))
    except (TypeError, ValueError):
        pass
    return flattened


def build_phase3_trend(
    current_report: Dict[str, Any],
    previous_report: Optional[Dict[str, Any]] = None,
    regression_tolerance: float = DEFAULT_PHASE3_TREND_TOLERANCE,
) -> Dict[str, Any]:
    current_metrics = flatten_phase3_metrics(current_report)
    previous_metrics = flatten_phase3_metrics(previous_report or {})

    regressions: List[Dict[str, Any]] = []
    improvements: List[Dict[str, Any]] = []
    unchanged: List[str] = []
    new_metrics: List[str] = []

    for metric_name, current_value in sorted(current_metrics.items()):
        if metric_name not in previous_metrics:
            new_metrics.append(metric_name)
            continue

        previous_value = previous_metrics[metric_name]
        delta = current_value - previous_value
        record = {
            "metric": metric_name,
            "previous": previous_value,
            "current": current_value,
            "delta": delta,
        }
        if delta < -regression_tolerance:
            regressions.append(record)
        elif delta > regression_tolerance:
            improvements.append(record)
        else:
            unchanged.append(metric_name)

    return {
        "has_previous": bool(previous_metrics),
        "regression_count": len(regressions),
        "gate_regression_count": len(
            [item for item in regressions if is_phase3_gate_trend_metric(str(item.get("metric", "")))]
        ),
        "improvement_count": len(improvements),
        "unchanged_count": len(unchanged),
        "new_metric_count": len(new_metrics),
        "regressions": regressions,
        "gate_regressions": [
            item for item in regressions if is_phase3_gate_trend_metric(str(item.get("metric", "")))
        ],
        "improvements": improvements,
        "unchanged": unchanged,
        "new_metrics": new_metrics,
    }


def load_phase3_history(history_path: str) -> List[Dict[str, Any]]:
    try:
        with open(history_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def latest_phase3_report(history_path: str) -> Optional[Dict[str, Any]]:
    history = load_phase3_history(history_path)
    if not history:
        return None
    return history[-1]


def append_phase3_history(
    history_path: str,
    report: Dict[str, Any],
    max_entries: int = 50,
) -> List[Dict[str, Any]]:
    history = load_phase3_history(history_path)
    entry = dict(report)
    entry.setdefault("recorded_at", time.time())
    history.append(entry)
    if max_entries > 0:
        history = history[-max_entries:]

    resolved_path = ensure_parent_directory(history_path)
    with open(resolved_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, ensure_ascii=False)
    return history
