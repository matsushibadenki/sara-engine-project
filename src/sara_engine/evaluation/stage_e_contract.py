# Directory Path: src/sara_engine/evaluation/stage_e_contract.py
# English Title: Stage E Modular Cognitive Runtime Readiness Contract
# Purpose/Content: Shared constants for Stage E minimum checks used by phase3 suite, release gate, and soak summaries.

from typing import Dict, List

from sara_engine.evaluation.phase3_tracking import (
    COGNITIVE_DELTA_MEMORY_METRIC_NAMES,
    COGNITIVE_LINEAR_SNN_FUSION_METRIC_NAMES,
    COGNITIVE_MANIFOLD_TRACE_METRIC_NAMES,
    COGNITIVE_PLASTIC_SUBMODEL_METRIC_NAMES,
    COGNITIVE_STAGE_E_ARCHITECTURE_INTEGRATION_METRIC_NAMES,
)


STAGE_E_MINIMUM_METRIC_NAMES: List[str] = [
    "common_spike_space_integrity",
    "temporal_compression_efficiency",
    "modality_temporal_budget_integrity",
    "dendritic_context_gate_stability",
    "spiking_hjepa_latent_transition",
    "reverse_reasoning_trace_integrity",
    "causal_candidate_trace_integrity",
    "module_orchestration_integrity",
    "counterfactual_lane_integrity",
    "action_trace_observability",
    "runtime_trace_replay_consistency",
]


STAGE_E_REQUIRED_MINIMUM_CHECKS: Dict[str, str] = {
    "metric.common_spike_space_integrity": "common spike space integrity",
    "metric.temporal_compression_efficiency": "temporal compression efficiency",
    "metric.modality_temporal_budget_integrity": "modality temporal budget integrity",
    "metric.dendritic_context_gate_stability": "dendritic context gate stability",
    "metric.spiking_hjepa_latent_transition": "Spiking H-JEPA latent transition integrity",
    "metric.reverse_reasoning_trace_integrity": "reverse reasoning trace integrity",
    "metric.causal_candidate_trace_integrity": "causal candidate trace integrity",
    "metric.module_orchestration_integrity": "modular cognitive runtime orchestration integrity",
    "metric.counterfactual_lane_integrity": "counterfactual lane integrity",
    "metric.action_trace_observability": "action trace observability",
    "metric.runtime_trace_replay_consistency": "runtime trace replay consistency",
}


STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_METRIC_NAMES: List[str] = list(
    dict.fromkeys(
        [
            *COGNITIVE_LINEAR_SNN_FUSION_METRIC_NAMES,
            *COGNITIVE_PLASTIC_SUBMODEL_METRIC_NAMES,
            *COGNITIVE_STAGE_E_ARCHITECTURE_INTEGRATION_METRIC_NAMES,
            *COGNITIVE_MANIFOLD_TRACE_METRIC_NAMES,
            *COGNITIVE_DELTA_MEMORY_METRIC_NAMES,
        ]
    )
)


STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_CHECKS: Dict[str, str] = {
    f"metric.{metric_name}": metric_name.replace("_", " ")
    for metric_name in STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_METRIC_NAMES
}


def stage_e_metric_check_name(metric_name: str) -> str:
    return f"metric.{metric_name}"
