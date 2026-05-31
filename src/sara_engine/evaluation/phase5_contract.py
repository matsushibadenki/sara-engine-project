# Directory Path: src/sara_engine/evaluation/phase5_contract.py
# English Title: Phase 5 Predictive Coding Readiness Contract
# Purpose/Content: Shared constants for Phase 5 entry checks used by benchmark, entry gate, and release validation.

from typing import Dict, List


PHASE5_ENTRY_METRIC_NAMES: List[str] = [
    "latent_transition_alignment",
    "prediction_error_observability",
    "correction_event_coverage",
    "anti_collapse_event_diversity",
    "counterfactual_transition_separation",
    "multi_step_latent_chain_integrity",
    "long_horizon_error_correction_convergence",
    "horizon_bucket_stability",
    "macro_action_effectiveness",
    "subgoal_decomposition_integrity",
    "depth_selective_routing_integrity",
    "micro_es_policy_refinement_integrity",
]


PHASE5_REQUIRED_ENTRY_CHECKS: Dict[str, str] = {
    "metric.latent_transition_alignment": "latent transition alignment",
    "metric.prediction_error_observability": "prediction error observability",
    "metric.correction_event_coverage": "correction event coverage",
    "metric.anti_collapse_event_diversity": "anti-collapse event diversity",
    "metric.counterfactual_transition_separation": "counterfactual transition separation",
    "metric.multi_step_latent_chain_integrity": "multi-step latent chain integrity",
    "metric.long_horizon_error_correction_convergence": "long-horizon error correction convergence",
    "metric.horizon_bucket_stability": "horizon bucket stability",
    "metric.macro_action_effectiveness": "macro-action effectiveness",
    "metric.subgoal_decomposition_integrity": "subgoal decomposition integrity",
    "metric.depth_selective_routing_integrity": "depth selective routing integrity",
    "metric.micro_es_policy_refinement_integrity": "energy-aware micro evolution strategy policy refinement",
}


def phase5_metric_check_name(metric_name: str) -> str:
    return f"metric.{metric_name}"
