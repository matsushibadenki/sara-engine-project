# Directory Path: src/sara_engine/evaluation/stage_b_contract.py
# English Title: Stage B World-Model Readiness Contract
# Purpose/Content: Shared constants for Stage B minimum checks used by phase3 suite, release gate, and soak summaries.

from typing import Dict, List


STAGE_B_MINIMUM_METRIC_NAMES: List[str] = [
    "future_state_transition_integrity",
    "future_state_command_integrity",
    "future_state_predictor_snapshot_integrity",
    "future_state_runtime_tracking_integrity",
    "future_state_shift_tracking_integrity",
    "future_state_transition_operator_coverage",
    "future_state_transition_operator_consistency",
    "future_state_counterfactual_branch_viability",
    "future_state_fluid_trace_integrity",
    "future_state_fluid_support_integrity",
    "future_state_refinement_loop_integrity",
    "future_state_adaptive_refinement",
    "future_state_rewarded_action_selection_integrity",
    "future_state_policy_update_stability",
    "future_state_energy_aware_action_preference",
    "future_state_focused_retrieval_hit_ratio",
    "future_state_branch_level_decision_consistency",
]


STAGE_B_REWARD_POLICY_MINIMUM_METRIC_NAMES: List[str] = [
    "future_state_rewarded_action_selection_integrity",
    "future_state_policy_update_stability",
    "future_state_energy_aware_action_preference",
]


STAGE_B_RLM_OBSERVATION_CANDIDATE_METRIC_NAMES: List[str] = [
    "future_state_focused_retrieval_hit_ratio",
    "future_state_branch_level_decision_consistency",
]


STAGE_B_REQUIRED_MINIMUM_CHECKS: Dict[str, str] = {
    "metric.future_state_transition_integrity": "predicted future-state transitions",
    "metric.future_state_command_integrity": "predicted next-step commands",
    "metric.future_state_predictor_snapshot_integrity": "predictor snapshots",
    "metric.future_state_runtime_tracking_integrity": "future-state runtime tracking",
    "metric.future_state_shift_tracking_integrity": "future-state shift tracking",
    "metric.future_state_transition_operator_coverage": "transition-operator coverage",
    "metric.future_state_transition_operator_consistency": "transition-operator consistency",
    "metric.future_state_counterfactual_branch_viability": "counterfactual branch viability",
    "metric.future_state_fluid_trace_integrity": "fluid trace integrity",
    "metric.future_state_fluid_support_integrity": "fluid support integrity",
    "metric.future_state_refinement_loop_integrity": "refinement loop integrity",
    "metric.future_state_adaptive_refinement": "adaptive refinement integrity",
    "metric.future_state_rewarded_action_selection_integrity": "rewarded action selection integrity",
    "metric.future_state_policy_update_stability": "policy update stability",
    "metric.future_state_energy_aware_action_preference": "energy-aware action preference",
    "metric.future_state_focused_retrieval_hit_ratio": "focused retrieval hit ratio",
    "metric.future_state_branch_level_decision_consistency": "branch-level decision consistency",
}


def stage_b_metric_check_name(metric_name: str) -> str:
    return f"metric.{metric_name}"
