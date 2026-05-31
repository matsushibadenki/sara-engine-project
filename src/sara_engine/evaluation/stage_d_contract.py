# Directory Path: src/sara_engine/evaluation/stage_d_contract.py
# English Title: Stage D Continual Consolidation Readiness Contract
# Purpose/Content: Shared constants for Stage D minimum checks used by phase3 suite, release gate, and soak summaries.

from typing import Dict, List


STAGE_D_MINIMUM_METRIC_NAMES: List[str] = [
    "replay_recovery_integrity",
    "long_horizon_consolidation_retention",
    "counterfactual_replay_selection_integrity",
    "replay_upgrade_reindex_integrity",
    "memory_health_index_integrity",
    "replay_noise_resilience_integrity",
    "astro_modulation_stability",
]


STAGE_D_DELTA_MEMORY_PROMOTION_METRIC_NAMES: List[str] = [
    "delta_memory_phase_retention_policy_observed",
    "delta_memory_crystal_retention_observed",
    "delta_memory_liquid_forget_observed",
    "delta_memory_astro_gate_alignment_observed",
    "delta_memory_policy_state_budget_observed",
    "delta_memory_multi_history_recall_observed",
    "delta_memory_multi_history_noise_resilience_observed",
    "delta_memory_multi_history_health_observed",
    "delta_memory_multi_history_manifold_guard_observed",
    "delta_memory_erase_write_decoupling_observed",
    "delta_memory_erase_preserves_stable_memory_observed",
    "delta_memory_write_commits_residual_observed",
]


STAGE_D_DELTA_MEMORY_PROMOTION_CHECKS: Dict[str, str] = {
    "metric.delta_memory_phase_retention_policy_observed": "delta-memory phase-aware retention policy",
    "metric.delta_memory_crystal_retention_observed": "delta-memory crystal retention",
    "metric.delta_memory_liquid_forget_observed": "delta-memory liquid-context forgetting",
    "metric.delta_memory_astro_gate_alignment_observed": "delta-memory astro gate alignment",
    "metric.delta_memory_policy_state_budget_observed": "delta-memory policy state budget",
    "metric.delta_memory_multi_history_recall_observed": "delta-memory multi-history recall",
    "metric.delta_memory_multi_history_noise_resilience_observed": "delta-memory multi-history noise resilience",
    "metric.delta_memory_multi_history_health_observed": "delta-memory multi-history health",
    "metric.delta_memory_multi_history_manifold_guard_observed": "delta-memory multi-history manifold leak guard",
    "metric.delta_memory_erase_write_decoupling_observed": "delta-memory separate erase and write gates",
    "metric.delta_memory_erase_preserves_stable_memory_observed": "delta-memory erase gate preserves stable memory",
    "metric.delta_memory_write_commits_residual_observed": "delta-memory write gate commits residual event",
}


STAGE_D_ACCEPTANCE_CANDIDATE_METRIC_NAMES: List[str] = [
    "synaptic_tag_integrity_observed",
    "memory_phase_transition_integrity_observed",
    "metabolic_budget_integrity_observed",
    "sleep_consolidation_retention_observed",
    *STAGE_D_DELTA_MEMORY_PROMOTION_METRIC_NAMES,
]


STAGE_D_ACCEPTANCE_CANDIDATE_CHECKS: Dict[str, str] = {
    "metric.synaptic_tag_integrity_observed": "synaptic tag stability and pruning priority",
    "metric.memory_phase_transition_integrity_observed": "liquid-glass-crystal memory phase transition",
    "metric.metabolic_budget_integrity_observed": "bounded metabolic and plasticity budget",
    "metric.sleep_consolidation_retention_observed": "sleep consolidation retention stability",
    **STAGE_D_DELTA_MEMORY_PROMOTION_CHECKS,
}


STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_NEXT_STEP_HINT = (
    "review_stage_d_acceptance_candidates_for_minimum_promotion"
)


STAGE_D_ACCEPTANCE_CANDIDATE_STABILITY_ACTIONS: List[str] = [
    "review stage_d_contract acceptance candidates and choose minimum promotion scope",
    "run python scripts/eval/phase3_accuracy_suite.py with persisted history and verify Stage D stability remains green",
    "run python scripts/eval/release_soak.py and verify operational acceptance-candidate summary remains green",
]


STAGE_D_REQUIRED_MINIMUM_CHECKS: Dict[str, str] = {
    "metric.replay_recovery_integrity": "replay recovery integrity",
    "metric.long_horizon_consolidation_retention": "long-horizon consolidation retention",
    "metric.counterfactual_replay_selection_integrity": "counterfactual replay selection integrity",
    "metric.replay_upgrade_reindex_integrity": "replay upgrade reindex integrity",
    "metric.memory_health_index_integrity": "memory health index integrity",
    "metric.replay_noise_resilience_integrity": "replay noise resilience integrity",
    "metric.astro_modulation_stability": "astro modulation stability",
}


def stage_d_metric_check_name(metric_name: str) -> str:
    return f"metric.{metric_name}"
