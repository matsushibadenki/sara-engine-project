# Directory Path: scripts/eval/phase3_accuracy_suite.py
# English Title: Phase 3 Accuracy Suite
# Purpose/Content: Aggregates lightweight Phase 3 benchmarks for SaraAgent, SaraInference, and SpikingLLM into a managed report under workspace/.

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, Optional


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_PATH = os.path.dirname(__file__)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SCRIPT_PATH not in sys.path:
    sys.path.insert(0, SCRIPT_PATH)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from agent_dialogue_benchmark import run_agent_dialogue_benchmark
from cognitive_runtime_benchmark import run_cognitive_runtime_benchmark
from continual_consolidation_benchmark import run_continual_consolidation_benchmark
from energy_efficiency_benchmark import run_energy_efficiency_benchmark
from future_state_consistency_benchmark import run_future_state_consistency_benchmark
from inference_accuracy_benchmark import run_inference_accuracy_benchmark
from nested_memory_readiness_benchmark import run_nested_memory_readiness_benchmark
from parameter_efficiency_benchmark import run_parameter_efficiency_benchmark
from phase5_predictive_coding_benchmark import run_phase5_predictive_coding_benchmark
from spiking_llm_accuracy_benchmark import run_spiking_llm_accuracy_benchmark
from task_switch_adaptation_benchmark import run_task_switch_adaptation_benchmark
from sara_engine.evaluation.phase3_tracking import (
    COGNITIVE_DELTA_MEMORY_METRIC_NAMES,
    COGNITIVE_LINEAR_SNN_FUSION_METRIC_NAMES,
    COGNITIVE_MANIFOLD_TRACE_METRIC_NAMES,
    COGNITIVE_PLASTIC_SUBMODEL_METRIC_NAMES,
    DEFAULT_PHASE3_TREND_TOLERANCE,
    append_phase3_history,
    build_cognitive_linear_snn_fusion_observed_trend,
    build_cognitive_stage_e_architecture_integration_observed_trend,
    build_phase3_trend,
    compact_neuromorphic_profile_trend,
    extract_cognitive_delta_memory_metrics,
    extract_cognitive_linear_snn_fusion_metrics,
    extract_cognitive_manifold_trace_metrics,
    extract_cognitive_plastic_submodel_metrics,
    load_phase3_history,
    latest_phase3_report,
)
from sara_engine.evaluation.stage_b_contract import (
    STAGE_B_MINIMUM_METRIC_NAMES,
    STAGE_B_RLM_OBSERVATION_CANDIDATE_METRIC_NAMES,
    STAGE_B_REWARD_POLICY_MINIMUM_METRIC_NAMES,
    STAGE_B_REQUIRED_MINIMUM_CHECKS,
    stage_b_metric_check_name,
)
from sara_engine.evaluation.stage_c_contract import (
    STAGE_C_MINIMUM_METRIC_NAMES,
    STAGE_C_REQUIRED_MINIMUM_CHECKS,
    stage_c_metric_check_name,
)
from sara_engine.evaluation.stage_d_contract import (
    STAGE_D_ACCEPTANCE_CANDIDATE_CHECKS,
    STAGE_D_ACCEPTANCE_CANDIDATE_METRIC_NAMES,
    STAGE_D_DELTA_MEMORY_PROMOTION_CHECKS,
    STAGE_D_DELTA_MEMORY_PROMOTION_METRIC_NAMES,
    STAGE_D_MINIMUM_METRIC_NAMES,
    STAGE_D_REQUIRED_MINIMUM_CHECKS,
    stage_d_metric_check_name,
)
from sara_engine.evaluation.stage_e_contract import (
    STAGE_E_MINIMUM_METRIC_NAMES,
    STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_CHECKS,
    STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_METRIC_NAMES,
    STAGE_E_REQUIRED_MINIMUM_CHECKS,
    stage_e_metric_check_name,
)
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


def _status_label(passed: bool) -> str:
    return "PASS" if passed else "WARN"


def _extract_metric_trend(
    trend: Dict[str, Any],
    metric_name: str,
) -> Dict[str, Any]:
    for bucket, status in [("improvements", "UP"), ("regressions", "DOWN")]:
        entries = trend.get(bucket, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and entry.get("metric") == metric_name:
                return {
                    "status": status,
                    "delta": float(entry.get("delta", 0.0) or 0.0),
                }

    unchanged = trend.get("unchanged", [])
    if isinstance(unchanged, list) and metric_name in unchanged:
        return {"status": "FLAT", "delta": 0.0}

    new_metrics = trend.get("new_metrics", [])
    if isinstance(new_metrics, list) and metric_name in new_metrics:
        return {"status": "NEW", "delta": None}

    return {"status": "NEW", "delta": None}


def _build_focus_trend(
    focus_summary: Dict[str, Any],
    previous_report: Optional[Dict[str, Any]],
    tolerance: float = DEFAULT_PHASE3_TREND_TOLERANCE,
) -> Dict[str, Any]:
    previous_focus = previous_report.get("focus_summary", {}) if isinstance(previous_report, dict) else {}
    if not isinstance(previous_focus, dict):
        previous_focus = {}

    focus_trend: Dict[str, Any] = {}
    for focus_name, focus_report in focus_summary.items():
        if not isinstance(focus_report, dict):
            continue
        current_score = float(focus_report.get("score", 0.0))
        previous_score = None
        previous_focus_report = previous_focus.get(focus_name, {})
        if isinstance(previous_focus_report, dict):
            try:
                previous_score = float(previous_focus_report.get("score", 0.0))
            except (TypeError, ValueError):
                previous_score = None

        if previous_score is None:
            delta = None
            status = "NEW"
        else:
            delta = current_score - previous_score
            if delta > tolerance:
                status = "UP"
            elif delta < -tolerance:
                status = "DOWN"
            else:
                status = "FLAT"

        focus_trend[focus_name] = {
            "current_score": current_score,
            "previous_score": previous_score,
            "delta": delta,
            "status": status,
        }
    return focus_trend


def _build_stage_b_promotion_readiness(
    current_stage_b: Dict[str, Any],
    history: Optional[list[Dict[str, Any]]] = None,
    required_streak: int = 3,
) -> Dict[str, Any]:
    if bool(current_stage_b.get("promotion_candidate_promoted", False)):
        return {
            "required_streak": int(required_streak),
            "consecutive_passes": 0,
            "recommended": False,
            "promoted_to_minimum": True,
        }

    streak = 0
    if bool(current_stage_b.get("promotion_candidate_ready", False)):
        streak = 1
    else:
        return {
            "required_streak": int(required_streak),
            "consecutive_passes": 0,
            "recommended": False,
            "promoted_to_minimum": False,
        }

    if isinstance(history, list):
        for item in reversed(history):
            if not isinstance(item, dict):
                break
            stage_b = item.get("stage_b_readiness", {})
            if not isinstance(stage_b, dict):
                break
            if not bool(stage_b.get("promotion_candidate_ready", False)):
                break
            streak += 1

    return {
        "required_streak": int(required_streak),
        "consecutive_passes": int(streak),
        "recommended": bool(streak >= required_streak),
        "promoted_to_minimum": False,
    }


def _build_stage_b_rlm_observation_promotion_readiness(
    current_stage_b: Dict[str, Any],
    history: Optional[list[Dict[str, Any]]] = None,
    required_streak: int = 3,
) -> Dict[str, Any]:
    if bool(current_stage_b.get("rlm_observation_candidate_promoted", False)):
        return {
            "required_streak": int(required_streak),
            "consecutive_passes": 0,
            "recommended": False,
            "promoted_to_minimum": True,
        }

    streak = 0
    if bool(current_stage_b.get("rlm_observation_candidate_ready", False)):
        streak = 1
    else:
        return {
            "required_streak": int(required_streak),
            "consecutive_passes": 0,
            "recommended": False,
            "promoted_to_minimum": False,
        }

    if isinstance(history, list):
        for item in reversed(history):
            if not isinstance(item, dict):
                break
            stage_b = item.get("stage_b_readiness", {})
            if not isinstance(stage_b, dict):
                break
            if not bool(stage_b.get("rlm_observation_candidate_ready", False)):
                break
            streak += 1

    return {
        "required_streak": int(required_streak),
        "consecutive_passes": int(streak),
        "recommended": bool(streak >= required_streak),
        "promoted_to_minimum": False,
    }


def _build_stage_d_delta_memory_promotion_readiness(
    current_stage_d: Dict[str, Any],
    history: Optional[list[Dict[str, Any]]] = None,
    required_streak: int = 3,
) -> Dict[str, Any]:
    if bool(current_stage_d.get("delta_memory_candidate_promoted", False)):
        return {
            "required_streak": int(required_streak),
            "consecutive_passes": 0,
            "recommended": False,
            "promoted_to_minimum": True,
        }

    streak = 0
    if bool(current_stage_d.get("delta_memory_candidate_ready", False)):
        streak = 1
    else:
        return {
            "required_streak": int(required_streak),
            "consecutive_passes": 0,
            "recommended": False,
            "promoted_to_minimum": False,
        }

    if isinstance(history, list):
        for item in reversed(history):
            if not isinstance(item, dict):
                break
            stage_d = item.get("stage_d_readiness", {})
            if not isinstance(stage_d, dict):
                break
            if not bool(stage_d.get("delta_memory_candidate_ready", False)):
                break
            streak += 1

    return {
        "required_streak": int(required_streak),
        "consecutive_passes": int(streak),
        "recommended": bool(streak >= required_streak),
        "promoted_to_minimum": False,
    }


def _build_stage_d_acceptance_candidate_stability(
    current_stage_d: Dict[str, Any],
    history: Optional[list[Dict[str, Any]]] = None,
    required_streak: int = 3,
) -> Dict[str, Any]:
    streak = 0
    if bool(current_stage_d.get("acceptance_candidates_ready", False)):
        streak = 1
    else:
        return {
            "required_streak": int(required_streak),
            "consecutive_passes": 0,
            "recommended": False,
        }

    if isinstance(history, list):
        for item in reversed(history):
            if not isinstance(item, dict):
                break
            stage_d = item.get("stage_d_readiness", {})
            if not isinstance(stage_d, dict):
                break
            if not bool(stage_d.get("acceptance_candidates_ready", False)):
                break
            streak += 1

    return {
        "required_streak": int(required_streak),
        "consecutive_passes": int(streak),
        "recommended": bool(streak >= required_streak),
    }


def _build_stage_e_acceptance_candidate_stability(
    current_stage_e: Dict[str, Any],
    history: Optional[list[Dict[str, Any]]] = None,
    required_streak: int = 3,
) -> Dict[str, Any]:
    streak = 0
    if bool(current_stage_e.get("observed_acceptance_candidates_ready", False)):
        streak = 1
    else:
        return {
            "required_streak": int(required_streak),
            "consecutive_passes": 0,
            "recommended": False,
        }

    if isinstance(history, list):
        for item in reversed(history):
            if not isinstance(item, dict):
                break
            stage_e = item.get("stage_e_readiness", {})
            if not isinstance(stage_e, dict):
                break
            if not bool(stage_e.get("observed_acceptance_candidates_ready", False)):
                break
            streak += 1

    return {
        "required_streak": int(required_streak),
        "consecutive_passes": int(streak),
        "recommended": bool(streak >= required_streak),
    }


def _build_focus_summary(component_reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    agent_metrics = component_reports.get("agent_dialogue", {}).get("metrics", {})
    inference_metrics = component_reports.get("sara_inference", {}).get("metrics", {})
    llm_metrics = component_reports.get("spiking_llm", {}).get("metrics", {})
    adaptation_metrics = component_reports.get("task_switch_adaptation", {}).get("metrics", {})
    future_state_metrics = component_reports.get("future_state_consistency", {}).get("metrics", {})
    efficiency_metrics = component_reports.get("energy_efficiency", {}).get("metrics", {})
    parameter_efficiency_metrics = component_reports.get("parameter_efficiency", {}).get("metrics", {})
    consolidation_metrics = component_reports.get("continual_consolidation", {}).get("metrics", {})
    nested_memory_metrics = component_reports.get("nested_memory", {}).get("metrics", {})
    cognitive_runtime_metrics = component_reports.get("cognitive_runtime", {}).get("metrics", {})
    phase5_metrics = component_reports.get("phase5_predictive_coding", {}).get("metrics", {})

    few_shot_metrics = {
        "sara_inference.few_shot_accuracy": float(inference_metrics.get("few_shot_accuracy", 0.0)),
        "spiking_llm.few_shot_context_accuracy": float(llm_metrics.get("few_shot_context_accuracy", 0.0)),
        "spiking_llm.hierarchical_context_integrity": float(
            llm_metrics.get("hierarchical_context_integrity", 0.0)
        ),
    }
    continual_metrics = {
        "sara_inference.continual_retention": float(inference_metrics.get("continual_retention", 0.0)),
        "sara_inference.long_horizon_retention": float(inference_metrics.get("long_horizon_retention", 0.0)),
        "spiking_llm.continual_memory_retention": float(llm_metrics.get("continual_memory_retention", 0.0)),
        "spiking_llm.long_horizon_memory_retention": float(llm_metrics.get("long_horizon_memory_retention", 0.0)),
    }
    retrieval_hygiene_metrics = {
        "agent_dialogue.retrieval_stability": float(agent_metrics.get("retrieval_stability", 0.0)),
        "agent_dialogue.off_topic_suppression": float(agent_metrics.get("off_topic_suppression", 0.0)),
        "agent_dialogue.retrieval_grounding": float(agent_metrics.get("retrieval_grounding", 0.0)),
    }
    adaptive_readiness_metrics = {
        "task_switch_adaptation.task_switch_adaptation": float(
            adaptation_metrics.get("task_switch_adaptation", 0.0)
        ),
        "task_switch_adaptation.session_memory_switch_grounding": float(
            adaptation_metrics.get("session_memory_switch_grounding", 0.0)
        ),
        "task_switch_adaptation.meta_adaptation_loop": float(
            adaptation_metrics.get("meta_adaptation_loop", 0.0)
        ),
        "task_switch_adaptation.meta_adaptation_parameter_integrity": float(
            adaptation_metrics.get(
                "meta_adaptation_parameter_integrity",
                adaptation_metrics.get("meta_adaptation_loop", 0.0),
            )
        ),
        "task_switch_adaptation.temporal_self_distillation_stability": float(
            adaptation_metrics.get(
                "temporal_self_distillation_stability",
                adaptation_metrics.get("meta_adaptation_loop", 0.0),
            )
        ),
        "agent_dialogue.direction_shift_following": float(
            agent_metrics.get("direction_shift_following", 0.0)
        ),
    }
    predictive_readiness_metrics = {
        "future_state_consistency.future_state_consistency": float(
            future_state_metrics.get("future_state_consistency", 0.0)
        ),
        "future_state_consistency.future_state_memory_grounding": float(
            future_state_metrics.get("future_state_memory_grounding", 0.0)
        ),
        "future_state_consistency.future_state_transition_integrity": float(
            future_state_metrics.get("future_state_transition_integrity", 0.0)
        ),
        "future_state_consistency.future_state_command_integrity": float(
            future_state_metrics.get("future_state_command_integrity", 0.0)
        ),
        "future_state_consistency.future_state_counterfactual_integrity": float(
            future_state_metrics.get("future_state_counterfactual_integrity", 0.0)
        ),
        "future_state_consistency.future_state_counterfactual_usefulness": float(
            future_state_metrics.get("future_state_counterfactual_usefulness", 0.0)
        ),
        "future_state_consistency.future_state_branching_integrity": float(
            future_state_metrics.get("future_state_branching_integrity", 0.0)
        ),
        "future_state_consistency.future_state_options_integrity": float(
            future_state_metrics.get("future_state_options_integrity", 0.0)
        ),
        "future_state_consistency.future_state_ranking_integrity": float(
            future_state_metrics.get("future_state_ranking_integrity", 0.0)
        ),
        "future_state_consistency.future_state_decision_brief_integrity": float(
            future_state_metrics.get("future_state_decision_brief_integrity", 0.0)
        ),
        "future_state_consistency.future_state_choice_integrity": float(
            future_state_metrics.get("future_state_choice_integrity", 0.0)
        ),
        "future_state_consistency.future_state_choice_reason_integrity": float(
            future_state_metrics.get("future_state_choice_reason_integrity", 0.0)
        ),
        "future_state_consistency.future_state_simulation_integrity": float(
            future_state_metrics.get("future_state_simulation_integrity", 0.0)
        ),
        "future_state_consistency.future_state_simulation_usefulness": float(
            future_state_metrics.get("future_state_simulation_usefulness", 0.0)
        ),
        "future_state_consistency.future_state_transition_operator_coverage": float(
            future_state_metrics.get("future_state_transition_operator_coverage", 0.0)
        ),
        "future_state_consistency.future_state_transition_operator_consistency": float(
            future_state_metrics.get("future_state_transition_operator_consistency", 0.0)
        ),
        "future_state_consistency.future_state_counterfactual_branch_viability": float(
            future_state_metrics.get("future_state_counterfactual_branch_viability", 0.0)
        ),
        "future_state_consistency.future_state_speculative_acceptance_ratio": float(
            future_state_metrics.get("future_state_speculative_acceptance_ratio", 0.0)
        ),
        "future_state_consistency.future_state_speculative_rollback_observability": float(
            future_state_metrics.get("future_state_speculative_rollback_observability", 0.0)
        ),
        "future_state_consistency.future_state_fluid_trace_integrity": float(
            future_state_metrics.get("future_state_fluid_trace_integrity", 0.0)
        ),
        "future_state_consistency.future_state_fluid_support_integrity": float(
            future_state_metrics.get("future_state_fluid_support_integrity", 0.0)
        ),
        "future_state_consistency.future_state_refinement_loop_integrity": float(
            future_state_metrics.get("future_state_refinement_loop_integrity", 0.0)
        ),
        "future_state_consistency.future_state_adaptive_refinement": float(
            future_state_metrics.get("future_state_adaptive_refinement", 0.0)
        ),
        "future_state_consistency.future_state_rewarded_action_selection_integrity": float(
            future_state_metrics.get("future_state_rewarded_action_selection_integrity", 0.0)
        ),
        "future_state_consistency.future_state_policy_update_stability": float(
            future_state_metrics.get("future_state_policy_update_stability", 0.0)
        ),
        "future_state_consistency.future_state_energy_aware_action_preference": float(
            future_state_metrics.get("future_state_energy_aware_action_preference", 0.0)
        ),
        "future_state_consistency.future_state_focused_retrieval_hit_ratio": float(
            future_state_metrics.get("future_state_focused_retrieval_hit_ratio", 0.0)
        ),
        "future_state_consistency.future_state_branch_level_decision_consistency": float(
            future_state_metrics.get("future_state_branch_level_decision_consistency", 0.0)
        ),
        "future_state_consistency.future_state_spatial_projection_integrity": float(
            future_state_metrics.get("future_state_spatial_projection_integrity", 0.0)
        ),
        "future_state_consistency.future_state_spatial_topology_consistency": float(
            future_state_metrics.get("future_state_spatial_topology_consistency", 0.0)
        ),
        "future_state_consistency.future_state_spatial_occlusion_reasoning": float(
            future_state_metrics.get("future_state_spatial_occlusion_reasoning", 0.0)
        ),
        "future_state_consistency.future_state_spatial_counterfactual_selection": float(
            future_state_metrics.get("future_state_spatial_counterfactual_selection", 0.0)
        ),
        "future_state_consistency.future_state_spatial_adjacency_consistency": float(
            future_state_metrics.get("future_state_spatial_adjacency_consistency", 0.0)
        ),
        "future_state_consistency.future_state_spatial_door_connectivity_integrity": float(
            future_state_metrics.get("future_state_spatial_door_connectivity_integrity", 0.0)
        ),
        "future_state_consistency.future_state_spatial_multi_room_counterfactual_selection": float(
            future_state_metrics.get("future_state_spatial_multi_room_counterfactual_selection", 0.0)
        ),
        "future_state_consistency.future_state_spatial_route_planning_integrity": float(
            future_state_metrics.get("future_state_spatial_route_planning_integrity", 0.0)
        ),
        "future_state_consistency.future_state_spatial_affordance_action_selection": float(
            future_state_metrics.get("future_state_spatial_affordance_action_selection", 0.0)
        ),
        "future_state_consistency.future_state_spatial_energy_aware_route_selection": float(
            future_state_metrics.get("future_state_spatial_energy_aware_route_selection", 0.0)
        ),
        "future_state_consistency.future_state_spatial_route_state_update_integrity": float(
            future_state_metrics.get("future_state_spatial_route_state_update_integrity", 0.0)
        ),
        "future_state_consistency.future_state_spatial_invalid_action_rejection": float(
            future_state_metrics.get("future_state_spatial_invalid_action_rejection", 0.0)
        ),
        "future_state_consistency.future_state_spatial_route_rollback_observability": float(
            future_state_metrics.get("future_state_spatial_route_rollback_observability", 0.0)
        ),
        "future_state_consistency.future_state_spatial_route_execution_cost_bound": float(
            future_state_metrics.get("future_state_spatial_route_execution_cost_bound", 0.0)
        ),
    }
    efficiency_readiness_metrics = {
        "energy_efficiency.energy_per_success_proxy": float(
            efficiency_metrics.get("energy_per_success_proxy", 0.0)
        ),
        "energy_efficiency.performance_energy_ratio_proxy": float(
            efficiency_metrics.get("performance_energy_ratio_proxy", 0.0)
        ),
        "energy_efficiency.ann_cost_advantage_proxy": float(
            efficiency_metrics.get("ann_cost_advantage_proxy", 0.0)
        ),
        "energy_efficiency.sparse_event_cost_score": float(
            efficiency_metrics.get("sparse_event_cost_score", 0.0)
        ),
        "energy_efficiency.brain_efficiency_alignment_proxy": float(
            efficiency_metrics.get("brain_efficiency_alignment_proxy", 0.0)
        ),
        "energy_efficiency.memory_per_success_proxy": float(
            efficiency_metrics.get("memory_per_success_proxy", 0.0)
        ),
        "energy_efficiency.low_overhead_route_score": float(
            efficiency_metrics.get("low_overhead_route_score", 0.0)
        ),
        "energy_efficiency.bounded_latency_score": float(
            efficiency_metrics.get("bounded_latency_score", 0.0)
        ),
        "energy_efficiency.stochastic_readout_integrity": float(
            efficiency_metrics.get("stochastic_readout_integrity", 0.0)
        ),
    }
    parameter_efficiency_focus_metrics = {
        "parameter_efficiency.quality_per_kparam_score": float(
            parameter_efficiency_metrics.get("quality_per_kparam_score", 0.0)
        ),
        "parameter_efficiency.quality_per_mb_score": float(
            parameter_efficiency_metrics.get("quality_per_mb_score", 0.0)
        ),
        "parameter_efficiency.bounded_parameter_footprint_score": float(
            parameter_efficiency_metrics.get("bounded_parameter_footprint_score", 0.0)
        ),
        "parameter_efficiency.bounded_artifact_footprint_score": float(
            parameter_efficiency_metrics.get("bounded_artifact_footprint_score", 0.0)
        ),
    }
    consolidation_readiness_metrics = {
        "continual_consolidation.replay_recovery_integrity": float(
            consolidation_metrics.get("replay_recovery_integrity", 0.0)
        ),
        "continual_consolidation.long_horizon_consolidation_retention": float(
            consolidation_metrics.get("long_horizon_consolidation_retention", 0.0)
        ),
        "continual_consolidation.counterfactual_replay_selection_integrity": float(
            consolidation_metrics.get("counterfactual_replay_selection_integrity", 0.0)
        ),
        "continual_consolidation.replay_upgrade_reindex_integrity": float(
            consolidation_metrics.get("replay_upgrade_reindex_integrity", 0.0)
        ),
        "continual_consolidation.memory_health_index_integrity": float(
            consolidation_metrics.get("memory_health_index_integrity", 0.0)
        ),
        "continual_consolidation.replay_noise_resilience_integrity": float(
            consolidation_metrics.get("replay_noise_resilience_integrity", 0.0)
        ),
        "continual_consolidation.astro_modulation_stability": float(
            consolidation_metrics.get("astro_modulation_stability", 0.0)
        ),
    }

    few_shot_score = sum(few_shot_metrics.values()) / max(len(few_shot_metrics), 1)
    continual_score = sum(continual_metrics.values()) / max(len(continual_metrics), 1)
    retrieval_hygiene_score = sum(retrieval_hygiene_metrics.values()) / max(len(retrieval_hygiene_metrics), 1)
    adaptive_readiness_score = sum(adaptive_readiness_metrics.values()) / max(len(adaptive_readiness_metrics), 1)
    predictive_readiness_score = sum(predictive_readiness_metrics.values()) / max(
        len(predictive_readiness_metrics), 1
    )
    efficiency_readiness_score = sum(efficiency_readiness_metrics.values()) / max(
        len(efficiency_readiness_metrics), 1
    )
    parameter_efficiency_score = sum(parameter_efficiency_focus_metrics.values()) / max(
        len(parameter_efficiency_focus_metrics), 1
    )
    consolidation_readiness_score = sum(consolidation_readiness_metrics.values()) / max(
        len(consolidation_readiness_metrics), 1
    )
    nested_memory_readiness_metrics = {
        "nested_memory.multi_rate_update_integrity": float(
            nested_memory_metrics.get("multi_rate_update_integrity", 0.0)
        ),
        "nested_memory.continuum_memory_transfer_stability": float(
            nested_memory_metrics.get("continuum_memory_transfer_stability", 0.0)
        ),
        "nested_memory.scheduler_energy_budget_integrity": float(
            nested_memory_metrics.get("scheduler_energy_budget_integrity", 0.0)
        ),
        "nested_memory.catastrophic_interference_guard": float(
            nested_memory_metrics.get("catastrophic_interference_guard", 0.0)
        ),
    }
    nested_memory_readiness_score = sum(nested_memory_readiness_metrics.values()) / max(
        len(nested_memory_readiness_metrics), 1
    )
    cognitive_runtime_readiness_metrics = {
        "cognitive_runtime.common_spike_space_integrity": float(
            cognitive_runtime_metrics.get("common_spike_space_integrity", 0.0)
        ),
        "cognitive_runtime.temporal_compression_efficiency": float(
            cognitive_runtime_metrics.get("temporal_compression_efficiency", 0.0)
        ),
        "cognitive_runtime.modality_temporal_budget_integrity": float(
            cognitive_runtime_metrics.get("modality_temporal_budget_integrity", 0.0)
        ),
        "cognitive_runtime.dendritic_context_gate_stability": float(
            cognitive_runtime_metrics.get("dendritic_context_gate_stability", 0.0)
        ),
        "cognitive_runtime.spiking_hjepa_latent_transition": float(
            cognitive_runtime_metrics.get("spiking_hjepa_latent_transition", 0.0)
        ),
        "cognitive_runtime.reverse_reasoning_trace_integrity": float(
            cognitive_runtime_metrics.get("reverse_reasoning_trace_integrity", 0.0)
        ),
        "cognitive_runtime.causal_candidate_trace_integrity": float(
            cognitive_runtime_metrics.get("causal_candidate_trace_integrity", 0.0)
        ),
        "cognitive_runtime.module_orchestration_integrity": float(
            cognitive_runtime_metrics.get("module_orchestration_integrity", 0.0)
        ),
        "cognitive_runtime.counterfactual_lane_integrity": float(
            cognitive_runtime_metrics.get("counterfactual_lane_integrity", 0.0)
        ),
        "cognitive_runtime.action_trace_observability": float(
            cognitive_runtime_metrics.get("action_trace_observability", 0.0)
        ),
        "cognitive_runtime.runtime_trace_replay_consistency": float(
            cognitive_runtime_metrics.get("runtime_trace_replay_consistency", 0.0)
        ),
    }
    cognitive_runtime_readiness_score = sum(cognitive_runtime_readiness_metrics.values()) / max(
        len(cognitive_runtime_readiness_metrics), 1
    )
    cognitive_linear_snn_fusion_metrics = {
        f"cognitive_runtime.{metric_name}": float(cognitive_runtime_metrics.get(metric_name, 0.0))
        for metric_name in COGNITIVE_LINEAR_SNN_FUSION_METRIC_NAMES
    }
    cognitive_plastic_submodel_metrics = {
        f"cognitive_runtime.{metric_name}": float(cognitive_runtime_metrics.get(metric_name, 0.0))
        for metric_name in COGNITIVE_PLASTIC_SUBMODEL_METRIC_NAMES
    }
    phase5_entry_readiness_metrics = {
        "phase5_predictive_coding.latent_transition_alignment": float(
            phase5_metrics.get("latent_transition_alignment", 0.0)
        ),
        "phase5_predictive_coding.prediction_error_observability": float(
            phase5_metrics.get("prediction_error_observability", 0.0)
        ),
        "phase5_predictive_coding.correction_event_coverage": float(
            phase5_metrics.get("correction_event_coverage", 0.0)
        ),
        "phase5_predictive_coding.anti_collapse_event_diversity": float(
            phase5_metrics.get("anti_collapse_event_diversity", 0.0)
        ),
        "phase5_predictive_coding.counterfactual_transition_separation": float(
            phase5_metrics.get("counterfactual_transition_separation", 0.0)
        ),
        "phase5_predictive_coding.multi_step_latent_chain_integrity": float(
            phase5_metrics.get("multi_step_latent_chain_integrity", 0.0)
        ),
        "phase5_predictive_coding.long_horizon_error_correction_convergence": float(
            phase5_metrics.get("long_horizon_error_correction_convergence", 0.0)
        ),
        "phase5_predictive_coding.horizon_bucket_stability": float(
            phase5_metrics.get("horizon_bucket_stability", 0.0)
        ),
        "phase5_predictive_coding.macro_action_effectiveness": float(
            phase5_metrics.get("macro_action_effectiveness", 0.0)
        ),
        "phase5_predictive_coding.subgoal_decomposition_integrity": float(
            phase5_metrics.get("subgoal_decomposition_integrity", 0.0)
        ),
        "phase5_predictive_coding.depth_selective_routing_integrity": float(
            phase5_metrics.get("depth_selective_routing_integrity", 0.0)
        ),
        "phase5_predictive_coding.micro_es_policy_refinement_integrity": float(
            phase5_metrics.get("micro_es_policy_refinement_integrity", 0.0)
        ),
    }
    phase5_entry_readiness_score = sum(phase5_entry_readiness_metrics.values()) / max(
        len(phase5_entry_readiness_metrics), 1
    )
    return {
        "few_shot": {
            "score": few_shot_score,
            "passed": all(value >= 1.0 for value in few_shot_metrics.values()),
            "metrics": few_shot_metrics,
        },
        "continual": {
            "score": continual_score,
            "passed": all(value >= 1.0 for value in continual_metrics.values()),
            "metrics": continual_metrics,
        },
        "retrieval_hygiene": {
            "score": retrieval_hygiene_score,
            "passed": bool(
                retrieval_hygiene_metrics["agent_dialogue.retrieval_stability"] >= 0.40
                and retrieval_hygiene_metrics["agent_dialogue.off_topic_suppression"] >= 0.75
                and retrieval_hygiene_metrics["agent_dialogue.retrieval_grounding"] >= 0.35
            ),
            "metrics": retrieval_hygiene_metrics,
        },
        "adaptive_readiness": {
            "score": adaptive_readiness_score,
            "passed": all(value >= 1.0 for value in adaptive_readiness_metrics.values()),
            "metrics": adaptive_readiness_metrics,
        },
        "predictive_readiness": {
            "score": predictive_readiness_score,
            "passed": all(value >= 1.0 for value in predictive_readiness_metrics.values()),
            "metrics": predictive_readiness_metrics,
        },
        "efficiency_readiness": {
            "score": efficiency_readiness_score,
            "passed": bool(
                efficiency_readiness_metrics["energy_efficiency.energy_per_success_proxy"] >= 1.0
                and efficiency_readiness_metrics["energy_efficiency.performance_energy_ratio_proxy"] >= 0.20
                and efficiency_readiness_metrics["energy_efficiency.ann_cost_advantage_proxy"] >= 8.0
                and efficiency_readiness_metrics["energy_efficiency.sparse_event_cost_score"] >= 1.0
                and efficiency_readiness_metrics["energy_efficiency.brain_efficiency_alignment_proxy"] >= 0.85
                and efficiency_readiness_metrics["energy_efficiency.memory_per_success_proxy"] >= 1.0
                and efficiency_readiness_metrics["energy_efficiency.low_overhead_route_score"] >= 1.0
                and efficiency_readiness_metrics["energy_efficiency.bounded_latency_score"] >= 0.80
                and efficiency_readiness_metrics["energy_efficiency.stochastic_readout_integrity"] >= 1.0
            ),
            "metrics": efficiency_readiness_metrics,
        },
        "parameter_efficiency": {
            "score": parameter_efficiency_score,
            "passed": all(value >= 0.5 for value in parameter_efficiency_focus_metrics.values()),
            "metrics": {
                **parameter_efficiency_focus_metrics,
                "parameter_efficiency.average_quality_per_kparam": float(
                    parameter_efficiency_metrics.get("average_quality_per_kparam", 0.0)
                ),
                "parameter_efficiency.average_quality_per_mb": float(
                    parameter_efficiency_metrics.get("average_quality_per_mb", 0.0)
                ),
            },
        },
        "consolidation_readiness": {
            "score": consolidation_readiness_score,
            "passed": all(value >= 1.0 for value in consolidation_readiness_metrics.values()),
            "metrics": consolidation_readiness_metrics,
        },
        "nested_memory_readiness": {
            "score": nested_memory_readiness_score,
            "passed": all(value >= 1.0 for value in nested_memory_readiness_metrics.values()),
            "observed_only": True,
            "metrics": {
                **nested_memory_readiness_metrics,
                "nested_memory.active_band_ratio": float(nested_memory_metrics.get("active_band_ratio", 0.0)),
                "nested_memory.slow_update_ratio": float(nested_memory_metrics.get("slow_update_ratio", 0.0)),
                "nested_memory.energy_budget_utilization": float(
                    nested_memory_metrics.get("energy_budget_utilization", 0.0)
                ),
            },
        },
        "cognitive_runtime_readiness": {
            "score": cognitive_runtime_readiness_score,
            "passed": all(value >= 1.0 for value in cognitive_runtime_readiness_metrics.values()),
            "observed_only": False,
            "observed_metric_policy": {
                "linear_snn_fusion_metrics_excluded_from_score": True,
                "linear_snn_fusion_metrics_excluded_from_release_gate": True,
            },
            "metrics": cognitive_runtime_readiness_metrics,
            "observed_metrics": cognitive_linear_snn_fusion_metrics,
            "plastic_submodel_observed_metrics": cognitive_plastic_submodel_metrics,
        },
        "phase5_entry_readiness": {
            "score": phase5_entry_readiness_score,
            "passed": all(value >= 1.0 for value in phase5_entry_readiness_metrics.values()),
            "metrics": phase5_entry_readiness_metrics,
        },
    }


def _build_stage_a_acceptance(
    report: Dict[str, Any],
) -> Dict[str, Any]:
    component_reports = report.get("component_reports", {})
    if not isinstance(component_reports, dict):
        component_reports = {}
    focus_summary = report.get("focus_summary", {})
    if not isinstance(focus_summary, dict):
        focus_summary = {}
    trend = report.get("trend", {})
    if not isinstance(trend, dict):
        trend = {}

    required_components = [
        "agent_dialogue",
        "sara_inference",
        "spiking_llm",
        "task_switch_adaptation",
        "future_state_consistency",
        "energy_efficiency",
        "continual_consolidation",
        "cognitive_runtime",
        "phase5_predictive_coding",
    ]
    required_focus = [
        "few_shot",
        "continual",
        "retrieval_hygiene",
        "adaptive_readiness",
        "predictive_readiness",
        "efficiency_readiness",
        "parameter_efficiency",
        "consolidation_readiness",
        "cognitive_runtime_readiness",
        "phase5_entry_readiness",
    ]

    component_checks = {
        name: bool(
            isinstance(component_reports.get(name), dict)
            and component_reports.get(name, {}).get("passed", False)
        )
        for name in required_components
    }
    focus_checks = {
        name: bool(
            isinstance(focus_summary.get(name), dict)
            and focus_summary.get(name, {}).get("passed", False)
        )
        for name in required_focus
    }

    gate_regression_count = int(trend.get("gate_regression_count", trend.get("regression_count", 0)) or 0)
    zero_regressions = gate_regression_count == 0
    overall_score = float(report.get("overall_score", 0.0))
    accuracy_target_met = overall_score >= 0.95

    checks: Dict[str, bool] = {}
    for name, passed in component_checks.items():
        checks[f"component.{name}"] = passed
    for name, passed in focus_checks.items():
        checks[f"focus.{name}"] = passed
    checks["trend.zero_regressions"] = zero_regressions
    checks["overall.acc_target_0_95"] = accuracy_target_met

    return {
        "stage": "Stage A",
        "label": "Evaluation First",
        "passed": all(checks.values()),
        "overall_score": overall_score,
        "acc_target": 0.95,
        "checks": checks,
    }


def _build_stage_b_readiness(
    report: Dict[str, Any],
) -> Dict[str, Any]:
    component_reports = report.get("component_reports", {})
    if not isinstance(component_reports, dict):
        component_reports = {}
    future_state_component = (
        component_reports.get("future_state_consistency", {})
        if isinstance(component_reports.get("future_state_consistency"), dict)
        else {}
    )
    metrics = future_state_component.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    tracked_metrics = {
        "future_state_transition_integrity": float(metrics.get("future_state_transition_integrity", 0.0)),
        "future_state_command_integrity": float(metrics.get("future_state_command_integrity", 0.0)),
        "future_state_predictor_snapshot_integrity": float(
            metrics.get("future_state_predictor_snapshot_integrity", 0.0)
        ),
        "future_state_counterfactual_integrity": float(metrics.get("future_state_counterfactual_integrity", 0.0)),
        "future_state_branching_integrity": float(metrics.get("future_state_branching_integrity", 0.0)),
        "future_state_choice_integrity": float(metrics.get("future_state_choice_integrity", 0.0)),
        "future_state_runtime_tracking_integrity": float(metrics.get("future_state_runtime_tracking_integrity", 0.0)),
        "future_state_shift_tracking_integrity": float(metrics.get("future_state_shift_tracking_integrity", 0.0)),
        "future_state_simulation_integrity": float(metrics.get("future_state_simulation_integrity", 0.0)),
        "future_state_simulation_usefulness": float(metrics.get("future_state_simulation_usefulness", 0.0)),
        "future_state_transition_operator_coverage": float(
            metrics.get("future_state_transition_operator_coverage", 0.0)
        ),
        "future_state_transition_operator_consistency": float(
            metrics.get("future_state_transition_operator_consistency", 0.0)
        ),
        "future_state_counterfactual_branch_viability": float(
            metrics.get("future_state_counterfactual_branch_viability", 0.0)
        ),
        "future_state_speculative_acceptance_ratio": float(
            metrics.get("future_state_speculative_acceptance_ratio", 0.0)
        ),
        "future_state_speculative_rollback_observability": float(
            metrics.get("future_state_speculative_rollback_observability", 0.0)
        ),
        "future_state_fluid_trace_integrity": float(metrics.get("future_state_fluid_trace_integrity", 0.0)),
        "future_state_fluid_support_integrity": float(metrics.get("future_state_fluid_support_integrity", 0.0)),
        "future_state_refinement_loop_integrity": float(
            metrics.get("future_state_refinement_loop_integrity", 0.0)
        ),
        "future_state_adaptive_refinement": float(metrics.get("future_state_adaptive_refinement", 0.0)),
        "future_state_rewarded_action_selection_integrity": float(
            metrics.get("future_state_rewarded_action_selection_integrity", 0.0)
        ),
        "future_state_policy_update_stability": float(
            metrics.get("future_state_policy_update_stability", 0.0)
        ),
        "future_state_energy_aware_action_preference": float(
            metrics.get("future_state_energy_aware_action_preference", 0.0)
        ),
        "future_state_focused_retrieval_hit_ratio": float(
            metrics.get("future_state_focused_retrieval_hit_ratio", 0.0)
        ),
        "future_state_branch_level_decision_consistency": float(
            metrics.get("future_state_branch_level_decision_consistency", 0.0)
        ),
        "future_state_spatial_projection_integrity": float(
            metrics.get("future_state_spatial_projection_integrity", 0.0)
        ),
        "future_state_spatial_topology_consistency": float(
            metrics.get("future_state_spatial_topology_consistency", 0.0)
        ),
        "future_state_spatial_occlusion_reasoning": float(
            metrics.get("future_state_spatial_occlusion_reasoning", 0.0)
        ),
        "future_state_spatial_counterfactual_selection": float(
            metrics.get("future_state_spatial_counterfactual_selection", 0.0)
        ),
        "future_state_spatial_adjacency_consistency": float(
            metrics.get("future_state_spatial_adjacency_consistency", 0.0)
        ),
        "future_state_spatial_door_connectivity_integrity": float(
            metrics.get("future_state_spatial_door_connectivity_integrity", 0.0)
        ),
        "future_state_spatial_multi_room_counterfactual_selection": float(
            metrics.get("future_state_spatial_multi_room_counterfactual_selection", 0.0)
        ),
        "future_state_spatial_route_planning_integrity": float(
            metrics.get("future_state_spatial_route_planning_integrity", 0.0)
        ),
        "future_state_spatial_affordance_action_selection": float(
            metrics.get("future_state_spatial_affordance_action_selection", 0.0)
        ),
        "future_state_spatial_energy_aware_route_selection": float(
            metrics.get("future_state_spatial_energy_aware_route_selection", 0.0)
        ),
        "future_state_spatial_route_state_update_integrity": float(
            metrics.get("future_state_spatial_route_state_update_integrity", 0.0)
        ),
        "future_state_spatial_invalid_action_rejection": float(
            metrics.get("future_state_spatial_invalid_action_rejection", 0.0)
        ),
        "future_state_spatial_route_rollback_observability": float(
            metrics.get("future_state_spatial_route_rollback_observability", 0.0)
        ),
        "future_state_spatial_route_execution_cost_bound": float(
            metrics.get("future_state_spatial_route_execution_cost_bound", 0.0)
        ),
    }
    checks = {
        f"metric.{name}": (
            value >= 1.0
            if name
            not in {
                "future_state_speculative_acceptance_ratio",
                "future_state_speculative_rollback_observability",
            }
            else value >= 0.80
        )
        for name, value in tracked_metrics.items()
    }
    readiness_score = sum(tracked_metrics.values()) / max(len(tracked_metrics), 1)
    minimum_checks = {
        stage_b_metric_check_name(name): checks.get(stage_b_metric_check_name(name), False)
        for name in STAGE_B_MINIMUM_METRIC_NAMES
    }
    minimum_failures = []
    for metric_name in STAGE_B_MINIMUM_METRIC_NAMES:
        check_name = stage_b_metric_check_name(metric_name)
        if bool(minimum_checks.get(check_name, False)):
            continue
        minimum_failures.append(
            {
                "check": check_name,
                "metric": metric_name,
                "description": STAGE_B_REQUIRED_MINIMUM_CHECKS.get(check_name, metric_name),
                "value": float(tracked_metrics.get(metric_name, 0.0)),
                "threshold": 1.0,
            }
        )

    promotion_candidate_metric_names = list(STAGE_B_REWARD_POLICY_MINIMUM_METRIC_NAMES)
    promotion_candidate_checks = {
        stage_b_metric_check_name(name): bool(
            checks.get(stage_b_metric_check_name(name), False)
        )
        for name in promotion_candidate_metric_names
    }
    promotion_candidate_promoted = all(name in STAGE_B_MINIMUM_METRIC_NAMES for name in promotion_candidate_metric_names)
    promotion_candidate_ready = all(promotion_candidate_checks.values())
    promotion_candidate_failure_count = sum(
        1 for passed in promotion_candidate_checks.values() if not bool(passed)
    )
    rlm_observation_candidate_metric_names = list(STAGE_B_RLM_OBSERVATION_CANDIDATE_METRIC_NAMES)
    rlm_observation_candidate_checks = {
        stage_b_metric_check_name(name): bool(
            checks.get(stage_b_metric_check_name(name), False)
        )
        for name in rlm_observation_candidate_metric_names
    }
    rlm_observation_candidate_promoted = all(
        name in STAGE_B_MINIMUM_METRIC_NAMES
        for name in rlm_observation_candidate_metric_names
    )
    rlm_observation_candidate_ready = all(rlm_observation_candidate_checks.values())
    rlm_observation_candidate_failure_count = sum(
        1 for passed in rlm_observation_candidate_checks.values() if not bool(passed)
    )

    return {
        "stage": "Stage B",
        "label": "Lightweight World Model Prototypes",
        "passed": all(checks.values()),
        "minimum_requirements_passed": all(minimum_checks.values()),
        "minimum_failure_count": len(minimum_failures),
        "minimum_failures": minimum_failures,
        "promotion_candidate_ready": bool(promotion_candidate_ready),
        "promotion_candidate_failure_count": int(promotion_candidate_failure_count),
        "promotion_candidate_promoted": bool(promotion_candidate_promoted),
        "promotion_candidate_checks": promotion_candidate_checks,
        "rlm_observation_candidate_ready": bool(rlm_observation_candidate_ready),
        "rlm_observation_candidate_failure_count": int(rlm_observation_candidate_failure_count),
        "rlm_observation_candidate_promoted": bool(rlm_observation_candidate_promoted),
        "rlm_observation_candidate_checks": rlm_observation_candidate_checks,
        "readiness_score": readiness_score,
        "checks": checks,
        "minimum_checks": minimum_checks,
        "metrics": tracked_metrics,
    }


def _build_stage_c_readiness(
    report: Dict[str, Any],
) -> Dict[str, Any]:
    component_reports = report.get("component_reports", {})
    if not isinstance(component_reports, dict):
        component_reports = {}
    adaptation_component = (
        component_reports.get("task_switch_adaptation", {})
        if isinstance(component_reports.get("task_switch_adaptation"), dict)
        else {}
    )
    metrics = adaptation_component.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    tracked_metrics = {
        "meta_adaptation_loop": float(metrics.get("meta_adaptation_loop", 0.0)),
        "meta_adaptation_parameter_integrity": float(
            metrics.get("meta_adaptation_parameter_integrity", 0.0)
        ),
        "temporal_self_distillation_stability": float(
            metrics.get("temporal_self_distillation_stability", 0.0)
        ),
    }
    checks = {
        f"metric.{name}": value >= 1.0
        for name, value in tracked_metrics.items()
    }
    readiness_score = sum(tracked_metrics.values()) / max(len(tracked_metrics), 1)
    minimum_checks = {
        stage_c_metric_check_name(name): checks.get(stage_c_metric_check_name(name), False)
        for name in STAGE_C_MINIMUM_METRIC_NAMES
    }
    minimum_failures = []
    for metric_name in STAGE_C_MINIMUM_METRIC_NAMES:
        check_name = stage_c_metric_check_name(metric_name)
        if bool(minimum_checks.get(check_name, False)):
            continue
        minimum_failures.append(
            {
                "check": check_name,
                "metric": metric_name,
                "description": STAGE_C_REQUIRED_MINIMUM_CHECKS.get(check_name, metric_name),
                "value": float(tracked_metrics.get(metric_name, 0.0)),
                "threshold": 1.0,
            }
        )

    return {
        "stage": "Stage C",
        "label": "Meta-Adaptation Experiments",
        "passed": all(checks.values()),
        "minimum_requirements_passed": all(minimum_checks.values()),
        "minimum_failure_count": len(minimum_failures),
        "minimum_failures": minimum_failures,
        "readiness_score": readiness_score,
        "checks": checks,
        "minimum_checks": minimum_checks,
        "metrics": tracked_metrics,
    }


def _build_stage_d_readiness(
    report: Dict[str, Any],
) -> Dict[str, Any]:
    component_reports = report.get("component_reports", {})
    if not isinstance(component_reports, dict):
        component_reports = {}
    consolidation_component = (
        component_reports.get("continual_consolidation", {})
        if isinstance(component_reports.get("continual_consolidation"), dict)
        else {}
    )
    metrics = consolidation_component.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    tracked_metrics = {
        "replay_recovery_integrity": float(metrics.get("replay_recovery_integrity", 0.0)),
        "long_horizon_consolidation_retention": float(
            metrics.get("long_horizon_consolidation_retention", 0.0)
        ),
        "counterfactual_replay_selection_integrity": float(
            metrics.get("counterfactual_replay_selection_integrity", 0.0)
        ),
        "replay_upgrade_reindex_integrity": float(
            metrics.get("replay_upgrade_reindex_integrity", 0.0)
        ),
        "memory_health_index_integrity": float(
            metrics.get("memory_health_index_integrity", 0.0)
        ),
        "replay_noise_resilience_integrity": float(
            metrics.get("replay_noise_resilience_integrity", 0.0)
        ),
        "astro_modulation_stability": float(
            metrics.get("astro_modulation_stability", 0.0)
        ),
    }
    checks = {
        f"metric.{name}": value >= 1.0
        for name, value in tracked_metrics.items()
    }
    delta_memory_candidate_metrics = {
        name: float(metrics.get(name, 0.0))
        for name in STAGE_D_DELTA_MEMORY_PROMOTION_METRIC_NAMES
    }
    delta_memory_candidate_checks = {
        stage_d_metric_check_name(name): value >= 1.0
        for name, value in delta_memory_candidate_metrics.items()
    }
    delta_memory_candidate_promoted = all(
        name in STAGE_D_MINIMUM_METRIC_NAMES
        for name in STAGE_D_DELTA_MEMORY_PROMOTION_METRIC_NAMES
    )
    delta_memory_candidate_ready = all(delta_memory_candidate_checks.values()) and all(checks.values())
    delta_memory_candidate_failure_count = sum(
        1 for passed in delta_memory_candidate_checks.values() if not bool(passed)
    )
    delta_memory_candidate_failures = []
    for metric_name in STAGE_D_DELTA_MEMORY_PROMOTION_METRIC_NAMES:
        check_name = stage_d_metric_check_name(metric_name)
        if bool(delta_memory_candidate_checks.get(check_name, False)):
            continue
        delta_memory_candidate_failures.append(
            {
                "check": check_name,
                "metric": metric_name,
                "description": STAGE_D_DELTA_MEMORY_PROMOTION_CHECKS.get(check_name, metric_name),
                "value": float(delta_memory_candidate_metrics.get(metric_name, 0.0)),
                "threshold": 1.0,
            }
        )
    acceptance_candidate_metrics = {
        name: float(metrics.get(name, 0.0))
        for name in STAGE_D_ACCEPTANCE_CANDIDATE_METRIC_NAMES
    }
    acceptance_candidates = [
        {
            "check": stage_d_metric_check_name(metric_name),
            "metric": metric_name,
            "description": STAGE_D_ACCEPTANCE_CANDIDATE_CHECKS.get(
                stage_d_metric_check_name(metric_name),
                metric_name,
            ),
            "value": float(acceptance_candidate_metrics.get(metric_name, 0.0)),
            "threshold": 1.0,
            "ready": float(acceptance_candidate_metrics.get(metric_name, 0.0)) >= 1.0,
            "promoted_to_minimum": metric_name in STAGE_D_MINIMUM_METRIC_NAMES,
        }
        for metric_name in STAGE_D_ACCEPTANCE_CANDIDATE_METRIC_NAMES
    ]
    acceptance_candidate_failures = [
        dict(item)
        for item in acceptance_candidates
        if not bool(item.get("ready", False))
    ]
    readiness_score = sum(tracked_metrics.values()) / max(len(tracked_metrics), 1)
    minimum_checks = {
        stage_d_metric_check_name(name): checks.get(stage_d_metric_check_name(name), False)
        for name in STAGE_D_MINIMUM_METRIC_NAMES
    }
    minimum_failures = []
    for metric_name in STAGE_D_MINIMUM_METRIC_NAMES:
        check_name = stage_d_metric_check_name(metric_name)
        if bool(minimum_checks.get(check_name, False)):
            continue
        minimum_failures.append(
            {
                "check": check_name,
                "metric": metric_name,
                "description": STAGE_D_REQUIRED_MINIMUM_CHECKS.get(check_name, metric_name),
                "value": float(tracked_metrics.get(metric_name, 0.0)),
                "threshold": 1.0,
            }
        )

    return {
        "stage": "Stage D",
        "label": "Continual Consolidation",
        "passed": all(checks.values()),
        "minimum_requirements_passed": all(minimum_checks.values()),
        "minimum_failure_count": len(minimum_failures),
        "minimum_failures": minimum_failures,
        "delta_memory_candidate_ready": bool(delta_memory_candidate_ready),
        "delta_memory_candidate_failure_count": int(delta_memory_candidate_failure_count),
        "delta_memory_candidate_failures": delta_memory_candidate_failures,
        "delta_memory_candidate_promoted": bool(delta_memory_candidate_promoted),
        "delta_memory_candidate_checks": delta_memory_candidate_checks,
        "acceptance_candidates": acceptance_candidates,
        "acceptance_candidate_failures": acceptance_candidate_failures,
        "acceptance_candidate_count": int(len(acceptance_candidates)),
        "acceptance_candidate_ready_count": int(
            sum(1 for item in acceptance_candidates if bool(item.get("ready", False)))
        ),
        "acceptance_candidates_ready": all(bool(item.get("ready", False)) for item in acceptance_candidates),
        "acceptance_candidate_failure_count": int(
            len(acceptance_candidate_failures)
        ),
        "readiness_score": readiness_score,
        "checks": checks,
        "minimum_checks": minimum_checks,
        "metrics": {**tracked_metrics, **acceptance_candidate_metrics},
    }


def _build_stage_e_readiness(
    report: Dict[str, Any],
) -> Dict[str, Any]:
    component_reports = report.get("component_reports", {})
    if not isinstance(component_reports, dict):
        component_reports = {}
    cognitive_component = (
        component_reports.get("cognitive_runtime", {})
        if isinstance(component_reports.get("cognitive_runtime"), dict)
        else {}
    )
    metrics = cognitive_component.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    tracked_metrics = {
        "common_spike_space_integrity": float(metrics.get("common_spike_space_integrity", 0.0)),
        "temporal_compression_efficiency": float(metrics.get("temporal_compression_efficiency", 0.0)),
        "modality_temporal_budget_integrity": float(metrics.get("modality_temporal_budget_integrity", 0.0)),
        "dendritic_context_gate_stability": float(metrics.get("dendritic_context_gate_stability", 0.0)),
        "spiking_hjepa_latent_transition": float(metrics.get("spiking_hjepa_latent_transition", 0.0)),
        "reverse_reasoning_trace_integrity": float(metrics.get("reverse_reasoning_trace_integrity", 0.0)),
        "causal_candidate_trace_integrity": float(metrics.get("causal_candidate_trace_integrity", 0.0)),
        "module_orchestration_integrity": float(metrics.get("module_orchestration_integrity", 0.0)),
        "counterfactual_lane_integrity": float(metrics.get("counterfactual_lane_integrity", 0.0)),
        "action_trace_observability": float(metrics.get("action_trace_observability", 0.0)),
        "runtime_trace_replay_consistency": float(metrics.get("runtime_trace_replay_consistency", 0.0)),
    }
    checks = {
        f"metric.{name}": value >= 1.0
        for name, value in tracked_metrics.items()
    }
    readiness_score = sum(tracked_metrics.values()) / max(len(tracked_metrics), 1)
    minimum_checks = {
        stage_e_metric_check_name(name): checks.get(stage_e_metric_check_name(name), False)
        for name in STAGE_E_MINIMUM_METRIC_NAMES
    }
    minimum_failures = []
    for metric_name in STAGE_E_MINIMUM_METRIC_NAMES:
        check_name = stage_e_metric_check_name(metric_name)
        if bool(minimum_checks.get(check_name, False)):
            continue
        minimum_failures.append(
            {
                "check": check_name,
                "metric": metric_name,
                "description": STAGE_E_REQUIRED_MINIMUM_CHECKS.get(check_name, metric_name),
                "value": float(tracked_metrics.get(metric_name, 0.0)),
                "threshold": 1.0,
            }
        )
    observed_acceptance_candidate_metrics = {
        metric_name: float(metrics.get(metric_name, 0.0) or 0.0)
        for metric_name in STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_METRIC_NAMES
    }
    observed_acceptance_candidates = [
        {
            "metric": metric_name,
            "check": stage_e_metric_check_name(metric_name),
            "description": STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_CHECKS.get(
                stage_e_metric_check_name(metric_name),
                metric_name,
            ),
            "value": float(observed_acceptance_candidate_metrics.get(metric_name, 0.0)),
            "threshold": 1.0,
            "ready": float(observed_acceptance_candidate_metrics.get(metric_name, 0.0)) >= 1.0,
            "promoted_to_minimum": metric_name in STAGE_E_MINIMUM_METRIC_NAMES,
            "policy": "observed_only_acceptance_candidate",
        }
        for metric_name in STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_METRIC_NAMES
    ]
    observed_acceptance_candidate_failures = [
        dict(item)
        for item in observed_acceptance_candidates
        if not bool(item.get("ready", False))
    ]
    observed_acceptance_candidates_ready = all(
        bool(item.get("ready", False)) for item in observed_acceptance_candidates
    )

    return {
        "stage": "Stage E",
        "label": "Modular Cognitive Runtime",
        "passed": all(checks.values()),
        "minimum_requirements_passed": all(minimum_checks.values()),
        "minimum_failure_count": len(minimum_failures),
        "minimum_failures": minimum_failures,
        "readiness_score": readiness_score,
        "checks": checks,
        "minimum_checks": minimum_checks,
        "metrics": {**tracked_metrics, **observed_acceptance_candidate_metrics},
        "observed_acceptance_candidates": observed_acceptance_candidates,
        "observed_acceptance_candidate_count": int(len(observed_acceptance_candidates)),
        "observed_acceptance_candidate_ready_count": int(
            sum(1 for item in observed_acceptance_candidates if bool(item.get("ready", False)))
        ),
        "observed_acceptance_candidates_ready": bool(observed_acceptance_candidates_ready),
        "observed_acceptance_candidate_failure_count": int(
            len(observed_acceptance_candidate_failures)
        ),
        "observed_acceptance_candidate_failures": observed_acceptance_candidate_failures,
    }


def _build_phase3_completion(report: Dict[str, Any]) -> Dict[str, Any]:
    focus_summary = report.get("focus_summary", {}) if isinstance(report.get("focus_summary"), dict) else {}
    trend = report.get("trend", {}) if isinstance(report.get("trend"), dict) else {}
    stage_a = report.get("stage_a_acceptance", {}) if isinstance(report.get("stage_a_acceptance"), dict) else {}
    stage_b = report.get("stage_b_readiness", {}) if isinstance(report.get("stage_b_readiness"), dict) else {}
    stage_c = report.get("stage_c_readiness", {}) if isinstance(report.get("stage_c_readiness"), dict) else {}
    stage_d = report.get("stage_d_readiness", {}) if isinstance(report.get("stage_d_readiness"), dict) else {}
    stage_e = report.get("stage_e_readiness", {}) if isinstance(report.get("stage_e_readiness"), dict) else {}

    required_focus_names = [
        "few_shot",
        "continual",
        "retrieval_hygiene",
        "adaptive_readiness",
        "predictive_readiness",
        "efficiency_readiness",
        "consolidation_readiness",
        "cognitive_runtime_readiness",
        "phase5_entry_readiness",
    ]
    focus_checks = {
        f"focus.{name}.passed": bool(
            isinstance(focus_summary.get(name), dict)
            and focus_summary.get(name, {}).get("passed", False)
        )
        for name in required_focus_names
    }
    checks = {
        "overall.score_at_least_0_95": float(report.get("overall_score", 0.0)) >= 0.95,
        "trend.zero_regressions": int(trend.get("gate_regression_count", trend.get("regression_count", 0)) or 0) == 0,
        "stage_a.accepted": bool(stage_a.get("passed", False)),
        "stage_b.minimum_requirements_passed": bool(stage_b.get("minimum_requirements_passed", False)),
        "stage_c.minimum_requirements_passed": bool(stage_c.get("minimum_requirements_passed", False)),
        "stage_d.minimum_requirements_passed": bool(stage_d.get("minimum_requirements_passed", False)),
        "stage_e.minimum_requirements_passed": bool(stage_e.get("minimum_requirements_passed", False)),
        **focus_checks,
    }
    failed_checks = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "label": "Phase 3 Completion Gate",
        "passed": len(failed_checks) == 0,
        "checks": checks,
        "failed_checks": failed_checks,
        "completion_score": (
            sum(1 for passed in checks.values() if bool(passed)) / max(len(checks), 1)
        ),
    }


def format_phase3_accuracy_summary(report: Dict[str, Any]) -> str:
    focus_summary = report.get("focus_summary", {}) if isinstance(report.get("focus_summary"), dict) else {}
    focus_trend = report.get("focus_trend", {}) if isinstance(report.get("focus_trend"), dict) else {}
    trend = report.get("trend", {}) if isinstance(report.get("trend"), dict) else {}
    linear_snn_fusion_observed_trend = (
        report.get("linear_snn_fusion_observed_trend", {})
        if isinstance(report.get("linear_snn_fusion_observed_trend"), dict)
        else {}
    )
    stage_e_architecture_integration_observed_trend = (
        report.get("stage_e_architecture_integration_observed_trend", {})
        if isinstance(report.get("stage_e_architecture_integration_observed_trend"), dict)
        else {}
    )
    stage_a_acceptance = (
        report.get("stage_a_acceptance", {})
        if isinstance(report.get("stage_a_acceptance"), dict)
        else {}
    )
    stage_b_readiness = (
        report.get("stage_b_readiness", {})
        if isinstance(report.get("stage_b_readiness"), dict)
        else {}
    )
    stage_c_readiness = (
        report.get("stage_c_readiness", {})
        if isinstance(report.get("stage_c_readiness"), dict)
        else {}
    )
    stage_d_readiness = (
        report.get("stage_d_readiness", {})
        if isinstance(report.get("stage_d_readiness"), dict)
        else {}
    )
    stage_e_readiness = (
        report.get("stage_e_readiness", {})
        if isinstance(report.get("stage_e_readiness"), dict)
        else {}
    )
    phase3_completion = (
        report.get("phase3_completion", {})
        if isinstance(report.get("phase3_completion"), dict)
        else {}
    )
    few_shot = focus_summary.get("few_shot", {}) if isinstance(focus_summary.get("few_shot"), dict) else {}
    continual = focus_summary.get("continual", {}) if isinstance(focus_summary.get("continual"), dict) else {}
    retrieval_hygiene = (
        focus_summary.get("retrieval_hygiene", {})
        if isinstance(focus_summary.get("retrieval_hygiene"), dict)
        else {}
    )
    adaptive_readiness = (
        focus_summary.get("adaptive_readiness", {})
        if isinstance(focus_summary.get("adaptive_readiness"), dict)
        else {}
    )
    adaptive_metrics_detail = (
        adaptive_readiness.get("metrics", {})
        if isinstance(adaptive_readiness.get("metrics"), dict)
        else {}
    )
    predictive_readiness = (
        focus_summary.get("predictive_readiness", {})
        if isinstance(focus_summary.get("predictive_readiness"), dict)
        else {}
    )
    efficiency_readiness = (
        focus_summary.get("efficiency_readiness", {})
        if isinstance(focus_summary.get("efficiency_readiness"), dict)
        else {}
    )
    parameter_efficiency = (
        focus_summary.get("parameter_efficiency", {})
        if isinstance(focus_summary.get("parameter_efficiency"), dict)
        else {}
    )
    consolidation_readiness = (
        focus_summary.get("consolidation_readiness", {})
        if isinstance(focus_summary.get("consolidation_readiness"), dict)
        else {}
    )
    cognitive_runtime_readiness = (
        focus_summary.get("cognitive_runtime_readiness", {})
        if isinstance(focus_summary.get("cognitive_runtime_readiness"), dict)
        else {}
    )
    nested_memory_readiness = (
        focus_summary.get("nested_memory_readiness", {})
        if isinstance(focus_summary.get("nested_memory_readiness"), dict)
        else {}
    )
    phase5_entry_readiness = (
        focus_summary.get("phase5_entry_readiness", {})
        if isinstance(focus_summary.get("phase5_entry_readiness"), dict)
        else {}
    )
    efficiency_metrics_detail = (
        efficiency_readiness.get("metrics", {})
        if isinstance(efficiency_readiness.get("metrics"), dict)
        else {}
    )
    component_reports = report.get("component_reports", {})
    if not isinstance(component_reports, dict):
        component_reports = {}
    efficiency_component = (
        component_reports.get("energy_efficiency", {})
        if isinstance(component_reports.get("energy_efficiency"), dict)
        else {}
    )
    efficiency_component_details = (
        efficiency_component.get("details", {})
        if isinstance(efficiency_component.get("details"), dict)
        else {}
    )
    efficiency_component_metrics = (
        efficiency_component.get("metrics", {})
        if isinstance(efficiency_component.get("metrics"), dict)
        else {}
    )
    neuromorphic_profile_trend = (
        efficiency_component.get("neuromorphic_profile_trend", {})
        if isinstance(efficiency_component.get("neuromorphic_profile_trend"), dict)
        else {}
    )
    neuromorphic_trend_compact = compact_neuromorphic_profile_trend(
        neuromorphic_profile_trend
    )
    consolidation_component = (
        component_reports.get("continual_consolidation", {})
        if isinstance(component_reports.get("continual_consolidation"), dict)
        else {}
    )
    consolidation_component_metrics = (
        consolidation_component.get("metrics", {})
        if isinstance(consolidation_component.get("metrics"), dict)
        else {}
    )
    phase5_component = (
        component_reports.get("phase5_predictive_coding", {})
        if isinstance(component_reports.get("phase5_predictive_coding"), dict)
        else {}
    )
    phase5_component_metrics = (
        phase5_component.get("metrics", {})
        if isinstance(phase5_component.get("metrics"), dict)
        else {}
    )
    cognitive_manifold_trace_metrics = extract_cognitive_manifold_trace_metrics(report)
    cognitive_delta_memory_metrics = extract_cognitive_delta_memory_metrics(report)
    cognitive_linear_snn_fusion_metrics = extract_cognitive_linear_snn_fusion_metrics(report)
    cognitive_plastic_submodel_metrics = extract_cognitive_plastic_submodel_metrics(report)
    retrieval_hygiene_trend = (
        focus_trend.get("retrieval_hygiene", {})
        if isinstance(focus_trend.get("retrieval_hygiene"), dict)
        else {}
    )
    adaptive_readiness_trend = (
        focus_trend.get("adaptive_readiness", {})
        if isinstance(focus_trend.get("adaptive_readiness"), dict)
        else {}
    )
    predictive_readiness_trend = (
        focus_trend.get("predictive_readiness", {})
        if isinstance(focus_trend.get("predictive_readiness"), dict)
        else {}
    )
    efficiency_readiness_trend = (
        focus_trend.get("efficiency_readiness", {})
        if isinstance(focus_trend.get("efficiency_readiness"), dict)
        else {}
    )
    parameter_efficiency_trend = (
        focus_trend.get("parameter_efficiency", {})
        if isinstance(focus_trend.get("parameter_efficiency"), dict)
        else {}
    )
    consolidation_readiness_trend = (
        focus_trend.get("consolidation_readiness", {})
        if isinstance(focus_trend.get("consolidation_readiness"), dict)
        else {}
    )
    cognitive_runtime_readiness_trend = (
        focus_trend.get("cognitive_runtime_readiness", {})
        if isinstance(focus_trend.get("cognitive_runtime_readiness"), dict)
        else {}
    )
    nested_memory_readiness_trend = (
        focus_trend.get("nested_memory_readiness", {})
        if isinstance(focus_trend.get("nested_memory_readiness"), dict)
        else {}
    )
    phase5_entry_readiness_trend = (
        focus_trend.get("phase5_entry_readiness", {})
        if isinstance(focus_trend.get("phase5_entry_readiness"), dict)
        else {}
    )
    direction_shift_trend = _extract_metric_trend(
        trend,
        "agent_dialogue.direction_shift_following",
    )
    adaptation_parameter_integrity_trend = _extract_metric_trend(
        trend,
        "task_switch_adaptation.meta_adaptation_parameter_integrity",
    )
    predictive_command_trend = _extract_metric_trend(
        trend,
        "future_state_consistency.future_state_command_integrity",
    )
    predictive_counterfactual_trend = _extract_metric_trend(
        trend,
        "future_state_consistency.future_state_counterfactual_integrity",
    )
    predictive_counterfactual_usefulness_trend = _extract_metric_trend(
        trend,
        "future_state_consistency.future_state_counterfactual_usefulness",
    )
    predictive_branching_trend = _extract_metric_trend(
        trend,
        "future_state_consistency.future_state_branching_integrity",
    )
    predictive_options_trend = _extract_metric_trend(
        trend,
        "future_state_consistency.future_state_options_integrity",
    )
    predictive_ranking_trend = _extract_metric_trend(
        trend,
        "future_state_consistency.future_state_ranking_integrity",
    )
    predictive_decision_brief_trend = _extract_metric_trend(
        trend,
        "future_state_consistency.future_state_decision_brief_integrity",
    )
    predictive_choice_trend = _extract_metric_trend(
        trend,
        "future_state_consistency.future_state_choice_integrity",
    )
    predictive_choice_reason_trend = _extract_metric_trend(
        trend,
        "future_state_consistency.future_state_choice_reason_integrity",
    )
    hierarchical_context_trend = _extract_metric_trend(
        trend,
        "spiking_llm.hierarchical_context_integrity",
    )
    memory_per_success_trend = _extract_metric_trend(
        trend,
        "energy_efficiency.memory_per_success_proxy",
    )
    stochastic_readout_trend = _extract_metric_trend(
        trend,
        "energy_efficiency.stochastic_readout_integrity",
    )
    predictive_shift_trend = _extract_metric_trend(
        trend,
        "future_state_consistency.future_state_shift_tracking_integrity",
    )
    predictive_simulation_trend = _extract_metric_trend(
        trend,
        "future_state_consistency.future_state_simulation_integrity",
    )
    predictive_fluid_trace_trend = _extract_metric_trend(
        trend,
        "future_state_consistency.future_state_fluid_trace_integrity",
    )
    predictive_fluid_support_trend = _extract_metric_trend(
        trend,
        "future_state_consistency.future_state_fluid_support_integrity",
    )
    predictive_refinement_loop_trend = _extract_metric_trend(
        trend,
        "future_state_consistency.future_state_refinement_loop_integrity",
    )
    predictive_adaptive_refinement_trend = _extract_metric_trend(
        trend,
        "future_state_consistency.future_state_adaptive_refinement",
    )
    predictive_rewarded_action_selection_trend = _extract_metric_trend(
        trend,
        "future_state_consistency.future_state_rewarded_action_selection_integrity",
    )
    predictive_policy_update_stability_trend = _extract_metric_trend(
        trend,
        "future_state_consistency.future_state_policy_update_stability",
    )
    predictive_energy_aware_preference_trend = _extract_metric_trend(
        trend,
        "future_state_consistency.future_state_energy_aware_action_preference",
    )
    consolidation_replay_recovery_trend = _extract_metric_trend(
        trend,
        "continual_consolidation.replay_recovery_integrity",
    )
    consolidation_reindex_trend = _extract_metric_trend(
        trend,
        "continual_consolidation.replay_upgrade_reindex_integrity",
    )
    consolidation_health_index_trend = _extract_metric_trend(
        trend,
        "continual_consolidation.memory_health_index_integrity",
    )
    consolidation_replay_noise_resilience_trend = _extract_metric_trend(
        trend,
        "continual_consolidation.replay_noise_resilience_integrity",
    )
    consolidation_astro_modulation_trend = _extract_metric_trend(
        trend,
        "continual_consolidation.astro_modulation_stability",
    )
    nested_multi_rate_trend = _extract_metric_trend(
        trend,
        "nested_memory.multi_rate_update_integrity",
    )
    nested_energy_budget_trend = _extract_metric_trend(
        trend,
        "nested_memory.scheduler_energy_budget_integrity",
    )
    nested_interference_guard_trend = _extract_metric_trend(
        trend,
        "nested_memory.catastrophic_interference_guard",
    )
    common_spike_space_trend = _extract_metric_trend(
        trend,
        "cognitive_runtime.common_spike_space_integrity",
    )
    temporal_compression_trend = _extract_metric_trend(
        trend,
        "cognitive_runtime.temporal_compression_efficiency",
    )
    dendritic_context_gate_trend = _extract_metric_trend(
        trend,
        "cognitive_runtime.dendritic_context_gate_stability",
    )
    reverse_reasoning_trace_trend = _extract_metric_trend(
        trend,
        "cognitive_runtime.reverse_reasoning_trace_integrity",
    )
    causal_candidate_trace_trend = _extract_metric_trend(
        trend,
        "cognitive_runtime.causal_candidate_trace_integrity",
    )
    phase5_latent_transition_trend = _extract_metric_trend(
        trend,
        "phase5_predictive_coding.latent_transition_alignment",
    )
    phase5_correction_event_trend = _extract_metric_trend(
        trend,
        "phase5_predictive_coding.correction_event_coverage",
    )
    phase5_counterfactual_transition_trend = _extract_metric_trend(
        trend,
        "phase5_predictive_coding.counterfactual_transition_separation",
    )
    phase5_multistep_chain_trend = _extract_metric_trend(
        trend,
        "phase5_predictive_coding.multi_step_latent_chain_integrity",
    )
    phase5_error_convergence_trend = _extract_metric_trend(
        trend,
        "phase5_predictive_coding.long_horizon_error_correction_convergence",
    )
    agent_dialogue_component = (
        component_reports.get("agent_dialogue", {})
        if isinstance(component_reports.get("agent_dialogue"), dict)
        else {}
    )
    agent_dialogue_metrics = (
        agent_dialogue_component.get("metrics", {})
        if isinstance(agent_dialogue_component.get("metrics"), dict)
        else {}
    )
    agent_dialogue_details = (
        agent_dialogue_component.get("details", {})
        if isinstance(agent_dialogue_component.get("details"), dict)
        else {}
    )
    agent_dialogue_results = (
        agent_dialogue_details.get("test_results", [])
        if isinstance(agent_dialogue_details.get("test_results"), list)
        else []
    )

    lines = [
        "SARA Engine Phase 3 Accuracy Summary",
        f"overall_status: {_status_label(bool(report.get('passed', False)))}",
        f"overall_score: {float(report.get('overall_score', 0.0)):.3f}",
        f"regression_count: {int(trend.get('regression_count', 0))}",
        f"gate_regression_count: {int(trend.get('gate_regression_count', trend.get('regression_count', 0)) or 0)}",
        "",
        "Phase 3 Completion",
        f"- phase3_completion_status: {_status_label(bool(phase3_completion.get('passed', False)))}",
        f"- phase3_completion_score: {float(phase3_completion.get('completion_score', 0.0)):.3f}",
        f"- phase3_completion_failed_checks: {int(len(phase3_completion.get('failed_checks', [])) if isinstance(phase3_completion.get('failed_checks', []), list) else 0)}",
        "",
        "Stage A Acceptance",
        f"- stage_a_status: {_status_label(bool(stage_a_acceptance.get('passed', False)))}",
        f"- stage_a_label: {stage_a_acceptance.get('label', 'Evaluation First')}",
        f"- stage_a_acc_target: {float(stage_a_acceptance.get('acc_target', 0.95)):.3f}",
        f"- stage_a_overall_score: {float(stage_a_acceptance.get('overall_score', report.get('overall_score', 0.0))):.3f}",
        f"- stage_a_zero_regressions: {bool(stage_a_acceptance.get('checks', {}).get('trend.zero_regressions', False))}",
        f"- stage_a_acc_target_met: {bool(stage_a_acceptance.get('checks', {}).get('overall.acc_target_0_95', False))}",
        "",
        "Stage B Readiness",
        f"- stage_b_status: {_status_label(bool(stage_b_readiness.get('passed', False)))}",
        f"- stage_b_label: {stage_b_readiness.get('label', 'Lightweight World Model Prototypes')}",
        f"- stage_b_readiness_score: {float(stage_b_readiness.get('readiness_score', 0.0)):.3f}",
        f"- stage_b_minimum_requirements_passed: {bool(stage_b_readiness.get('minimum_requirements_passed', False))}",
        f"- stage_b_minimum_failure_count: {int(stage_b_readiness.get('minimum_failure_count', 0) or 0)}",
        f"- stage_b_transition_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_transition_integrity', False))}",
        f"- stage_b_command_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_command_integrity', False))}",
        f"- stage_b_predictor_snapshot_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_predictor_snapshot_integrity', False))}",
        f"- stage_b_runtime_tracking_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_runtime_tracking_integrity', False))}",
        f"- stage_b_shift_tracking_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_shift_tracking_integrity', False))}",
        f"- stage_b_operator_coverage_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_transition_operator_coverage', False))}",
        f"- stage_b_operator_consistency_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_transition_operator_consistency', False))}",
        f"- stage_b_counterfactual_viability_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_counterfactual_branch_viability', False))}",
        f"- stage_b_fluid_trace_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_fluid_trace_integrity', False))}",
        f"- stage_b_fluid_support_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_fluid_support_integrity', False))}",
        f"- stage_b_refinement_loop_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_refinement_loop_integrity', False))}",
        f"- stage_b_adaptive_refinement_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_adaptive_refinement', False))}",
        f"- stage_b_branching_ready: {bool(stage_b_readiness.get('checks', {}).get('metric.future_state_branching_integrity', False))}",
        f"- stage_b_simulation_ready: {bool(stage_b_readiness.get('checks', {}).get('metric.future_state_simulation_integrity', False))}",
        f"- stage_b_simulation_useful: {bool(stage_b_readiness.get('checks', {}).get('metric.future_state_simulation_usefulness', False))}",
        f"- stage_b_speculative_acceptance_ready: {bool(stage_b_readiness.get('checks', {}).get('metric.future_state_speculative_acceptance_ratio', False))}",
        f"- stage_b_speculative_rollback_ready: {bool(stage_b_readiness.get('checks', {}).get('metric.future_state_speculative_rollback_observability', False))}",
        f"- stage_b_promotion_candidate_ready: {bool(stage_b_readiness.get('promotion_candidate_ready', False))}",
        f"- stage_b_promotion_candidate_failure_count: {int(stage_b_readiness.get('promotion_candidate_failure_count', 0) or 0)}",
        f"- stage_b_promotion_candidate_promoted: {bool(stage_b_readiness.get('promotion_candidate_promoted', False))}",
        f"- stage_b_promotion_consecutive_passes: {int(stage_b_readiness.get('promotion_readiness', {}).get('consecutive_passes', 0) if isinstance(stage_b_readiness.get('promotion_readiness'), dict) else 0)}",
        f"- stage_b_promotion_required_streak: {int(stage_b_readiness.get('promotion_readiness', {}).get('required_streak', 3) if isinstance(stage_b_readiness.get('promotion_readiness'), dict) else 3)}",
        f"- stage_b_promotion_recommended: {bool(stage_b_readiness.get('promotion_readiness', {}).get('recommended', False) if isinstance(stage_b_readiness.get('promotion_readiness'), dict) else False)}",
        f"- stage_b_rewarded_action_selection_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_rewarded_action_selection_integrity', False))}",
        f"- stage_b_policy_update_stability_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_policy_update_stability', False))}",
        f"- stage_b_energy_aware_preference_ready: {bool(stage_b_readiness.get('minimum_checks', {}).get('metric.future_state_energy_aware_action_preference', False))}",
        f"- stage_b_rewarded_action_selection_observed: {bool(stage_b_readiness.get('checks', {}).get('metric.future_state_rewarded_action_selection_integrity', False))}",
        f"- stage_b_policy_update_stability_observed: {bool(stage_b_readiness.get('checks', {}).get('metric.future_state_policy_update_stability', False))}",
        f"- stage_b_energy_aware_preference_observed: {bool(stage_b_readiness.get('checks', {}).get('metric.future_state_energy_aware_action_preference', False))}",
        f"- stage_b_focused_retrieval_observed: {bool(stage_b_readiness.get('checks', {}).get('metric.future_state_focused_retrieval_hit_ratio', False))}",
        f"- stage_b_branch_decision_consistency_observed: {bool(stage_b_readiness.get('checks', {}).get('metric.future_state_branch_level_decision_consistency', False))}",
        f"- stage_b_rlm_observation_candidate_ready: {bool(stage_b_readiness.get('rlm_observation_candidate_ready', False))}",
        f"- stage_b_rlm_observation_candidate_failure_count: {int(stage_b_readiness.get('rlm_observation_candidate_failure_count', 0) or 0)}",
        f"- stage_b_rlm_observation_candidate_promoted: {bool(stage_b_readiness.get('rlm_observation_candidate_promoted', False))}",
        f"- stage_b_rlm_observation_consecutive_passes: {int(stage_b_readiness.get('rlm_observation_promotion_readiness', {}).get('consecutive_passes', 0) if isinstance(stage_b_readiness.get('rlm_observation_promotion_readiness'), dict) else 0)}",
        f"- stage_b_rlm_observation_required_streak: {int(stage_b_readiness.get('rlm_observation_promotion_readiness', {}).get('required_streak', 3) if isinstance(stage_b_readiness.get('rlm_observation_promotion_readiness'), dict) else 3)}",
        f"- stage_b_rlm_observation_promotion_recommended: {bool(stage_b_readiness.get('rlm_observation_promotion_readiness', {}).get('recommended', False) if isinstance(stage_b_readiness.get('rlm_observation_promotion_readiness'), dict) else False)}",
        "",
        "Stage C Readiness",
        f"- stage_c_status: {_status_label(bool(stage_c_readiness.get('passed', False)))}",
        f"- stage_c_label: {stage_c_readiness.get('label', 'Meta-Adaptation Experiments')}",
        f"- stage_c_readiness_score: {float(stage_c_readiness.get('readiness_score', 0.0)):.3f}",
        f"- stage_c_minimum_requirements_passed: {bool(stage_c_readiness.get('minimum_requirements_passed', False))}",
        f"- stage_c_minimum_failure_count: {int(stage_c_readiness.get('minimum_failure_count', 0) or 0)}",
        f"- stage_c_meta_adaptation_loop_ready: {bool(stage_c_readiness.get('minimum_checks', {}).get('metric.meta_adaptation_loop', False))}",
        f"- stage_c_parameter_integrity_ready: {bool(stage_c_readiness.get('minimum_checks', {}).get('metric.meta_adaptation_parameter_integrity', False))}",
        f"- stage_c_temporal_self_distillation_ready: {bool(stage_c_readiness.get('minimum_checks', {}).get('metric.temporal_self_distillation_stability', False))}",
        "",
        "Stage D Readiness",
        f"- stage_d_status: {_status_label(bool(stage_d_readiness.get('passed', False)))}",
        f"- stage_d_label: {stage_d_readiness.get('label', 'Continual Consolidation')}",
        f"- stage_d_readiness_score: {float(stage_d_readiness.get('readiness_score', 0.0)):.3f}",
        f"- stage_d_minimum_requirements_passed: {bool(stage_d_readiness.get('minimum_requirements_passed', False))}",
        f"- stage_d_minimum_failure_count: {int(stage_d_readiness.get('minimum_failure_count', 0) or 0)}",
        f"- stage_d_replay_recovery_ready: {bool(stage_d_readiness.get('minimum_checks', {}).get('metric.replay_recovery_integrity', False))}",
        f"- stage_d_long_horizon_retention_ready: {bool(stage_d_readiness.get('minimum_checks', {}).get('metric.long_horizon_consolidation_retention', False))}",
        f"- stage_d_counterfactual_replay_ready: {bool(stage_d_readiness.get('minimum_checks', {}).get('metric.counterfactual_replay_selection_integrity', False))}",
        f"- stage_d_reindex_ready: {bool(stage_d_readiness.get('minimum_checks', {}).get('metric.replay_upgrade_reindex_integrity', False))}",
        f"- stage_d_memory_health_ready: {bool(stage_d_readiness.get('minimum_checks', {}).get('metric.memory_health_index_integrity', False))}",
        f"- stage_d_replay_noise_resilience_ready: {bool(stage_d_readiness.get('minimum_checks', {}).get('metric.replay_noise_resilience_integrity', False))}",
        f"- stage_d_astro_modulation_ready: {bool(stage_d_readiness.get('minimum_checks', {}).get('metric.astro_modulation_stability', False))}",
        f"- stage_d_delta_memory_candidate_ready: {bool(stage_d_readiness.get('delta_memory_candidate_ready', False))}",
        f"- stage_d_delta_memory_candidate_failure_count: {int(stage_d_readiness.get('delta_memory_candidate_failure_count', 0) or 0)}",
        f"- stage_d_delta_memory_candidate_promoted: {bool(stage_d_readiness.get('delta_memory_candidate_promoted', False))}",
        f"- stage_d_delta_memory_consecutive_passes: {int(stage_d_readiness.get('delta_memory_promotion_readiness', {}).get('consecutive_passes', 0) if isinstance(stage_d_readiness.get('delta_memory_promotion_readiness'), dict) else 0)}",
        f"- stage_d_delta_memory_required_streak: {int(stage_d_readiness.get('delta_memory_promotion_readiness', {}).get('required_streak', 3) if isinstance(stage_d_readiness.get('delta_memory_promotion_readiness'), dict) else 3)}",
        f"- stage_d_delta_memory_promotion_recommended: {bool(stage_d_readiness.get('delta_memory_promotion_readiness', {}).get('recommended', False) if isinstance(stage_d_readiness.get('delta_memory_promotion_readiness'), dict) else False)}",
        f"- stage_d_acceptance_candidate_count: {int(stage_d_readiness.get('acceptance_candidate_count', 0) or 0)}",
        f"- stage_d_acceptance_candidate_ready_count: {int(stage_d_readiness.get('acceptance_candidate_ready_count', 0) or 0)}",
        f"- stage_d_acceptance_candidates_ready: {bool(stage_d_readiness.get('acceptance_candidates_ready', False))}",
        f"- stage_d_acceptance_candidate_failure_count: {int(stage_d_readiness.get('acceptance_candidate_failure_count', 0) or 0)}",
        f"- stage_d_acceptance_candidate_consecutive_passes: {int(stage_d_readiness.get('acceptance_candidate_stability', {}).get('consecutive_passes', 0) if isinstance(stage_d_readiness.get('acceptance_candidate_stability'), dict) else 0)}",
        f"- stage_d_acceptance_candidate_required_streak: {int(stage_d_readiness.get('acceptance_candidate_stability', {}).get('required_streak', 3) if isinstance(stage_d_readiness.get('acceptance_candidate_stability'), dict) else 3)}",
        f"- stage_d_acceptance_candidate_stability_recommended: {bool(stage_d_readiness.get('acceptance_candidate_stability', {}).get('recommended', False) if isinstance(stage_d_readiness.get('acceptance_candidate_stability'), dict) else False)}",
        "",
        "Stage E Readiness",
        f"- stage_e_status: {_status_label(bool(stage_e_readiness.get('passed', False)))}",
        f"- stage_e_label: {stage_e_readiness.get('label', 'Modular Cognitive Runtime')}",
        f"- stage_e_readiness_score: {float(stage_e_readiness.get('readiness_score', 0.0)):.3f}",
        f"- stage_e_minimum_requirements_passed: {bool(stage_e_readiness.get('minimum_requirements_passed', False))}",
        f"- stage_e_minimum_failure_count: {int(stage_e_readiness.get('minimum_failure_count', 0) or 0)}",
        f"- stage_e_observed_acceptance_candidate_count: {int(stage_e_readiness.get('observed_acceptance_candidate_count', 0) or 0)}",
        f"- stage_e_observed_acceptance_candidate_ready_count: {int(stage_e_readiness.get('observed_acceptance_candidate_ready_count', 0) or 0)}",
        f"- stage_e_observed_acceptance_candidates_ready: {bool(stage_e_readiness.get('observed_acceptance_candidates_ready', False))}",
        f"- stage_e_observed_acceptance_candidate_failure_count: {int(stage_e_readiness.get('observed_acceptance_candidate_failure_count', 0) or 0)}",
        f"- stage_e_observed_acceptance_candidate_consecutive_passes: {int(stage_e_readiness.get('observed_acceptance_candidate_stability', {}).get('consecutive_passes', 0) if isinstance(stage_e_readiness.get('observed_acceptance_candidate_stability'), dict) else 0)}",
        f"- stage_e_observed_acceptance_candidate_required_streak: {int(stage_e_readiness.get('observed_acceptance_candidate_stability', {}).get('required_streak', 3) if isinstance(stage_e_readiness.get('observed_acceptance_candidate_stability'), dict) else 3)}",
        f"- stage_e_observed_acceptance_candidate_stability_recommended: {bool(stage_e_readiness.get('observed_acceptance_candidate_stability', {}).get('recommended', False) if isinstance(stage_e_readiness.get('observed_acceptance_candidate_stability'), dict) else False)}",
        f"- stage_e_common_spike_space_ready: {bool(stage_e_readiness.get('minimum_checks', {}).get('metric.common_spike_space_integrity', False))}",
        f"- stage_e_temporal_compression_ready: {bool(stage_e_readiness.get('minimum_checks', {}).get('metric.temporal_compression_efficiency', False))}",
        f"- stage_e_modality_budget_ready: {bool(stage_e_readiness.get('minimum_checks', {}).get('metric.modality_temporal_budget_integrity', False))}",
        f"- stage_e_dendritic_gate_ready: {bool(stage_e_readiness.get('minimum_checks', {}).get('metric.dendritic_context_gate_stability', False))}",
        f"- stage_e_spiking_hjepa_ready: {bool(stage_e_readiness.get('minimum_checks', {}).get('metric.spiking_hjepa_latent_transition', False))}",
        f"- stage_e_reverse_reasoning_ready: {bool(stage_e_readiness.get('minimum_checks', {}).get('metric.reverse_reasoning_trace_integrity', False))}",
        f"- stage_e_causal_candidate_trace_ready: {bool(stage_e_readiness.get('minimum_checks', {}).get('metric.causal_candidate_trace_integrity', False))}",
        f"- stage_e_module_orchestration_ready: {bool(stage_e_readiness.get('minimum_checks', {}).get('metric.module_orchestration_integrity', False))}",
        f"- stage_e_counterfactual_lane_ready: {bool(stage_e_readiness.get('minimum_checks', {}).get('metric.counterfactual_lane_integrity', False))}",
        f"- stage_e_action_trace_ready: {bool(stage_e_readiness.get('minimum_checks', {}).get('metric.action_trace_observability', False))}",
        f"- stage_e_runtime_trace_replay_ready: {bool(stage_e_readiness.get('minimum_checks', {}).get('metric.runtime_trace_replay_consistency', False))}",
        "",
        "Focus",
        f"- few_shot_status: {_status_label(bool(few_shot.get('passed', False)))}",
        f"- few_shot_score: {float(few_shot.get('score', 0.0)):.3f}",
        f"- hierarchical_context_trend: {hierarchical_context_trend.get('status', 'NEW')}",
        f"- hierarchical_context_delta: {float(hierarchical_context_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- continual_status: {_status_label(bool(continual.get('passed', False)))}",
        f"- continual_score: {float(continual.get('score', 0.0)):.3f}",
        f"- retrieval_hygiene_status: {_status_label(bool(retrieval_hygiene.get('passed', False)))}",
        f"- retrieval_hygiene_score: {float(retrieval_hygiene.get('score', 0.0)):.3f}",
        f"- retrieval_hygiene_trend: {retrieval_hygiene_trend.get('status', 'NEW')}",
        f"- retrieval_hygiene_delta: {float(retrieval_hygiene_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- adaptive_readiness_status: {_status_label(bool(adaptive_readiness.get('passed', False)))}",
        f"- adaptive_readiness_score: {float(adaptive_readiness.get('score', 0.0)):.3f}",
        f"- adaptive_readiness_trend: {adaptive_readiness_trend.get('status', 'NEW')}",
        f"- adaptive_readiness_delta: {float(adaptive_readiness_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- adaptation_parameter_integrity: {float(adaptive_metrics_detail.get('task_switch_adaptation.meta_adaptation_parameter_integrity', 0.0)):.3f}",
        f"- adaptation_parameter_integrity_trend: {adaptation_parameter_integrity_trend.get('status', 'NEW')}",
        f"- adaptation_parameter_integrity_delta: {float(adaptation_parameter_integrity_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- direction_shift_following: {float(agent_dialogue_metrics.get('direction_shift_following', 0.0)):.3f}",
        f"- direction_shift_trend: {direction_shift_trend.get('status', 'NEW')}",
        f"- direction_shift_delta: {float(direction_shift_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- predictive_readiness_status: {_status_label(bool(predictive_readiness.get('passed', False)))}",
        f"- predictive_readiness_score: {float(predictive_readiness.get('score', 0.0)):.3f}",
        f"- predictive_readiness_trend: {predictive_readiness_trend.get('status', 'NEW')}",
        f"- predictive_readiness_delta: {float(predictive_readiness_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- predictive_command_trend: {predictive_command_trend.get('status', 'NEW')}",
        f"- predictive_command_delta: {float(predictive_command_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- predictive_counterfactual_trend: {predictive_counterfactual_trend.get('status', 'NEW')}",
        f"- predictive_counterfactual_delta: {float(predictive_counterfactual_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- predictive_counterfactual_usefulness_trend: {predictive_counterfactual_usefulness_trend.get('status', 'NEW')}",
        f"- predictive_counterfactual_usefulness_delta: {float(predictive_counterfactual_usefulness_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- predictive_branching_trend: {predictive_branching_trend.get('status', 'NEW')}",
        f"- predictive_branching_delta: {float(predictive_branching_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- predictive_options_trend: {predictive_options_trend.get('status', 'NEW')}",
        f"- predictive_options_delta: {float(predictive_options_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- predictive_ranking_trend: {predictive_ranking_trend.get('status', 'NEW')}",
        f"- predictive_ranking_delta: {float(predictive_ranking_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- predictive_decision_brief_trend: {predictive_decision_brief_trend.get('status', 'NEW')}",
        f"- predictive_decision_brief_delta: {float(predictive_decision_brief_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- predictive_choice_trend: {predictive_choice_trend.get('status', 'NEW')}",
        f"- predictive_choice_delta: {float(predictive_choice_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- predictive_choice_reason_trend: {predictive_choice_reason_trend.get('status', 'NEW')}",
        f"- predictive_choice_reason_delta: {float(predictive_choice_reason_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- predictive_shift_trend: {predictive_shift_trend.get('status', 'NEW')}",
        f"- predictive_shift_delta: {float(predictive_shift_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- predictive_simulation_trend: {predictive_simulation_trend.get('status', 'NEW')}",
        f"- predictive_simulation_delta: {float(predictive_simulation_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- predictive_fluid_trace_trend: {predictive_fluid_trace_trend.get('status', 'NEW')}",
        f"- predictive_fluid_trace_delta: {float(predictive_fluid_trace_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- predictive_fluid_support_trend: {predictive_fluid_support_trend.get('status', 'NEW')}",
        f"- predictive_fluid_support_delta: {float(predictive_fluid_support_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- predictive_refinement_loop_trend: {predictive_refinement_loop_trend.get('status', 'NEW')}",
        f"- predictive_refinement_loop_delta: {float(predictive_refinement_loop_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- predictive_adaptive_refinement_trend: {predictive_adaptive_refinement_trend.get('status', 'NEW')}",
        f"- predictive_adaptive_refinement_delta: {float(predictive_adaptive_refinement_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- predictive_rewarded_action_selection_trend: {predictive_rewarded_action_selection_trend.get('status', 'NEW')}",
        f"- predictive_rewarded_action_selection_delta: {float(predictive_rewarded_action_selection_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- predictive_policy_update_stability_trend: {predictive_policy_update_stability_trend.get('status', 'NEW')}",
        f"- predictive_policy_update_stability_delta: {float(predictive_policy_update_stability_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- predictive_energy_aware_preference_trend: {predictive_energy_aware_preference_trend.get('status', 'NEW')}",
        f"- predictive_energy_aware_preference_delta: {float(predictive_energy_aware_preference_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- efficiency_readiness_status: {_status_label(bool(efficiency_readiness.get('passed', False)))}",
        f"- efficiency_readiness_score: {float(efficiency_readiness.get('score', 0.0)):.3f}",
        f"- energy_per_success_proxy: {float(efficiency_metrics_detail.get('energy_efficiency.energy_per_success_proxy', 0.0)):.3f}",
        f"- performance_energy_ratio_proxy: {float(efficiency_metrics_detail.get('energy_efficiency.performance_energy_ratio_proxy', 0.0)):.3f}",
        f"- ann_cost_advantage_proxy: {float(efficiency_metrics_detail.get('energy_efficiency.ann_cost_advantage_proxy', 0.0)):.3f}",
        f"- sparse_event_cost_score: {float(efficiency_metrics_detail.get('energy_efficiency.sparse_event_cost_score', 0.0)):.3f}",
        f"- brain_efficiency_alignment_proxy: {float(efficiency_metrics_detail.get('energy_efficiency.brain_efficiency_alignment_proxy', 0.0)):.3f}",
        f"- memory_per_success_proxy: {float(efficiency_metrics_detail.get('energy_efficiency.memory_per_success_proxy', 0.0)):.3f}",
        f"- low_overhead_route_score: {float(efficiency_metrics_detail.get('energy_efficiency.low_overhead_route_score', 0.0)):.3f}",
        f"- bounded_latency_score: {float(efficiency_metrics_detail.get('energy_efficiency.bounded_latency_score', 0.0)):.3f}",
        f"- stochastic_readout_integrity: {float(efficiency_metrics_detail.get('energy_efficiency.stochastic_readout_integrity', 0.0)):.3f}",
        f"- edge_delta_state_persistence_observed: {float(efficiency_component_metrics.get('edge_delta_state_persistence_observed', 0.0)):.3f}",
        f"- edge_delta_state_budget_observed: {float(efficiency_component_metrics.get('edge_delta_state_budget_observed', 0.0)):.3f}",
        f"- edge_delta_state_manifest_integrity_observed: {float(efficiency_component_metrics.get('edge_delta_state_manifest_integrity_observed', 0.0)):.3f}",
        f"- neuromorphic_ir_schema_integrity_observed: {float(efficiency_component_metrics.get('neuromorphic_ir_schema_integrity_observed', 0.0)):.3f}",
        f"- neuromorphic_capability_manifest_integrity_observed: {float(efficiency_component_metrics.get('neuromorphic_capability_manifest_integrity_observed', 0.0)):.3f}",
        f"- neuromorphic_backend_profile_compatibility_observed: {float(efficiency_component_metrics.get('neuromorphic_backend_profile_compatibility_observed', 0.0)):.3f}",
        f"- neuromorphic_sparse_event_budget_observed: {float(efficiency_component_metrics.get('neuromorphic_sparse_event_budget_observed', 0.0)):.3f}",
        f"- neuromorphic_profile_report_integrity_observed: {float(efficiency_component_metrics.get('neuromorphic_profile_report_integrity_observed', 0.0)):.3f}",
        f"- neuromorphic_stage_e_state_trace_ir_observed: {float(efficiency_component_metrics.get('neuromorphic_stage_e_state_trace_ir_observed', 0.0)):.3f}",
        f"- neuromorphic_stage_e_routing_hint_coverage_observed: {float(efficiency_component_metrics.get('neuromorphic_stage_e_routing_hint_coverage_observed', 0.0)):.3f}",
        f"- neuromorphic_stage_e_online_update_policy_observed: {float(efficiency_component_metrics.get('neuromorphic_stage_e_online_update_policy_observed', 0.0)):.3f}",
        f"- neuromorphic_stage_e_event_budget_observed: {float(efficiency_component_metrics.get('neuromorphic_stage_e_event_budget_observed', 0.0)):.3f}",
        f"- neuromorphic_profile_history_regression_observed: {float(efficiency_component_metrics.get('neuromorphic_profile_history_regression_observed', 0.0)):.3f}",
        f"- neuromorphic_profile_trend_has_previous: {bool(neuromorphic_profile_trend.get('has_previous', False))}",
        f"- neuromorphic_profile_trend_regression_count: {int(neuromorphic_profile_trend.get('regression_count', 0) or 0)}",
        f"- neuromorphic_profile_trend_policy_change_count: {int(neuromorphic_profile_trend.get('policy_change_count', 0) or 0)}",
        f"- neuromorphic_profile_trend_regression_details: {str(neuromorphic_trend_compact.get('regression_detail_line', 'none') or 'none')}",
        f"- neuromorphic_profile_trend_policy_change_details: {str(neuromorphic_trend_compact.get('policy_change_detail_line', 'none') or 'none')}",
        f"- average_state_units: {float(efficiency_component_details.get('average_state_units', 0.0) or 0.0):.3f}",
        f"- memory_per_success_trend: {memory_per_success_trend.get('status', 'NEW')}",
        f"- memory_per_success_delta: {float(memory_per_success_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- stochastic_readout_trend: {stochastic_readout_trend.get('status', 'NEW')}",
        f"- stochastic_readout_delta: {float(stochastic_readout_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- efficiency_readiness_trend: {efficiency_readiness_trend.get('status', 'NEW')}",
        f"- efficiency_readiness_delta: {float(efficiency_readiness_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- parameter_efficiency_status: {_status_label(bool(parameter_efficiency.get('passed', False)))}",
        f"- parameter_efficiency_score: {float(parameter_efficiency.get('score', 0.0)):.3f}",
        f"- average_quality_per_kparam: {float(parameter_efficiency.get('metrics', {}).get('parameter_efficiency.average_quality_per_kparam', 0.0)):.3f}",
        f"- average_quality_per_mb: {float(parameter_efficiency.get('metrics', {}).get('parameter_efficiency.average_quality_per_mb', 0.0)):.3f}",
        f"- parameter_efficiency_trend: {parameter_efficiency_trend.get('status', 'NEW')}",
        f"- parameter_efficiency_delta: {float(parameter_efficiency_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- consolidation_readiness_status: {_status_label(bool(consolidation_readiness.get('passed', False)))}",
        f"- consolidation_readiness_score: {float(consolidation_readiness.get('score', 0.0)):.3f}",
        f"- consolidation_replay_recovery_integrity: {float(consolidation_readiness.get('metrics', {}).get('continual_consolidation.replay_recovery_integrity', 0.0)):.3f}",
        f"- consolidation_replay_upgrade_reindex_integrity: {float(consolidation_readiness.get('metrics', {}).get('continual_consolidation.replay_upgrade_reindex_integrity', 0.0)):.3f}",
        f"- consolidation_memory_health_index_integrity: {float(consolidation_readiness.get('metrics', {}).get('continual_consolidation.memory_health_index_integrity', 0.0)):.3f}",
        f"- consolidation_replay_noise_resilience_integrity: {float(consolidation_readiness.get('metrics', {}).get('continual_consolidation.replay_noise_resilience_integrity', 0.0)):.3f}",
        f"- consolidation_astro_modulation_stability: {float(consolidation_readiness.get('metrics', {}).get('continual_consolidation.astro_modulation_stability', 0.0)):.3f}",
        f"- consolidation_delta_memory_residual_write_integrity_observed: {float(consolidation_component_metrics.get('delta_memory_residual_write_integrity_observed', 0.0)):.3f}",
        f"- consolidation_delta_memory_retention_gate_stability_observed: {float(consolidation_component_metrics.get('delta_memory_retention_gate_stability_observed', 0.0)):.3f}",
        f"- consolidation_delta_memory_context_recall_without_text_reinjection_observed: {float(consolidation_component_metrics.get('delta_memory_context_recall_without_text_reinjection_observed', 0.0)):.3f}",
        f"- consolidation_delta_memory_state_budget_integrity_observed: {float(consolidation_component_metrics.get('delta_memory_state_budget_integrity_observed', 0.0)):.3f}",
        f"- consolidation_delta_memory_interference_guard_observed: {float(consolidation_component_metrics.get('delta_memory_interference_guard_observed', 0.0)):.3f}",
        f"- consolidation_manifold_continual_retention_observed: {float(consolidation_component_metrics.get('manifold_continual_retention_observed', 0.0)):.3f}",
        f"- consolidation_manifold_trajectory_case_coverage_observed: {float(consolidation_component_metrics.get('manifold_trajectory_case_coverage_observed', 0.0)):.3f}",
        f"- consolidation_manifold_average_case_recall_observed: {float(consolidation_component_metrics.get('manifold_average_case_recall_observed', 0.0)):.3f}",
        f"- consolidation_manifold_scan_budget_integrity_observed: {float(consolidation_component_metrics.get('manifold_scan_budget_integrity_observed', 0.0)):.3f}",
        f"- consolidation_manifold_indexed_candidate_integrity_observed: {float(consolidation_component_metrics.get('manifold_indexed_candidate_integrity_observed', 0.0)):.3f}",
        f"- consolidation_manifold_index_scan_reduction_observed: {float(consolidation_component_metrics.get('manifold_index_scan_reduction_observed', 0.0)):.3f}",
        f"- consolidation_manifold_capacity_pressure_recall_observed: {float(consolidation_component_metrics.get('manifold_capacity_pressure_recall_observed', 0.0)):.3f}",
        f"- consolidation_manifold_capacity_pressure_scan_reduction_observed: {float(consolidation_component_metrics.get('manifold_capacity_pressure_scan_reduction_observed', 0.0)):.3f}",
        f"- consolidation_manifold_replay_refresh_retention_observed: {float(consolidation_component_metrics.get('manifold_replay_refresh_retention_observed', 0.0)):.3f}",
        f"- consolidation_manifold_replay_refresh_eviction_integrity_observed: {float(consolidation_component_metrics.get('manifold_replay_refresh_eviction_integrity_observed', 0.0)):.3f}",
        f"- consolidation_synaptic_tag_integrity_observed: {float(consolidation_component_metrics.get('synaptic_tag_integrity_observed', 0.0)):.3f}",
        f"- consolidation_synaptic_tag_importance_score_observed: {float(consolidation_component_metrics.get('synaptic_tag_importance_score_observed', 0.0)):.3f}",
        f"- consolidation_synaptic_tag_replay_priority_observed: {float(consolidation_component_metrics.get('synaptic_tag_replay_priority_observed', 0.0)):.3f}",
        f"- consolidation_synaptic_tag_pruning_candidate_observed: {float(consolidation_component_metrics.get('synaptic_tag_pruning_candidate_observed', 0.0)):.3f}",
        f"- consolidation_synaptic_tag_state_budget_observed: {float(consolidation_component_metrics.get('synaptic_tag_state_budget_observed', 0.0)):.3f}",
        f"- consolidation_memory_phase_transition_integrity_observed: {float(consolidation_component_metrics.get('memory_phase_transition_integrity_observed', 0.0)):.3f}",
        f"- consolidation_memory_phase_retention_protection_observed: {float(consolidation_component_metrics.get('memory_phase_retention_protection_observed', 0.0)):.3f}",
        f"- consolidation_memory_phase_plasticity_guard_observed: {float(consolidation_component_metrics.get('memory_phase_plasticity_guard_observed', 0.0)):.3f}",
        f"- consolidation_memory_phase_overfixation_guard_observed: {float(consolidation_component_metrics.get('memory_phase_overfixation_guard_observed', 0.0)):.3f}",
        f"- consolidation_memory_phase_state_budget_observed: {float(consolidation_component_metrics.get('memory_phase_state_budget_observed', 0.0)):.3f}",
        f"- consolidation_metabolic_budget_integrity_observed: {float(consolidation_component_metrics.get('metabolic_budget_integrity_observed', 0.0)):.3f}",
        f"- consolidation_plasticity_reserve_integrity_observed: {float(consolidation_component_metrics.get('plasticity_reserve_integrity_observed', 0.0)):.3f}",
        f"- consolidation_structural_growth_bounded_observed: {float(consolidation_component_metrics.get('structural_growth_bounded_observed', 0.0)):.3f}",
        f"- consolidation_pruning_reason_trace_observed: {float(consolidation_component_metrics.get('pruning_reason_trace_observed', 0.0)):.3f}",
        f"- consolidation_resource_pressure_observed: {float(consolidation_component_metrics.get('resource_pressure_observed', 0.0)):.3f}",
        f"- consolidation_sleep_retention_observed: {float(consolidation_component_metrics.get('sleep_consolidation_retention_observed', 0.0)):.3f}",
        f"- consolidation_latent_replay_noise_resilience_observed: {float(consolidation_component_metrics.get('latent_replay_noise_resilience_observed', 0.0)):.3f}",
        f"- consolidation_sleep_memory_health_observed: {float(consolidation_component_metrics.get('sleep_consolidation_memory_health_observed', 0.0)):.3f}",
        f"- consolidation_latent_replay_counterfactual_branch_observed: {float(consolidation_component_metrics.get('latent_replay_counterfactual_branch_observed', 0.0)):.3f}",
        f"- consolidation_sleep_energy_budget_observed: {float(consolidation_component_metrics.get('sleep_consolidation_energy_budget_observed', 0.0)):.3f}",
        f"- consolidation_astro_structural_unlock_observed: {float(consolidation_component_metrics.get('astro_structural_unlock_observed', 0.0)):.3f}",
        f"- consolidation_astro_structural_lock_observed: {float(consolidation_component_metrics.get('astro_structural_lock_observed', 0.0)):.3f}",
        f"- consolidation_astro_bounded_stdp_fallback_observed: {float(consolidation_component_metrics.get('astro_bounded_stdp_fallback_observed', 0.0)):.3f}",
        f"- consolidation_world_model_replay_policy_trace_observed: {float(consolidation_component_metrics.get('world_model_replay_policy_trace_observed', 0.0)):.3f}",
        f"- consolidation_astro_policy_state_budget_observed: {float(consolidation_component_metrics.get('astro_policy_state_budget_observed', 0.0)):.3f}",
        f"- consolidation_delta_memory_phase_retention_policy_observed: {float(consolidation_component_metrics.get('delta_memory_phase_retention_policy_observed', 0.0)):.3f}",
        f"- consolidation_delta_memory_crystal_retention_observed: {float(consolidation_component_metrics.get('delta_memory_crystal_retention_observed', 0.0)):.3f}",
        f"- consolidation_delta_memory_liquid_forget_observed: {float(consolidation_component_metrics.get('delta_memory_liquid_forget_observed', 0.0)):.3f}",
        f"- consolidation_delta_memory_astro_gate_alignment_observed: {float(consolidation_component_metrics.get('delta_memory_astro_gate_alignment_observed', 0.0)):.3f}",
        f"- consolidation_delta_memory_policy_state_budget_observed: {float(consolidation_component_metrics.get('delta_memory_policy_state_budget_observed', 0.0)):.3f}",
        f"- consolidation_delta_memory_multi_history_recall_observed: {float(consolidation_component_metrics.get('delta_memory_multi_history_recall_observed', 0.0)):.3f}",
        f"- consolidation_delta_memory_multi_history_noise_resilience_observed: {float(consolidation_component_metrics.get('delta_memory_multi_history_noise_resilience_observed', 0.0)):.3f}",
        f"- consolidation_delta_memory_multi_history_health_observed: {float(consolidation_component_metrics.get('delta_memory_multi_history_health_observed', 0.0)):.3f}",
        f"- consolidation_delta_memory_multi_history_manifold_guard_observed: {float(consolidation_component_metrics.get('delta_memory_multi_history_manifold_guard_observed', 0.0)):.3f}",
        f"- consolidation_delta_memory_erase_write_decoupling_observed: {float(consolidation_component_metrics.get('delta_memory_erase_write_decoupling_observed', 0.0)):.3f}",
        f"- consolidation_delta_memory_erase_preserves_stable_memory_observed: {float(consolidation_component_metrics.get('delta_memory_erase_preserves_stable_memory_observed', 0.0)):.3f}",
        f"- consolidation_delta_memory_write_commits_residual_observed: {float(consolidation_component_metrics.get('delta_memory_write_commits_residual_observed', 0.0)):.3f}",
        f"- consolidation_readiness_trend: {consolidation_readiness_trend.get('status', 'NEW')}",
        f"- consolidation_readiness_delta: {float(consolidation_readiness_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- consolidation_replay_recovery_trend: {consolidation_replay_recovery_trend.get('status', 'NEW')}",
        f"- consolidation_replay_recovery_delta: {float(consolidation_replay_recovery_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- consolidation_reindex_trend: {consolidation_reindex_trend.get('status', 'NEW')}",
        f"- consolidation_reindex_delta: {float(consolidation_reindex_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- consolidation_health_index_trend: {consolidation_health_index_trend.get('status', 'NEW')}",
        f"- consolidation_health_index_delta: {float(consolidation_health_index_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- consolidation_replay_noise_resilience_trend: {consolidation_replay_noise_resilience_trend.get('status', 'NEW')}",
        f"- consolidation_replay_noise_resilience_delta: {float(consolidation_replay_noise_resilience_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- consolidation_astro_modulation_trend: {consolidation_astro_modulation_trend.get('status', 'NEW')}",
        f"- consolidation_astro_modulation_delta: {float(consolidation_astro_modulation_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- nested_memory_readiness_status: {_status_label(bool(nested_memory_readiness.get('passed', False)))}",
        f"- nested_memory_readiness_score: {float(nested_memory_readiness.get('score', 0.0)):.3f}",
        f"- nested_memory_observed_only: {bool(nested_memory_readiness.get('observed_only', False))}",
        f"- nested_multi_rate_update_integrity: {float(nested_memory_readiness.get('metrics', {}).get('nested_memory.multi_rate_update_integrity', 0.0)):.3f}",
        f"- nested_continuum_transfer_stability: {float(nested_memory_readiness.get('metrics', {}).get('nested_memory.continuum_memory_transfer_stability', 0.0)):.3f}",
        f"- nested_scheduler_energy_budget_integrity: {float(nested_memory_readiness.get('metrics', {}).get('nested_memory.scheduler_energy_budget_integrity', 0.0)):.3f}",
        f"- nested_interference_guard: {float(nested_memory_readiness.get('metrics', {}).get('nested_memory.catastrophic_interference_guard', 0.0)):.3f}",
        f"- nested_energy_budget_utilization: {float(nested_memory_readiness.get('metrics', {}).get('nested_memory.energy_budget_utilization', 0.0)):.3f}",
        f"- nested_memory_readiness_trend: {nested_memory_readiness_trend.get('status', 'NEW')}",
        f"- nested_memory_readiness_delta: {float(nested_memory_readiness_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- nested_multi_rate_trend: {nested_multi_rate_trend.get('status', 'NEW')}",
        f"- nested_multi_rate_delta: {float(nested_multi_rate_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- nested_energy_budget_trend: {nested_energy_budget_trend.get('status', 'NEW')}",
        f"- nested_energy_budget_delta: {float(nested_energy_budget_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- nested_interference_guard_trend: {nested_interference_guard_trend.get('status', 'NEW')}",
        f"- nested_interference_guard_delta: {float(nested_interference_guard_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- cognitive_runtime_readiness_status: {_status_label(bool(cognitive_runtime_readiness.get('passed', False)))}",
        f"- cognitive_runtime_readiness_score: {float(cognitive_runtime_readiness.get('score', 0.0)):.3f}",
        f"- common_spike_space_integrity: {float(cognitive_runtime_readiness.get('metrics', {}).get('cognitive_runtime.common_spike_space_integrity', 0.0)):.3f}",
        f"- temporal_compression_efficiency: {float(cognitive_runtime_readiness.get('metrics', {}).get('cognitive_runtime.temporal_compression_efficiency', 0.0)):.3f}",
        f"- dendritic_context_gate_stability: {float(cognitive_runtime_readiness.get('metrics', {}).get('cognitive_runtime.dendritic_context_gate_stability', 0.0)):.3f}",
        f"- reverse_reasoning_trace_integrity: {float(cognitive_runtime_readiness.get('metrics', {}).get('cognitive_runtime.reverse_reasoning_trace_integrity', 0.0)):.3f}",
        f"- causal_candidate_trace_integrity: {float(cognitive_runtime_readiness.get('metrics', {}).get('cognitive_runtime.causal_candidate_trace_integrity', 0.0)):.3f}",
        f"- module_orchestration_integrity: {float(cognitive_runtime_readiness.get('metrics', {}).get('cognitive_runtime.module_orchestration_integrity', 0.0)):.3f}",
        f"- counterfactual_lane_integrity: {float(cognitive_runtime_readiness.get('metrics', {}).get('cognitive_runtime.counterfactual_lane_integrity', 0.0)):.3f}",
        f"- action_trace_observability: {float(cognitive_runtime_readiness.get('metrics', {}).get('cognitive_runtime.action_trace_observability', 0.0)):.3f}",
        f"- runtime_trace_replay_consistency: {float(cognitive_runtime_readiness.get('metrics', {}).get('cognitive_runtime.runtime_trace_replay_consistency', 0.0)):.3f}",
        *[
            f"- cognitive_{metric_name}: {cognitive_manifold_trace_metrics[metric_name]:.3f}"
            for metric_name in COGNITIVE_MANIFOLD_TRACE_METRIC_NAMES
        ],
        *[
            f"- cognitive_{metric_name}: {cognitive_delta_memory_metrics[metric_name]:.3f}"
            for metric_name in COGNITIVE_DELTA_MEMORY_METRIC_NAMES
        ],
        f"- cognitive_linear_snn_fusion_observed_policy: excluded_from_score_and_release_gate",
        f"- cognitive_linear_snn_fusion_trend_has_previous: {bool(linear_snn_fusion_observed_trend.get('has_previous', False))}",
        f"- cognitive_linear_snn_fusion_trend_regression_count: {int(linear_snn_fusion_observed_trend.get('regression_count', 0) or 0)}",
        f"- cognitive_linear_snn_fusion_trend_release_gate_blocking: {bool(linear_snn_fusion_observed_trend.get('release_gate_blocking', False))}",
        *[
            f"- cognitive_{metric_name}: {cognitive_linear_snn_fusion_metrics[metric_name]:.3f}"
            for metric_name in COGNITIVE_LINEAR_SNN_FUSION_METRIC_NAMES
        ],
        f"- cognitive_stage_e_architecture_integration_observed_policy: excluded_from_score_and_release_gate",
        f"- cognitive_stage_e_architecture_integration_trend_has_previous: {bool(stage_e_architecture_integration_observed_trend.get('has_previous', False))}",
        f"- cognitive_stage_e_architecture_integration_trend_regression_count: {int(stage_e_architecture_integration_observed_trend.get('regression_count', 0) or 0)}",
        f"- cognitive_stage_e_architecture_integration_trend_release_gate_blocking: {bool(stage_e_architecture_integration_observed_trend.get('release_gate_blocking', False))}",
        *[
            f"- cognitive_{metric_name}: {cognitive_plastic_submodel_metrics[metric_name]:.3f}"
            for metric_name in COGNITIVE_PLASTIC_SUBMODEL_METRIC_NAMES
        ],
        f"- cognitive_runtime_readiness_trend: {cognitive_runtime_readiness_trend.get('status', 'NEW')}",
        f"- cognitive_runtime_readiness_delta: {float(cognitive_runtime_readiness_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- common_spike_space_trend: {common_spike_space_trend.get('status', 'NEW')}",
        f"- common_spike_space_delta: {float(common_spike_space_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- temporal_compression_trend: {temporal_compression_trend.get('status', 'NEW')}",
        f"- temporal_compression_delta: {float(temporal_compression_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- dendritic_context_gate_trend: {dendritic_context_gate_trend.get('status', 'NEW')}",
        f"- dendritic_context_gate_delta: {float(dendritic_context_gate_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- reverse_reasoning_trace_trend: {reverse_reasoning_trace_trend.get('status', 'NEW')}",
        f"- reverse_reasoning_trace_delta: {float(reverse_reasoning_trace_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- causal_candidate_trace_trend: {causal_candidate_trace_trend.get('status', 'NEW')}",
        f"- causal_candidate_trace_delta: {float(causal_candidate_trace_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- phase5_entry_readiness_status: {_status_label(bool(phase5_entry_readiness.get('passed', False)))}",
        f"- phase5_entry_readiness_score: {float(phase5_entry_readiness.get('score', 0.0)):.3f}",
        f"- phase5_latent_transition_alignment: {float(phase5_entry_readiness.get('metrics', {}).get('phase5_predictive_coding.latent_transition_alignment', 0.0)):.3f}",
        f"- phase5_prediction_error_observability: {float(phase5_entry_readiness.get('metrics', {}).get('phase5_predictive_coding.prediction_error_observability', 0.0)):.3f}",
        f"- phase5_correction_event_coverage: {float(phase5_entry_readiness.get('metrics', {}).get('phase5_predictive_coding.correction_event_coverage', 0.0)):.3f}",
        f"- phase5_anti_collapse_event_diversity: {float(phase5_entry_readiness.get('metrics', {}).get('phase5_predictive_coding.anti_collapse_event_diversity', 0.0)):.3f}",
        f"- phase5_counterfactual_transition_separation: {float(phase5_entry_readiness.get('metrics', {}).get('phase5_predictive_coding.counterfactual_transition_separation', 0.0)):.3f}",
        f"- phase5_multi_step_latent_chain_integrity: {float(phase5_entry_readiness.get('metrics', {}).get('phase5_predictive_coding.multi_step_latent_chain_integrity', 0.0)):.3f}",
        f"- phase5_long_horizon_error_correction_convergence: {float(phase5_entry_readiness.get('metrics', {}).get('phase5_predictive_coding.long_horizon_error_correction_convergence', 0.0)):.3f}",
        f"- phase5_horizon_bucket_stability: {float(phase5_entry_readiness.get('metrics', {}).get('phase5_predictive_coding.horizon_bucket_stability', 0.0)):.3f}",
        f"- phase5_macro_action_effectiveness: {float(phase5_entry_readiness.get('metrics', {}).get('phase5_predictive_coding.macro_action_effectiveness', 0.0)):.3f}",
        f"- phase5_subgoal_decomposition_integrity: {float(phase5_entry_readiness.get('metrics', {}).get('phase5_predictive_coding.subgoal_decomposition_integrity', 0.0)):.3f}",
        f"- phase5_depth_selective_routing_integrity: {float(phase5_entry_readiness.get('metrics', {}).get('phase5_predictive_coding.depth_selective_routing_integrity', 0.0)):.3f}",
        f"- phase5_micro_es_policy_refinement_integrity: {float(phase5_entry_readiness.get('metrics', {}).get('phase5_predictive_coding.micro_es_policy_refinement_integrity', 0.0)):.3f}",
        f"- phase5_manifold_transition_locality_observed: {float(phase5_component_metrics.get('manifold_transition_locality', 0.0)):.3f}",
        f"- phase5_manifold_rollout_stability_observed: {float(phase5_component_metrics.get('manifold_rollout_stability', 0.0)):.3f}",
        f"- phase5_causal_route_sparsity_observed: {float(phase5_component_metrics.get('causal_route_sparsity', 0.0)):.3f}",
        f"- phase5_withheld_trajectory_recall_observed: {float(phase5_component_metrics.get('withheld_trajectory_recall', 0.0)):.3f}",
        f"- phase5_manifold_trajectory_case_coverage_observed: {float(phase5_component_metrics.get('manifold_trajectory_case_coverage', 0.0)):.3f}",
        f"- phase5_manifold_average_case_recall_observed: {float(phase5_component_metrics.get('manifold_average_case_recall', 0.0)):.3f}",
        f"- phase5_manifold_scan_budget_integrity_observed: {float(phase5_component_metrics.get('manifold_scan_budget_integrity', 0.0)):.3f}",
        f"- phase5_manifold_indexed_candidate_integrity_observed: {float(phase5_component_metrics.get('manifold_indexed_candidate_integrity', 0.0)):.3f}",
        f"- phase5_manifold_index_scan_reduction_observed: {float(phase5_component_metrics.get('manifold_index_scan_reduction', 0.0)):.3f}",
        f"- phase5_manifold_candidate_miss_guard_observed: {float(phase5_component_metrics.get('manifold_candidate_miss_guard', 0.0)):.3f}",
        f"- phase5_entry_readiness_trend: {phase5_entry_readiness_trend.get('status', 'NEW')}",
        f"- phase5_entry_readiness_delta: {float(phase5_entry_readiness_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- phase5_latent_transition_trend: {phase5_latent_transition_trend.get('status', 'NEW')}",
        f"- phase5_latent_transition_delta: {float(phase5_latent_transition_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- phase5_correction_event_trend: {phase5_correction_event_trend.get('status', 'NEW')}",
        f"- phase5_correction_event_delta: {float(phase5_correction_event_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- phase5_counterfactual_transition_trend: {phase5_counterfactual_transition_trend.get('status', 'NEW')}",
        f"- phase5_counterfactual_transition_delta: {float(phase5_counterfactual_transition_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- phase5_multistep_chain_trend: {phase5_multistep_chain_trend.get('status', 'NEW')}",
        f"- phase5_multistep_chain_delta: {float(phase5_multistep_chain_trend.get('delta', 0.0) or 0.0):+.3f}",
        f"- phase5_error_convergence_trend: {phase5_error_convergence_trend.get('status', 'NEW')}",
        f"- phase5_error_convergence_delta: {float(phase5_error_convergence_trend.get('delta', 0.0) or 0.0):+.3f}",
    ]

    if agent_dialogue_results and isinstance(agent_dialogue_results[-1], dict):
        shift_detail = agent_dialogue_results[-1]
        lines.extend(
            [
                "",
                "Dialogue Shift Detail",
                f"- shift_from: {shift_detail.get('shift_from', '')}",
                f"- shift_query: {shift_detail.get('user_input', '')}",
                f"- shift_following_score: {float(shift_detail.get('shift_following_score', 0.0) or 0.0):.3f}",
            ]
        )

    lines.extend(
        [
            "",
            "Components",
        ]
    )
    for failure in (
        stage_d_readiness.get("delta_memory_candidate_failures", [])
        if isinstance(stage_d_readiness.get("delta_memory_candidate_failures"), list)
        else []
    ):
        if not isinstance(failure, dict):
            continue
        lines.append(
            "- stage_d_delta_memory_candidate_failure: "
            f"{failure.get('check', '')} value={float(failure.get('value', 0.0) or 0.0):.3f} "
            f"required>={float(failure.get('threshold', 1.0) or 1.0):.3f} "
            f"description={failure.get('description', '')}"
        )
    for failure in (
        stage_d_readiness.get("acceptance_candidate_failures", [])
        if isinstance(stage_d_readiness.get("acceptance_candidate_failures"), list)
        else []
    )[:5]:
        if not isinstance(failure, dict):
            continue
        lines.append(
            "- stage_d_acceptance_candidate_failure: "
            f"{failure.get('check', '')} value={float(failure.get('value', 0.0) or 0.0):.3f} "
            f"required>={float(failure.get('threshold', 1.0) or 1.0):.3f} "
            f"description={failure.get('description', '')}"
        )
    for failure in (
        stage_e_readiness.get("observed_acceptance_candidate_failures", [])
        if isinstance(stage_e_readiness.get("observed_acceptance_candidate_failures"), list)
        else []
    )[:5]:
        if not isinstance(failure, dict):
            continue
        lines.append(
            "- stage_e_observed_acceptance_candidate_failure: "
            f"{failure.get('check', '')} value={float(failure.get('value', 0.0) or 0.0):.3f} "
            f"required>={float(failure.get('threshold', 1.0) or 1.0):.3f} "
            f"description={failure.get('description', '')}"
        )

    for component_name, component_report in sorted(component_reports.items()):
        if not isinstance(component_report, dict):
            continue
        lines.append(
            f"- {component_name}: {_status_label(bool(component_report.get('passed', False)))} "
            f"score={float(component_report.get('overall_score', 0.0)):.3f}"
        )

    return "\n".join(lines) + "\n"


def run_phase3_accuracy_suite(
    history_path: Optional[str] = None,
    persist_history: bool = False,
    history_limit: int = 50,
    stage_b_promotion_required_streak: int = 3,
    regression_tolerance: float = DEFAULT_PHASE3_TREND_TOLERANCE,
) -> Dict[str, Any]:
    if int(stage_b_promotion_required_streak) < 1:
        raise ValueError("stage_b_promotion_required_streak must be >= 1.")
    benchmarks: Dict[str, Callable[[], Dict[str, Any]]] = {
        "agent_dialogue": run_agent_dialogue_benchmark,
        "sara_inference": run_inference_accuracy_benchmark,
        "spiking_llm": run_spiking_llm_accuracy_benchmark,
        "task_switch_adaptation": run_task_switch_adaptation_benchmark,
        "future_state_consistency": run_future_state_consistency_benchmark,
        "energy_efficiency": run_energy_efficiency_benchmark,
        "parameter_efficiency": run_parameter_efficiency_benchmark,
        "continual_consolidation": run_continual_consolidation_benchmark,
        "nested_memory": run_nested_memory_readiness_benchmark,
        "cognitive_runtime": run_cognitive_runtime_benchmark,
        "phase5_predictive_coding": run_phase5_predictive_coding_benchmark,
    }
    reports = {name: benchmark() for name, benchmark in benchmarks.items()}
    overall_score = sum(report["overall_score"] for report in reports.values()) / max(len(reports), 1)
    passed = all(bool(report.get("passed", False)) for report in reports.values())

    previous_report = latest_phase3_report(history_path) if history_path else None
    previous_history = load_phase3_history(history_path) if history_path else []
    if not isinstance(previous_history, list):
        previous_history = []
    focus_summary = _build_focus_summary(reports)
    focus_trend = _build_focus_trend(focus_summary, previous_report)
    current_observed_trend_report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": overall_score,
        "component_reports": reports,
        "focus_summary": focus_summary,
        "focus_trend": focus_trend,
        "passed": passed,
    }
    linear_snn_fusion_observed_trend = build_cognitive_linear_snn_fusion_observed_trend(
        current_report=current_observed_trend_report,
        previous_report=previous_report,
        regression_tolerance=float(max(regression_tolerance, 0.0)),
    )
    stage_e_architecture_integration_observed_trend = (
        build_cognitive_stage_e_architecture_integration_observed_trend(
            current_report=current_observed_trend_report,
            previous_report=previous_report,
            regression_tolerance=float(max(regression_tolerance, 0.0)),
        )
    )
    report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": overall_score,
        "component_reports": reports,
        "focus_summary": focus_summary,
        "focus_trend": focus_trend,
        "linear_snn_fusion_observed_trend": linear_snn_fusion_observed_trend,
        "stage_e_architecture_integration_observed_trend": stage_e_architecture_integration_observed_trend,
        "passed": passed,
        "trend": build_phase3_trend(
            current_report=current_observed_trend_report,
            previous_report=previous_report,
            regression_tolerance=float(max(regression_tolerance, 0.0)),
        ),
    }
    report["stage_a_acceptance"] = _build_stage_a_acceptance(report)
    report["stage_b_readiness"] = _build_stage_b_readiness(report)
    report["stage_b_readiness"]["promotion_readiness"] = _build_stage_b_promotion_readiness(
        report["stage_b_readiness"],
        history=previous_history,
        required_streak=int(stage_b_promotion_required_streak),
    )
    report["stage_b_readiness"]["rlm_observation_promotion_readiness"] = (
        _build_stage_b_rlm_observation_promotion_readiness(
            report["stage_b_readiness"],
            history=previous_history,
            required_streak=int(stage_b_promotion_required_streak),
        )
    )
    report["stage_c_readiness"] = _build_stage_c_readiness(report)
    report["stage_d_readiness"] = _build_stage_d_readiness(report)
    report["stage_d_readiness"]["delta_memory_promotion_readiness"] = (
        _build_stage_d_delta_memory_promotion_readiness(
            report["stage_d_readiness"],
            history=previous_history,
            required_streak=int(stage_b_promotion_required_streak),
        )
    )
    report["stage_d_readiness"]["acceptance_candidate_stability"] = (
        _build_stage_d_acceptance_candidate_stability(
            report["stage_d_readiness"],
            history=previous_history,
            required_streak=int(stage_b_promotion_required_streak),
        )
    )
    report["stage_e_readiness"] = _build_stage_e_readiness(report)
    report["stage_e_readiness"]["observed_acceptance_candidate_stability"] = (
        _build_stage_e_acceptance_candidate_stability(
            report["stage_e_readiness"],
            history=previous_history,
            required_streak=int(stage_b_promotion_required_streak),
        )
    )
    report["phase3_completion"] = _build_phase3_completion(report)
    if previous_report is not None:
        report["previous_overall_score"] = previous_report.get("overall_score")

    if history_path and persist_history:
        history = append_phase3_history(
            history_path=history_path,
            report=report,
            max_entries=history_limit,
        )
        report["history_length"] = len(history)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the aggregated Phase 3 accuracy suite.")
    parser.add_argument(
        "--report-path",
        default=workspace_path("evaluation", "phase3_accuracy_suite.json"),
        help="Managed output path for the aggregated report.",
    )
    parser.add_argument(
        "--summary-path",
        default=workspace_path("evaluation", "phase3_accuracy_summary.txt"),
        help="Managed output path for the human-readable accuracy summary.",
    )
    parser.add_argument(
        "--history-path",
        default=workspace_path("evaluation", "phase3_accuracy_history.json"),
        help="Managed output path for suite history snapshots.",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=50,
        help="Maximum number of suite snapshots to keep in history.",
    )
    parser.add_argument(
        "--stage-b-promotion-required-streak",
        type=int,
        default=3,
        help="Consecutive pass count required before Stage B promotion is recommended.",
    )
    parser.add_argument(
        "--regression-tolerance",
        type=float,
        default=DEFAULT_PHASE3_TREND_TOLERANCE,
        help="Tolerance used by trend regression detection (higher is less sensitive).",
    )
    args = parser.parse_args()

    report = run_phase3_accuracy_suite(
        history_path=args.history_path,
        persist_history=True,
        history_limit=args.history_limit,
        stage_b_promotion_required_streak=args.stage_b_promotion_required_streak,
        regression_tolerance=args.regression_tolerance,
    )
    report_path = ensure_parent_directory(args.report_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    summary_path = ensure_parent_directory(args.summary_path)
    summary_text = format_phase3_accuracy_summary(report)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(summary_text)

    print("Phase 3 accuracy suite completed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {report_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
