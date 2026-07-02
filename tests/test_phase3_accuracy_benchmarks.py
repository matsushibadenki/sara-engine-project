import importlib.util
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sara_engine.evaluation.phase3_tracking import (
    COGNITIVE_DELTA_MEMORY_METRIC_NAMES,
    COGNITIVE_LINEAR_SNN_FUSION_METRIC_NAMES,
    COGNITIVE_MANIFOLD_TRACE_METRIC_NAMES,
    COGNITIVE_STAGE_E_ARCHITECTURE_INTEGRATION_METRIC_NAMES,
    append_phase3_history,
    build_cognitive_linear_snn_fusion_observed_trend,
    build_cognitive_stage_e_architecture_integration_observed_trend,
    build_phase3_trend,
    compact_neuromorphic_profile_trend,
    extract_cognitive_delta_memory_metrics,
    extract_cognitive_linear_snn_fusion_metrics,
    extract_cognitive_manifold_trace_metrics,
    extract_cognitive_stage_e_architecture_integration_metrics,
    load_phase3_history,
    phase3_component_metrics,
)
from sara_engine.evaluation.stage_e_contract import STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_METRIC_NAMES
from sara_engine.utils.project_paths import workspace_path


def _load_script(script_name: str):
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", script_name)
    )
    spec = importlib.util.spec_from_file_location(f"{script_name}_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inference_accuracy_benchmark_returns_expected_metrics():
    module = _load_script("inference_accuracy_benchmark.py")
    report = module.run_inference_accuracy_benchmark()

    assert report["evaluator_name"] == "InferenceSequenceEvaluator"
    assert report["passed"] is True
    assert report["metrics"]["one_shot_accuracy"] == 1.0
    assert report["metrics"]["few_shot_accuracy"] == 1.0
    assert report["metrics"]["fuzzy_retrieval_accuracy"] == 1.0
    assert report["metrics"]["noise_robustness"] == 1.0
    assert report["metrics"]["continual_retention"] == 1.0
    assert report["metrics"]["long_horizon_retention"] == 1.0


def test_spiking_llm_accuracy_benchmark_returns_expected_metrics():
    module = _load_script("spiking_llm_accuracy_benchmark.py")
    report = module.run_spiking_llm_accuracy_benchmark()

    assert report["evaluator_name"] == "SpikingLLMSequenceEvaluator"
    assert report["passed"] is True
    assert report["metrics"]["next_token_accuracy"] == 1.0
    assert report["metrics"]["few_shot_context_accuracy"] == 1.0
    assert report["metrics"]["stream_completion_rate"] == 1.0
    assert report["metrics"]["hierarchical_context_integrity"] == 1.0
    assert report["metrics"]["noise_robust_context_accuracy"] == 1.0
    assert report["metrics"]["continual_memory_retention"] == 1.0
    assert report["metrics"]["long_horizon_memory_retention"] == 1.0


def test_phase3_accuracy_suite_aggregates_component_reports():
    module = _load_script("phase3_accuracy_suite.py")
    report = module.run_phase3_accuracy_suite()

    assert report["suite_name"] == "Phase3AccuracySuite"
    assert report["passed"] is True
    assert "stage_b_readiness" in report
    assert report["stage_b_readiness"]["passed"] is True
    assert report["stage_b_readiness"]["minimum_failure_count"] == 0
    assert report["stage_b_readiness"]["promotion_candidate_ready"] is True
    assert report["stage_b_readiness"]["promotion_candidate_failure_count"] == 0
    assert report["stage_b_readiness"]["promotion_candidate_promoted"] is True
    assert report["stage_b_readiness"]["promotion_readiness"]["consecutive_passes"] == 0
    assert report["stage_b_readiness"]["promotion_readiness"]["required_streak"] == 3
    assert report["stage_b_readiness"]["promotion_readiness"]["recommended"] is False
    assert report["stage_b_readiness"]["rlm_observation_candidate_ready"] is True
    assert report["stage_b_readiness"]["rlm_observation_candidate_failure_count"] == 0
    assert report["stage_b_readiness"]["rlm_observation_candidate_promoted"] is True
    assert report["stage_b_readiness"]["rlm_observation_promotion_readiness"]["consecutive_passes"] == 0
    assert report["stage_b_readiness"]["rlm_observation_promotion_readiness"]["required_streak"] == 3
    assert report["stage_b_readiness"]["rlm_observation_promotion_readiness"]["recommended"] is False
    assert report["stage_b_readiness"]["minimum_checks"]["metric.future_state_focused_retrieval_hit_ratio"] is True
    assert report["stage_b_readiness"]["minimum_checks"]["metric.future_state_branch_level_decision_consistency"] is True
    assert "stage_c_readiness" in report
    assert report["stage_c_readiness"]["passed"] is True
    assert report["stage_c_readiness"]["minimum_failure_count"] == 0
    assert "stage_d_readiness" in report
    assert report["stage_d_readiness"]["passed"] is True
    assert report["stage_d_readiness"]["minimum_failure_count"] == 0
    assert report["stage_d_readiness"]["delta_memory_candidate_ready"] is True
    assert report["stage_d_readiness"]["delta_memory_candidate_failure_count"] == 0
    assert report["stage_d_readiness"]["delta_memory_candidate_promoted"] is False
    assert report["stage_d_readiness"]["delta_memory_promotion_readiness"]["consecutive_passes"] == 1
    assert report["stage_d_readiness"]["delta_memory_promotion_readiness"]["required_streak"] == 3
    assert report["stage_d_readiness"]["delta_memory_promotion_readiness"]["recommended"] is False
    assert report["stage_d_readiness"]["acceptance_candidate_count"] >= 4
    assert (
        report["stage_d_readiness"]["acceptance_candidate_ready_count"]
        == report["stage_d_readiness"]["acceptance_candidate_count"]
    )
    assert report["stage_d_readiness"]["acceptance_candidates_ready"] is True
    assert report["stage_d_readiness"]["acceptance_candidate_failure_count"] == 0
    assert report["stage_d_readiness"]["acceptance_candidate_stability"]["consecutive_passes"] == 1
    assert report["stage_d_readiness"]["acceptance_candidate_stability"]["required_streak"] == 3
    assert report["stage_d_readiness"]["acceptance_candidate_stability"]["recommended"] is False
    assert any(
        item["metric"] == "delta_memory_multi_history_recall_observed"
        for item in report["stage_d_readiness"]["acceptance_candidates"]
    )
    assert "stage_e_readiness" in report
    assert report["stage_e_readiness"]["passed"] is True
    assert report["stage_e_readiness"]["minimum_failure_count"] == 0
    assert report["stage_e_readiness"]["observed_acceptance_candidate_count"] == len(
        STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_METRIC_NAMES
    )
    assert (
        report["stage_e_readiness"]["observed_acceptance_candidate_ready_count"]
        == report["stage_e_readiness"]["observed_acceptance_candidate_count"]
    )
    assert report["stage_e_readiness"]["observed_acceptance_candidates_ready"] is True
    assert report["stage_e_readiness"]["observed_acceptance_candidate_failure_count"] == 0
    assert report["stage_e_readiness"]["observed_acceptance_candidate_stability"]["consecutive_passes"] == 1
    assert report["stage_e_readiness"]["observed_acceptance_candidate_stability"]["required_streak"] == 3
    assert report["stage_e_readiness"]["observed_acceptance_candidate_stability"]["recommended"] is False
    assert any(
        item["metric"] == "micro_turn_event_budget_observed"
        for item in report["stage_e_readiness"]["observed_acceptance_candidates"]
    )
    assert "agent_dialogue" in report["component_reports"]
    assert "sara_inference" in report["component_reports"]
    assert "spiking_llm" in report["component_reports"]
    assert "task_switch_adaptation" in report["component_reports"]
    assert "future_state_consistency" in report["component_reports"]
    assert "energy_efficiency" in report["component_reports"]
    assert "parameter_efficiency" in report["component_reports"]
    assert "continual_consolidation" in report["component_reports"]
    assert "nested_memory" in report["component_reports"]
    assert "cognitive_runtime" in report["component_reports"]
    assert "phase5_predictive_coding" in report["component_reports"]
    assert "focus_summary" in report
    assert "stage_a_acceptance" in report
    assert report["stage_a_acceptance"]["passed"] is True
    assert report["focus_summary"]["few_shot"]["score"] == 1.0
    assert report["focus_summary"]["few_shot"]["metrics"]["spiking_llm.hierarchical_context_integrity"] == 1.0
    assert report["focus_summary"]["continual"]["score"] == 1.0
    assert "retrieval_hygiene" in report["focus_summary"]
    assert "adaptive_readiness" in report["focus_summary"]
    assert "predictive_readiness" in report["focus_summary"]
    assert "efficiency_readiness" in report["focus_summary"]
    assert "parameter_efficiency" in report["focus_summary"]
    assert "consolidation_readiness" in report["focus_summary"]
    assert "nested_memory_readiness" in report["focus_summary"]
    assert "cognitive_runtime_readiness" in report["focus_summary"]
    assert (
        report["focus_summary"]["cognitive_runtime_readiness"]["observed_metric_policy"][
            "linear_snn_fusion_metrics_excluded_from_release_gate"
        ]
        is True
    )
    assert (
        report["focus_summary"]["cognitive_runtime_readiness"]["observed_metrics"][
            "cognitive_runtime.predictive_spike_entropy_reduction_observed"
        ]
        == 1.0
    )
    assert (
        report["focus_summary"]["cognitive_runtime_readiness"]["plastic_submodel_observed_metrics"][
            "cognitive_runtime.plastic_submodel_registry_integrity_observed"
        ]
        == 1.0
    )
    assert "phase5_entry_readiness" in report["focus_summary"]
    assert 0.0 <= report["focus_summary"]["retrieval_hygiene"]["score"] <= 1.0
    assert report["focus_summary"]["adaptive_readiness"]["score"] == 1.0
    assert (
        report["focus_summary"]["adaptive_readiness"]["metrics"][
            "task_switch_adaptation.meta_adaptation_parameter_integrity"
        ]
        == 1.0
    )
    assert (
        report["focus_summary"]["adaptive_readiness"]["metrics"][
            "task_switch_adaptation.temporal_self_distillation_stability"
        ]
        == 1.0
    )
    assert report["focus_summary"]["predictive_readiness"]["score"] == 1.0
    assert report["focus_summary"]["predictive_readiness"]["metrics"]["future_state_consistency.future_state_simulation_integrity"] == 1.0
    assert report["focus_summary"]["predictive_readiness"]["metrics"]["future_state_consistency.future_state_simulation_usefulness"] == 1.0
    assert report["focus_summary"]["efficiency_readiness"]["score"] >= 0.8
    assert "focus_trend" in report
    assert "retrieval_hygiene" in report["focus_trend"]
    assert "adaptive_readiness" in report["focus_trend"]
    assert "predictive_readiness" in report["focus_trend"]
    assert "efficiency_readiness" in report["focus_trend"]
    assert "parameter_efficiency" in report["focus_trend"]
    assert "consolidation_readiness" in report["focus_trend"]
    assert "nested_memory_readiness" in report["focus_trend"]
    assert "cognitive_runtime_readiness" in report["focus_trend"]
    assert "phase5_entry_readiness" in report["focus_trend"]
    assert "trend" in report
    assert report["linear_snn_fusion_observed_trend"]["observed_only"] is True
    assert report["linear_snn_fusion_observed_trend"]["release_gate_blocking"] is False
    assert report["linear_snn_fusion_observed_trend"]["regression_count"] == 0
    assert report["stage_e_architecture_integration_observed_trend"]["observed_only"] is True
    assert report["stage_e_architecture_integration_observed_trend"]["release_gate_blocking"] is False
    assert report["stage_e_architecture_integration_observed_trend"]["regression_count"] == 0
    assert "phase3_completion" in report
    assert report["phase3_completion"]["passed"] is True
    assert report["phase3_completion"]["completion_score"] == 1.0
    assert report["focus_summary"]["phase5_entry_readiness"]["score"] == 1.0
    assert report["focus_summary"]["phase5_entry_readiness"]["metrics"][
        "phase5_predictive_coding.correction_event_coverage"
    ] == 1.0
    assert report["focus_summary"]["phase5_entry_readiness"]["metrics"][
        "phase5_predictive_coding.horizon_bucket_stability"
    ] == 1.0
    assert report["focus_summary"]["phase5_entry_readiness"]["metrics"][
        "phase5_predictive_coding.macro_action_effectiveness"
    ] == 1.0
    assert report["focus_summary"]["phase5_entry_readiness"]["metrics"][
        "phase5_predictive_coding.subgoal_decomposition_integrity"
    ] == 1.0
    assert report["focus_summary"]["phase5_entry_readiness"]["metrics"][
        "phase5_predictive_coding.depth_selective_routing_integrity"
    ] == 1.0
    assert report["focus_summary"]["nested_memory_readiness"]["passed"] is True
    assert report["focus_summary"]["nested_memory_readiness"]["observed_only"] is True
    assert report["focus_summary"]["nested_memory_readiness"]["metrics"][
        "nested_memory.multi_rate_update_integrity"
    ] == 1.0


def test_task_switch_adaptation_benchmark_returns_expected_metrics():
    module = _load_script("task_switch_adaptation_benchmark.py")
    report = module.run_task_switch_adaptation_benchmark()

    assert report["evaluator_name"] == "TaskSwitchAdaptationBenchmark"
    assert report["passed"] is True
    assert report["metrics"]["task_switch_adaptation"] == 1.0
    assert report["metrics"]["session_memory_switch_grounding"] == 1.0
    assert report["metrics"]["meta_adaptation_loop"] == 1.0
    assert report["metrics"]["meta_adaptation_parameter_integrity"] == 1.0
    assert report["metrics"]["temporal_self_distillation_stability"] == 1.0
    meta_case = next(
        case
        for case in report["details"]["test_results"]
        if isinstance(case, dict) and "adaptation_state" in case
    )
    assert meta_case["adaptation_state"]["response_mode"] == "directive"
    assert "Do this now:" in meta_case["second_response"]


def test_continual_consolidation_benchmark_returns_expected_metrics():
    module = _load_script("continual_consolidation_benchmark.py")
    report = module.run_continual_consolidation_benchmark()

    assert report["evaluator_name"] == "ContinualConsolidationBenchmark"
    assert report["passed"] is True
    assert report["metrics"]["replay_recovery_integrity"] == 1.0
    assert report["metrics"]["long_horizon_consolidation_retention"] == 1.0
    assert report["metrics"]["counterfactual_replay_selection_integrity"] == 1.0
    assert report["metrics"]["replay_upgrade_reindex_integrity"] == 1.0
    assert report["metrics"]["memory_health_index_integrity"] == 1.0
    assert report["metrics"]["replay_noise_resilience_integrity"] == 1.0
    assert report["metrics"]["astro_modulation_stability"] == 1.0
    assert report["metrics"]["delta_memory_residual_write_integrity_observed"] == 1.0
    assert report["metrics"]["delta_memory_retention_gate_stability_observed"] == 1.0
    assert report["metrics"]["delta_memory_context_recall_without_text_reinjection_observed"] == 1.0
    assert report["metrics"]["delta_memory_state_budget_integrity_observed"] == 1.0
    assert report["metrics"]["delta_memory_interference_guard_observed"] == 1.0
    assert report["metrics"]["manifold_continual_retention_observed"] == 1.0
    assert report["metrics"]["manifold_trajectory_case_coverage_observed"] == 1.0
    assert report["metrics"]["manifold_average_case_recall_observed"] == 1.0
    assert report["metrics"]["manifold_scan_budget_integrity_observed"] == 1.0
    assert report["metrics"]["manifold_indexed_candidate_integrity_observed"] == 1.0
    assert report["metrics"]["manifold_index_scan_reduction_observed"] == 1.0
    assert report["metrics"]["manifold_capacity_pressure_recall_observed"] == 1.0
    assert report["metrics"]["manifold_capacity_pressure_scan_reduction_observed"] > 0.80
    assert report["metrics"]["manifold_replay_refresh_retention_observed"] == 1.0
    assert report["metrics"]["manifold_replay_refresh_eviction_integrity_observed"] == 1.0
    assert report["metrics"]["synaptic_tag_integrity_observed"] == 1.0
    assert report["metrics"]["synaptic_tag_importance_score_observed"] == 1.0
    assert report["metrics"]["synaptic_tag_replay_priority_observed"] == 1.0
    assert report["metrics"]["synaptic_tag_pruning_candidate_observed"] == 1.0
    assert report["metrics"]["synaptic_tag_state_budget_observed"] == 1.0
    assert report["metrics"]["memory_phase_transition_integrity_observed"] == 1.0
    assert report["metrics"]["memory_phase_retention_protection_observed"] == 1.0
    assert report["metrics"]["memory_phase_plasticity_guard_observed"] == 1.0
    assert report["metrics"]["memory_phase_overfixation_guard_observed"] == 1.0
    assert report["metrics"]["memory_phase_state_budget_observed"] == 1.0
    assert report["metrics"]["metabolic_budget_integrity_observed"] == 1.0
    assert report["metrics"]["plasticity_reserve_integrity_observed"] == 1.0
    assert report["metrics"]["structural_growth_bounded_observed"] == 1.0
    assert report["metrics"]["pruning_reason_trace_observed"] == 1.0
    assert report["metrics"]["resource_pressure_observed"] == 1.0
    assert report["metrics"]["sleep_consolidation_retention_observed"] == 1.0
    assert report["metrics"]["latent_replay_noise_resilience_observed"] == 1.0
    assert report["metrics"]["sleep_consolidation_memory_health_observed"] == 1.0
    assert report["metrics"]["latent_replay_counterfactual_branch_observed"] == 1.0
    assert report["metrics"]["sleep_consolidation_energy_budget_observed"] == 1.0
    assert report["metrics"]["astro_structural_unlock_observed"] == 1.0
    assert report["metrics"]["astro_structural_lock_observed"] == 1.0
    assert report["metrics"]["astro_bounded_stdp_fallback_observed"] == 1.0
    assert report["metrics"]["world_model_replay_policy_trace_observed"] == 1.0
    assert report["metrics"]["astro_policy_state_budget_observed"] == 1.0
    assert report["metrics"]["delta_memory_phase_retention_policy_observed"] == 1.0
    assert report["metrics"]["delta_memory_crystal_retention_observed"] == 1.0
    assert report["metrics"]["delta_memory_liquid_forget_observed"] == 1.0
    assert report["metrics"]["delta_memory_astro_gate_alignment_observed"] == 1.0
    assert report["metrics"]["delta_memory_policy_state_budget_observed"] == 1.0
    assert report["metrics"]["delta_memory_multi_history_recall_observed"] == 1.0
    assert report["metrics"]["delta_memory_multi_history_noise_resilience_observed"] == 1.0
    assert report["metrics"]["delta_memory_multi_history_health_observed"] == 1.0
    assert report["metrics"]["delta_memory_multi_history_manifold_guard_observed"] == 1.0
    assert report["metrics"]["delta_memory_erase_write_decoupling_observed"] == 1.0
    assert report["metrics"]["delta_memory_erase_preserves_stable_memory_observed"] == 1.0
    assert report["metrics"]["delta_memory_write_commits_residual_observed"] == 1.0
    assert report["metrics"]["idle_maintenance_trace_integrity_observed"] == 1.0
    assert report["metrics"]["idle_maintenance_phase_alignment_observed"] == 1.0
    assert report["metrics"]["idle_maintenance_cache_refresh_observed"] == 1.0
    assert report["metrics"]["idle_maintenance_multimodal_bundle_visibility_observed"] == 1.0
    delta_case = next(
        case
        for case in report["details"]["test_results"]
        if isinstance(case, dict) and "delta_memory_report" in case
    )
    assert delta_case["delta_memory_report"]["observed_only"] is True
    assert delta_case["delta_memory_report"]["traces"]["recall"]["predicted_ids"] == [11]
    capacity_case = report["details"]["test_results"][-9]
    assert capacity_case["trajectory_count"] == 9
    assert capacity_case["manifold_report"]["trajectory_top_match_ratio"] == 1.0
    refresh_case = report["details"]["test_results"][-8]
    assert refresh_case["trajectory_ids"] == [
        "refresh-anchor-path",
        "refresh-distractor-c",
        "refresh-distractor-d",
    ]
    synaptic_case = report["details"]["test_results"][-7]
    assert synaptic_case["synaptic_tag_report"]["observed_only"] is True
    assert synaptic_case["top_synaptic_tag"]["tag"] == "consolidate"
    phase_case = report["details"]["test_results"][-6]
    assert phase_case["memory_phase_report"]["observed_only"] is True
    assert phase_case["anchor_phase_path"] == ["liquid", "glass", "crystal"]
    budget_case = report["details"]["test_results"][-5]
    assert budget_case["metabolic_budget_report"]["observed_only"] is True
    assert budget_case["rejected_reason_count"] >= 1
    sleep_case = report["details"]["test_results"][-4]
    assert sleep_case["sleep_consolidation_report"]["observed_only"] is True
    assert sleep_case["sleep_consolidation_report"]["event_budget_ok"] is True
    astro_gate_case = report["details"]["test_results"][-3]
    assert astro_gate_case["astro_structural_gate_report"]["observed_only"] is True
    assert astro_gate_case["astro_structural_gate_report"]["final_structural_unlocked"] is False
    delta_policy_case = report["details"]["test_results"][-2]
    assert delta_policy_case["delta_retention_policy_report"]["observed_only"] is True
    assert delta_policy_case["delta_retention_policy_stress_report"]["observed_only"] is True
    assert delta_policy_case["delta_erase_write_decoupling_report"]["observed_only"] is True
    assert (
        delta_policy_case["delta_retention_policy_report"]["metrics"][
            "delta_memory_crystal_retention_observed"
        ]
        == 1.0
    )
    assert (
        delta_policy_case["delta_erase_write_decoupling_report"]["metrics"][
            "delta_memory_erase_write_decoupling_observed"
        ]
        == 1.0
    )
    assert (
        delta_policy_case["delta_retention_policy_stress_report"]["metrics"][
            "delta_memory_multi_history_recall_observed"
        ]
        == 1.0
    )
    idle_maintenance_case = report["details"]["test_results"][-1]
    assert idle_maintenance_case["idle_consolidation_loop_report"]["sleep_consolidation_report"]["observed_only"] is True
    assert idle_maintenance_case["idle_consolidation_loop_report"]["memory_phase_report"]["observed_only"] is True
    assert idle_maintenance_case["idle_consolidation_loop_report"]["delta_retention_policy_report"]["observed_only"] is True
    assert "multimodal_bundle_summary" in idle_maintenance_case["idle_consolidation_loop_report"]
    assert idle_maintenance_case["selected_count"] >= 1
    assert idle_maintenance_case["refresh_count"] >= 1
    assert refresh_case["anchor_refresh_count"] == 2
    assert refresh_case["manifold_report"]["trajectory_top_match_ratio"] == 1.0


def test_nested_memory_readiness_benchmark_returns_expected_metrics():
    module = _load_script("nested_memory_readiness_benchmark.py")
    report = module.run_nested_memory_readiness_benchmark()

    assert report["evaluator_name"] == "NestedMemoryReadinessBenchmark"
    assert report["passed"] is True
    assert report["metrics"]["multi_rate_update_integrity"] == 1.0
    assert report["metrics"]["continuum_memory_transfer_stability"] == 1.0
    assert report["metrics"]["scheduler_energy_budget_integrity"] == 1.0
    assert report["metrics"]["catastrophic_interference_guard"] == 1.0


def test_cognitive_runtime_benchmark_returns_expected_metrics():
    module = _load_script("cognitive_runtime_benchmark.py")
    report = module.run_cognitive_runtime_benchmark()

    assert report["suite_name"] == "CognitiveRuntimeBenchmark"
    assert report["passed"] is True
    assert report["metric_policy"]["gate_score_source"] == "gate_metrics"
    assert report["metric_policy"]["observed_metrics_excluded_from_overall_score"] is True
    assert report["metric_policy"]["observed_metrics_excluded_from_release_gate"] is True
    assert "predictive_spike_entropy_reduction_observed" not in report["gate_metrics"]
    assert report["observed_metrics"]["predictive_spike_entropy_reduction_observed"] == 1.0
    assert report["metrics"]["common_spike_space_integrity"] == 1.0
    assert report["metrics"]["temporal_compression_efficiency"] == 1.0
    assert report["metrics"]["modality_temporal_budget_integrity"] == 1.0
    assert report["metrics"]["dendritic_context_gate_stability"] == 1.0
    assert report["metrics"]["spiking_hjepa_latent_transition"] == 1.0
    assert report["metrics"]["reverse_reasoning_trace_integrity"] == 1.0
    assert report["metrics"]["causal_candidate_trace_integrity"] == 1.0
    assert report["metrics"]["module_orchestration_integrity"] == 1.0
    assert report["metrics"]["counterfactual_lane_integrity"] == 1.0
    assert report["metrics"]["action_trace_observability"] == 1.0
    assert report["metrics"]["runtime_trace_replay_consistency"] == 1.0
    assert report["metrics"]["manifold_trace_support_observed"] == 1.0
    assert report["metrics"]["manifold_trace_recall_observed"] == 1.0
    assert report["metrics"]["manifold_trace_scan_budget_observed"] == 1.0
    assert report["metrics"]["manifold_trace_index_scan_reduction_observed"] == 1.0
    assert report["metrics"]["manifold_trace_candidate_guard_observed"] == 1.0
    assert report["metrics"]["delta_memory_steering_integrity_observed"] == 1.0
    assert report["metrics"]["delta_memory_counterfactual_isolation_observed"] == 1.0
    assert report["metrics"]["delta_memory_trace_observability_observed"] == 1.0
    assert report["metrics"]["predictive_spike_entropy_reduction_observed"] == 1.0
    assert report["metrics"]["phase_binding_coincidence_integrity_observed"] == 1.0
    assert report["metrics"]["forward_only_local_update_stability_observed"] == 1.0
    assert report["metrics"]["lejepa_linear_identifiability_proxy_observed"] == 1.0
    assert report["metrics"]["lejepa_latent_whitening_health_observed"] == 1.0
    assert report["metrics"]["lejepa_factor_disentanglement_observed"] == 1.0
    assert report["metrics"]["lejepa_latent_planning_consistency_observed"] == 1.0
    assert report["metrics"]["lejepa_positive_pair_alignment_observed"] == 1.0
    assert report["metrics"]["plastic_submodel_registry_integrity_observed"] == 1.0
    assert report["metrics"]["dynamic_submodel_route_integrity_observed"] == 1.0
    assert report["metrics"]["submodel_relearning_trace_integrity_observed"] == 1.0
    assert report["metrics"]["interpretable_submodel_concept_trace_observed"] == 1.0
    assert report["metrics"]["runtime_submodel_route_action_grounding_observed"] == 1.0
    assert report["metrics"]["runtime_submodel_counterfactual_route_separation_observed"] == 1.0
    assert report["metrics"]["runtime_submodel_concept_trace_observed"] == 1.0
    assert report["metrics"]["submodel_intervention_trace_integrity_observed"] == 1.0
    assert report["metrics"]["submodel_ablation_effect_observed"] == 1.0
    assert report["metrics"]["submodel_reactivation_recovery_observed"] == 1.0
    assert report["metrics"]["submodel_credit_assignment_trace_integrity_observed"] == 1.0
    assert report["metrics"]["submodel_credit_selectivity_observed"] == 1.0
    assert report["metrics"]["submodel_credit_state_budget_observed"] == 1.0
    assert report["metrics"]["runtime_submodel_local_credit_assignment_observed"] == 1.0
    assert report["metrics"]["runtime_submodel_feedback_trace_observed"] == 1.0
    assert report["metrics"]["submodel_structural_adaptation_trace_integrity_observed"] == 1.0
    assert report["metrics"]["submodel_structural_growth_bounded_observed"] == 1.0
    assert report["metrics"]["submodel_structural_pruning_observed"] == 1.0
    assert report["metrics"]["submodel_scientific_hypothesis_trace_integrity_observed"] == 1.0
    assert report["metrics"]["submodel_counterexample_revision_observed"] == 1.0
    assert report["metrics"]["submodel_scientific_model_budget_observed"] == 1.0
    assert report["metrics"]["submodel_hypothesis_bank_integrity_observed"] == 1.0
    assert report["metrics"]["submodel_open_ended_selection_observed"] == 1.0
    assert report["metrics"]["submodel_hypothesis_bank_budget_observed"] == 1.0
    assert report["metrics"]["micro_turn_event_budget_observed"] == 1.0
    assert report["metrics"]["foreground_background_context_handoff_observed"] == 1.0
    assert report["metrics"]["interrupt_recovery_trace_observed"] == 1.0
    assert report["metrics"]["simultaneous_stream_route_integrity_observed"] == 1.0
    assert report["metrics"]["time_aligned_backchannel_policy_observed"] == 1.0
    assert report["metrics"]["phase_assigned_submodel_route_observed"] == 1.0
    assert report["metrics"]["uncertainty_bucket_specialization_observed"] == 1.0
    assert report["metrics"]["denoising_correction_trace_integrity_observed"] == 1.0
    assert report["metrics"]["block_independent_local_update_budget_observed"] == 1.0
    assert report["metrics"]["sparse_verifier_grounding_observed"] == 1.0
    assert report["metrics"]["sparse_verifier_trace_integrity_observed"] == 1.0
    assert report["metrics"]["sparse_verifier_energy_budget_observed"] == 1.0
    assert report["metrics"]["sparse_verifier_uncertainty_observed"] == 1.0
    assert report["metrics"]["sparse_verifier_selection_observed"] == 1.0
    assert report["metrics"]["sparse_best_of_n_bounded_count_observed"] == 1.0
    assert report["metrics"]["sparse_best_of_n_branch_diversity_observed"] == 1.0
    assert report["metrics"]["sparse_best_of_n_verifier_selection_observed"] == 1.0
    assert report["metrics"]["sparse_best_of_n_summary_alignment_observed"] == 1.0
    assert report["metrics"]["self_correction_bounded_loop_observed"] == 1.0
    assert report["metrics"]["self_correction_improvement_observed"] == 1.0
    assert report["metrics"]["self_correction_rollback_reason_observed"] == 1.0
    assert report["metrics"]["self_correction_verifier_failure_observed"] == 1.0
    assert report["metrics"]["bounded_tree_search_depth_observed"] == 1.0
    assert report["metrics"]["bounded_tree_search_branch_factor_observed"] == 1.0
    assert report["metrics"]["bounded_tree_search_event_budget_observed"] == 1.0
    assert report["metrics"]["bounded_tree_search_verifier_selection_observed"] == 1.0
    assert report["metrics"]["reasoning_forest_lane_bounded_count_observed"] == 1.0
    assert report["metrics"]["reasoning_forest_lane_read_only_snapshot_observed"] == 1.0
    assert report["metrics"]["reasoning_forest_lane_diversity_observed"] == 1.0
    assert report["metrics"]["reasoning_forest_lane_verifier_selection_observed"] == 1.0
    assert report["metrics"]["reasoning_forest_lane_selection_reason_observed"] == 1.0
    assert report["metrics"]["hierarchical_reasoning_instruction_observed"] == 1.0
    assert report["metrics"]["hierarchical_reasoning_execution_trace_observed"] == 1.0
    assert report["metrics"]["hierarchical_reasoning_verification_trace_observed"] == 1.0
    assert report["metrics"]["hierarchical_reasoning_plan_alignment_observed"] == 1.0
    runtime = report["details"]["modular_runtime"]
    assert runtime["module_order"] == ["encoder", "memory_controller", "world_model", "planner", "actor"]
    assert runtime["selected_action"]["branch_id"] == "primary"
    assert runtime["counterfactual_action"]["branch_id"] == "counterfactual-1"
    assert report["details"]["runtime_trace_digest"]["trace_digest"]
    assert report["details"]["runtime_trace_replay"]["consistent"] is True
    assert report["details"]["lejepa_latent_health"]["observed_only"] is True
    assert report["details"]["lejepa_latent_health"]["metrics"]["lejepa_factor_disentanglement"] == 1.0
    assert report["details"]["micro_turn_interaction"]["observed_only"] is True
    assert report["details"]["micro_turn_interaction"]["trace"]["simultaneous_bucket_count"] >= 1
    assert report["details"]["phase_assigned_submodel_blocks"]["observed_only"] is True
    assert (
        report["details"]["phase_assigned_submodel_blocks"]["metrics"][
            "block_independent_local_update_budget"
        ]
        == 1.0
    )
    manifold = report["details"]["manifold_trace_support"]
    assert manifold["observed_only"] is True
    assert manifold["trajectory_case_count"] == 2
    assert manifold["dense_scan_baseline_count"] == 3
    assert manifold["indexed_scan_reduction_ratio"] > 0.0
    assert manifold["trajectory_top_match_ratio"] == 1.0
    assert manifold["candidate_miss"] is False
    assert all(case["top_match"] for case in manifold["case_results"])
    assert all(case["indexed_candidate_ok"] for case in manifold["case_results"])
    delta_steering = report["details"]["delta_memory_steering"]
    assert delta_steering["observed_only"] is True
    assert delta_steering["traces"]["primary_steering_event"]["event_type"] == "memory_steering_event"
    assert delta_steering["traces"]["primary_steering_event"]["text_reinjection_used"] is False
    assert delta_steering["traces"]["primary_steering_event"]["steering_ids"] == [301]
    assert delta_steering["traces"]["counterfactual_steering_event"]["steering_ids"] == [302]
    predictive_spike = report["details"]["predictive_error_gated_spike"]
    assert predictive_spike["observed_only"] is True
    assert predictive_spike["expected_correction_spikes"] == 0
    assert predictive_spike["surprise_correction_spikes"] > 0
    assert predictive_spike["entropy_reduction"] == 1.0
    assert predictive_spike["state_budget_ok"] is True
    phase_binding = report["details"]["phase_synchronized_binding"]
    assert phase_binding["observed_only"] is True
    assert phase_binding["metrics"]["phase_binding_coincidence_integrity"] == 1.0
    assert phase_binding["traces"]["bound_pairs"] == [(101, 301)]
    forward_only = report["details"]["forward_only_local_update"]
    assert forward_only["observed_only"] is True
    assert forward_only["metrics"]["forward_only_local_update_stability"] == 1.0
    sparse_verifier = report["details"]["sparse_verifier"]
    assert sparse_verifier["observed_only"] is True
    assert sparse_verifier["selected_branch"] == "primary"
    assert sparse_verifier["selected_passed"] is True
    sparse_best_of_n = report["details"]["sparse_best_of_n"]
    assert sparse_best_of_n["observed_only"] is True
    assert sparse_best_of_n["candidate_count"] == 3
    assert sparse_best_of_n["selected_branch"] == "primary"
    assert sparse_best_of_n["summary_matches_selection"] is True
    self_correction = report["details"]["self_correction_trace"]
    assert self_correction["observed_only"] is True
    assert self_correction["max_loops"] == 2
    assert self_correction["selected_branch"] == "primary"
    assert self_correction["correction_applied"] is True
    bounded_tree_search = report["details"]["bounded_tree_search"]
    assert bounded_tree_search["observed_only"] is True
    assert bounded_tree_search["max_depth"] == 2
    assert bounded_tree_search["selected_branch"] == "primary"
    assert bounded_tree_search["dropped_candidates"][0]["drop_reason"] == "depth_limit"
    reasoning_forest = report["details"]["reasoning_forest_lane"]
    assert reasoning_forest["observed_only"] is True
    assert reasoning_forest["lane_count"] == 3
    assert reasoning_forest["selected_branch"] == "primary"
    assert all(item["snapshot_read_only"] for item in reasoning_forest["lane_summaries"])
    hierarchical = report["details"]["hierarchical_reasoning"]
    assert hierarchical["observed_only"] is True
    assert hierarchical["selected_branch"] == "primary"
    assert hierarchical["plan_execution_alignment"] is True
    assert forward_only["traces"]["positive_update"]["bptt_used"] is False


def test_phase4_scale_continual_benchmark_returns_expected_metrics():
    module = _load_script("phase4_scale_continual_benchmark.py")
    report = module.run_phase4_scale_continual_benchmark()

    assert report["evaluator_name"] == "Phase4ScaleContinualBenchmark"
    assert report["passed"] is True
    assert report["metrics"]["structural_plasticity_stability"] == 1.0
    assert report["metrics"]["hippocampal_transfer_integrity"] == 1.0
    assert report["metrics"]["scale_out_retention_integrity"] == 1.0
    assert report["metrics"]["continual_drift_recovery_integrity"] == 1.0
    assert report["quality_metrics"]["scale_out_retention_rate"] >= 0.99
    assert report["quality_metrics"]["scale_out_average_query_ms"] <= 30.0
    assert report["quality_metrics"]["continual_baseline_recovered"] == 1.0


def test_agent_dialogue_benchmark_tracks_direction_shift_following():
    module = _load_script("agent_dialogue_benchmark.py")
    report = module.run_agent_dialogue_benchmark()

    assert report["evaluator_name"] == "AgentDialogueEvaluator"
    assert report["passed"] is True
    assert report["metrics"]["direction_shift_following"] == 1.0
    shift_case = report["details"]["test_results"][-1]
    assert shift_case["shift_following_score"] == 1.0
    assert "可読性" in shift_case["response"]
    assert "引数" not in shift_case["response"]


def test_future_state_consistency_benchmark_returns_expected_metrics():
    module = _load_script("future_state_consistency_benchmark.py")
    report = module.run_future_state_consistency_benchmark()

    assert report["evaluator_name"] == "FutureStateConsistencyBenchmark"
    assert report["passed"] is True
    assert report["metrics"]["future_state_consistency"] == 1.0
    assert report["metrics"]["future_state_memory_grounding"] == 1.0
    assert report["metrics"]["future_state_transition_integrity"] == 1.0
    assert report["metrics"]["future_state_command_integrity"] == 1.0
    assert report["metrics"]["future_state_predictor_snapshot_integrity"] == 1.0
    assert report["metrics"]["future_state_counterfactual_integrity"] == 1.0
    assert report["metrics"]["future_state_counterfactual_usefulness"] == 1.0
    assert report["metrics"]["future_state_branching_integrity"] == 1.0
    assert report["metrics"]["future_state_options_integrity"] == 1.0
    assert report["metrics"]["future_state_ranking_integrity"] == 1.0
    assert report["metrics"]["future_state_decision_brief_integrity"] == 1.0
    assert report["metrics"]["future_state_choice_integrity"] == 1.0
    assert report["metrics"]["future_state_choice_reason_integrity"] == 1.0
    assert report["metrics"]["future_state_simulation_integrity"] == 1.0
    assert report["metrics"]["future_state_simulation_usefulness"] == 1.0
    assert report["metrics"]["future_state_shift_tracking_integrity"] == 1.0
    assert report["metrics"]["future_state_transition_operator_coverage"] == 1.0
    assert report["metrics"]["future_state_transition_operator_consistency"] == 1.0
    assert report["metrics"]["future_state_counterfactual_branch_viability"] == 1.0
    assert report["metrics"]["future_state_speculative_acceptance_ratio"] == 1.0
    assert report["metrics"]["future_state_speculative_rollback_observability"] == 1.0
    assert report["metrics"]["future_state_fluid_trace_integrity"] == 1.0
    assert report["metrics"]["future_state_fluid_support_integrity"] == 1.0
    assert report["metrics"]["future_state_refinement_loop_integrity"] == 1.0
    assert report["metrics"]["future_state_adaptive_refinement"] == 1.0
    assert report["metrics"]["future_state_adaptive_depth_efficiency_observed"] == 1.0
    assert report["metrics"]["future_state_rewarded_action_selection_integrity"] == 1.0
    assert report["metrics"]["future_state_policy_update_stability"] == 1.0
    assert report["metrics"]["future_state_energy_aware_action_preference"] == 1.0
    assert report["metrics"]["future_state_focused_retrieval_hit_ratio"] == 1.0
    assert report["metrics"]["future_state_branch_level_decision_consistency"] == 1.0
    assert report["metrics"]["future_state_spatial_projection_integrity"] == 1.0
    assert report["metrics"]["future_state_spatial_topology_consistency"] == 1.0
    assert report["metrics"]["future_state_spatial_occlusion_reasoning"] == 1.0
    assert report["metrics"]["future_state_spatial_counterfactual_selection"] == 1.0
    assert report["metrics"]["future_state_spatial_adjacency_consistency"] == 1.0
    assert report["metrics"]["future_state_spatial_door_connectivity_integrity"] == 1.0
    assert report["metrics"]["future_state_spatial_multi_room_counterfactual_selection"] == 1.0
    assert report["metrics"]["future_state_spatial_route_planning_integrity"] == 1.0
    assert report["metrics"]["future_state_spatial_affordance_action_selection"] == 1.0
    assert report["metrics"]["future_state_spatial_energy_aware_route_selection"] == 1.0
    assert report["metrics"]["future_state_spatial_route_state_update_integrity"] == 1.0
    assert report["metrics"]["future_state_spatial_invalid_action_rejection"] == 1.0
    assert report["metrics"]["future_state_spatial_route_rollback_observability"] == 1.0
    assert report["metrics"]["future_state_spatial_route_execution_cost_bound"] == 1.0
    spatial_case = report["details"]["spatial_room_geometry"]
    assert spatial_case["success"] is True
    assert spatial_case["selected_hypothesis"] == "observed_occlusion"
    assert len(spatial_case["counterfactual_hypotheses"]) >= 3
    assert spatial_case["hypothesis"]["closed_room"] is True
    assert spatial_case["hypothesis"]["door_wall"] == "south"
    assert spatial_case["hypothesis"]["room_area"] == 24
    spatial_adjacency = report["details"]["spatial_adjacency"]
    assert spatial_adjacency["success"] is True
    assert spatial_adjacency["selected_hypothesis"] == "observed_adjacency"
    assert len(spatial_adjacency["counterfactual_hypotheses"]) >= 3
    assert spatial_adjacency["hypothesis"]["all_rooms_connected"] is True
    assert spatial_adjacency["hypothesis"]["door_links_valid"] == 1
    assert spatial_adjacency["hypothesis"]["total_area"] == 28
    spatial_route = report["details"]["spatial_route_planning"]
    assert spatial_route["success"] is True
    assert spatial_route["selected_route"] == "door_route"
    assert spatial_route["ranked_routes"][0]["path"] == ["entry", "kitchen"]
    assert spatial_route["ranked_routes"][0]["required_affordance"] == "door_opening"
    spatial_execution = report["details"]["spatial_route_execution"]
    assert spatial_execution["success"] is True
    assert spatial_execution["accepted_trace"]["end_room"] == "kitchen"
    assert spatial_execution["rejected_trace"]["route"] == "wall_crossing"
    assert spatial_execution["rejected_trace"]["rollback_observable"] is True
    first_case = report["details"]["test_results"][0]
    assert first_case["predicted_action"]
    assert first_case["predicted_target_state"] == "ship the release"
    assert first_case["predicted_command"] == "python scripts/eval/release_soak.py --include-accuracy"
    assert first_case["alternative_action"]
    assert first_case["alternative_target_state"] == "ship the release"
    assert first_case["alternative_command"] == "python scripts/eval/release_soak.py --include-accuracy"
    assert first_case["secondary_alternative_action"]
    assert first_case["secondary_alternative_target_state"] == "ship the release"
    assert first_case["secondary_alternative_command"] == "python scripts/eval/release_gate.py"
    assert "Primary:" in first_case["options_response"]
    assert "1. Alternative:" in first_case["ranked_options_response"]
    assert "Decision brief:" in first_case["decision_brief_response"]
    assert first_case["simulation_response"].startswith("Lightweight simulation:")
    assert first_case["chosen_plan"] == "alternative"
    assert first_case["choice_reason"]
    assert "alternative plan" in first_case["choice_response"]
    assert first_case["predictor_state"]["category"] == "release"
    assert first_case["predictor_state"]["target_state"] == "ship the release"
    assert first_case["predictor_state"]["best_simulated_branch"] == "alternative"
    assert isinstance(first_case["predictor_state"]["simulated_branch_candidates"], list)
    assert first_case["fluid_trace"]["bounded"] is True
    assert first_case["reward_trace"]["total_reward"] >= 0.55
    assert first_case["policy_trace"]["policy_stability"] >= 0.55
    assert first_case["fluid_trace"]["support_score"] > 0.0
    assert isinstance(first_case["refinement_trace"], dict)
    assert first_case["refinement_trace"]["loop_count"] >= 1
    assert first_case["refinement_trace"]["adaptive_depth_budget"]["allocated_loop_budget"] >= 1
    assert first_case["runtime_state"]["transition_count"] >= 1
    assert report["metrics"]["future_state_runtime_tracking_integrity"] == 1.0
    shift_case = report["details"]["test_results"][2]
    assert shift_case["runtime_state"]["shift_count"] >= 1
    assert shift_case["runtime_state"]["last_shift_from"] == "ship the release"
    assert shift_case["runtime_state"]["last_shift_to"] == "improve the design"
    assert shift_case["runtime_state"]["last_best_simulated_branch"]


def test_energy_efficiency_benchmark_returns_expected_metrics():
    module = _load_script("energy_efficiency_benchmark.py")
    report = module.run_energy_efficiency_benchmark()

    assert report["evaluator_name"] == "EnergyEfficiencyBenchmark"
    assert report["passed"] is True
    assert report["overall_score"] == 1.0
    assert report["metrics"]["energy_per_success_proxy"] == 1.0
    assert report["metrics"]["performance_energy_ratio_proxy"] >= 0.20
    assert report["metrics"]["ann_cost_advantage_proxy"] >= 8.0
    assert report["metrics"]["sparse_event_cost_score"] == 1.0
    assert report["metrics"]["brain_efficiency_alignment_proxy"] >= 0.85
    assert report["metrics"]["memory_per_success_proxy"] == 1.0
    assert report["metrics"]["low_overhead_route_score"] == 1.0
    assert report["metrics"]["bounded_latency_score"] >= 0.8
    assert report["metrics"]["stochastic_readout_integrity"] == 1.0
    assert report["metrics"]["edge_low_precision_persistence_observed"] == 1.0
    assert report["metrics"]["edge_sparse_routing_table_observed"] == 1.0
    assert report["metrics"]["edge_event_compression_observed"] == 1.0
    assert report["metrics"]["edge_sparse_readout_storage_observed"] == 1.0
    assert report["metrics"]["edge_storage_profile_integrity_observed"] == 1.0
    assert report["metrics"]["edge_sparse_readout_row_reduction_observed"] == 0.5
    assert report["metrics"]["edge_multilevel_weight_profile_observed"] == 1.0
    assert report["metrics"]["edge_format_compatibility_observed"] == 1.0
    assert report["metrics"]["edge_manifest_integrity_observed"] == 1.0
    assert report["metrics"]["edge_strict_format_validation_observed"] == 1.0
    assert report["metrics"]["edge_payload_validation_report_observed"] == 1.0
    assert report["metrics"]["edge_delta_state_persistence_observed"] == 1.0
    assert report["metrics"]["edge_delta_state_budget_observed"] == 1.0
    assert report["metrics"]["edge_delta_state_manifest_integrity_observed"] == 1.0
    assert report["metrics"]["neuromorphic_ir_schema_integrity_observed"] == 1.0
    assert report["metrics"]["neuromorphic_capability_manifest_integrity_observed"] == 1.0
    assert report["metrics"]["neuromorphic_backend_profile_compatibility_observed"] == 1.0
    assert report["metrics"]["neuromorphic_sparse_event_budget_observed"] == 1.0
    assert report["metrics"]["neuromorphic_profile_report_integrity_observed"] == 1.0
    assert report["metrics"]["neuromorphic_stage_e_state_trace_ir_observed"] == 1.0
    assert report["metrics"]["neuromorphic_stage_e_routing_hint_coverage_observed"] == 1.0
    assert report["metrics"]["neuromorphic_stage_e_online_update_policy_observed"] == 1.0
    assert report["metrics"]["neuromorphic_stage_e_event_budget_observed"] == 1.0
    assert report["metrics"]["neuromorphic_profile_history_regression_observed"] == 1.0
    assert report["neuromorphic_profile_trend"]["schema"] == "sara-neuromorphic-profile-trend-v1"
    assert report["neuromorphic_profile_trend"]["has_previous"] is False
    assert report["neuromorphic_profile_trend"]["regression_count"] == 0
    assert report["metric_scores"]["ann_cost_advantage_proxy"] == 1.0
    edge_case = report["details"]["test_results"][2]
    assert edge_case["memory_hit"] == "edge_low_precision_persistence"
    assert edge_case["predicted_token"] == 65
    assert edge_case["compact_row_qweight_count"] == 2
    assert edge_case["active_row_deltas"] == [0, 1]
    assert edge_case["stored_row_count"] == 2
    assert edge_case["readout_row_count"] == 4
    assert edge_case["row_reduction_ratio"] == 0.5
    assert edge_case["compact_weight_count"] == 4
    assert edge_case["multilevel_weight_levels"] == 8
    assert edge_case["quantized_weight_count"] == 4
    assert edge_case["format_version"] == 2
    assert edge_case["unsupported_capabilities"] == []
    assert "active_row_readout_storage" in edge_case["format_capabilities"]
    assert edge_case["manifest_schema"] == "sara-edge-manifest-v1"
    assert edge_case["manifest_digest_algorithm"] == "sha256"
    assert edge_case["delta_state_units"] == 2
    assert edge_case["delta_state_entry_count"] == 2
    assert edge_case["delta_state_budget_ok"] is True
    assert edge_case["spike_event_ir_schema"] == "sara-spike-event-ir-v1"
    assert edge_case["spike_event_ir_event_count"] == 6
    assert "neuromorphic_state_trace_ir" in edge_case["format_capabilities"]
    assert edge_case["neuromorphic_stage_e_architecture_profile"]["state_trace_event_count"] == 4
    assert edge_case["neuromorphic_stage_e_architecture_profile"]["state_budget_units"] == 8
    assert edge_case["neuromorphic_stage_e_architecture_profile"]["state_budget_limit"] == 8
    assert edge_case["neuromorphic_stage_e_architecture_profile"]["routing_hints"] == [
        "denoising_correction_trace",
        "foreground_background_handoff",
        "micro_turn_interaction",
        "phase_assigned_submodel_block",
    ]
    assert edge_case["neuromorphic_stage_e_architecture_profile"]["adapter_policies"] == [
        "freeze_state_for_inference_profile",
        "native_online_update",
    ]
    assert edge_case["neuromorphic_profiles"] == ["lava", "spinnaker", "akida"]
    assert edge_case["neuromorphic_backend_compatibility"] == {
        "akida": True,
        "lava": True,
        "spinnaker": True,
    }
    assert edge_case["neuromorphic_profile_count"] == 3
    assert edge_case["neuromorphic_profile_compatibility"] == {
        "akida": True,
        "lava": True,
        "spinnaker": True,
    }
    assert edge_case["neuromorphic_profile_report_checks"]["akida"]["checks"][
        "low_precision_weight_ok"
    ] is True
    assert (
        edge_case["neuromorphic_profile_report_checks"]["akida"][
            "online_update_adapter_policy"
        ]
        == "freeze_state_for_inference_profile"
    )
    assert edge_case["validation_errors"] == []
    stochastic_case = report["details"]["test_results"][-1]
    assert stochastic_case["memory_hit"] == "stochastic_readout"
    assert stochastic_case["selected_label"] == "primary"
    assert stochastic_case["success_per_energy_proxy"] > 0.0
    assert report["details"]["total_ann_reference_cost_units"] > report["details"]["total_sara_energy_cost_units"]


def test_energy_efficiency_neuromorphic_profile_trend_detects_regression():
    module = _load_script("energy_efficiency_benchmark.py")
    previous = module.run_energy_efficiency_benchmark()
    current = copy.deepcopy(previous)
    edge_case = current["details"]["test_results"][2]
    edge_case["neuromorphic_profile_compatibility"]["akida"] = False
    edge_case["neuromorphic_profile_report_checks"]["akida"]["compatible"] = False
    edge_case["neuromorphic_profile_report_checks"]["akida"]["checks"][
        "low_precision_weight_ok"
    ] = False

    trend = module.build_neuromorphic_profile_trend(current, previous)

    assert trend["has_previous"] is True
    assert trend["regression_count"] == 2
    assert {
        (item.get("profile"), item.get("kind"), item.get("check", ""))
        for item in trend["regressions"]
    } == {
        ("akida", "compatibility_regression", ""),
        ("akida", "check_regression", "low_precision_weight_ok"),
    }


def test_parameter_efficiency_benchmark_returns_expected_metrics():
    module = _load_script("parameter_efficiency_benchmark.py")
    report = module.run_parameter_efficiency_benchmark()

    assert report["evaluator_name"] == "ParameterEfficiencyBenchmark"
    assert report["passed"] is True
    assert report["metrics"]["quality_per_kparam_score"] >= 0.5
    assert report["metrics"]["quality_per_mb_score"] >= 0.5
    assert report["metrics"]["bounded_parameter_footprint_score"] == 1.0
    assert report["metrics"]["bounded_artifact_footprint_score"] == 1.0
    assert report["metrics"]["average_quality_per_kparam"] > 0.0
    assert report["metrics"]["average_quality_per_mb"] > 0.0
    assert len(report["details"]["test_results"]) == 3
    for case in report["details"]["test_results"]:
        assert case["active_parameter_count"] > 0
        assert case["artifact_size_bytes"] > 0
        assert case["quality_per_kparam"] > 0.0
        assert case["quality_per_mb"] > 0.0


def test_phase3_accuracy_suite_formats_human_readable_summary():
    module = _load_script("phase3_accuracy_suite.py")
    report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 1.0,
        "passed": True,
        "trend": {
            "regression_count": 0,
            "improvements": [
                {
                    "metric": "agent_dialogue.direction_shift_following",
                    "delta": 0.1,
                },
                {
                    "metric": "future_state_consistency.future_state_command_integrity",
                    "delta": 0.05,
                },
                {
                    "metric": "future_state_consistency.future_state_counterfactual_integrity",
                    "delta": 0.03,
                },
                {
                    "metric": "future_state_consistency.future_state_counterfactual_usefulness",
                    "delta": 0.02,
                },
                {
                    "metric": "future_state_consistency.future_state_choice_integrity",
                    "delta": 0.01,
                },
                {
                    "metric": "future_state_consistency.future_state_choice_reason_integrity",
                    "delta": 0.01,
                },
                {
                    "metric": "spiking_llm.hierarchical_context_integrity",
                    "delta": 0.02,
                },
                {
                    "metric": "energy_efficiency.memory_per_success_proxy",
                    "delta": 0.03,
                },
                {
                    "metric": "energy_efficiency.stochastic_readout_integrity",
                    "delta": 0.04,
                },
                {
                    "metric": "future_state_consistency.future_state_branching_integrity",
                    "delta": 0.02,
                },
                {
                    "metric": "future_state_consistency.future_state_options_integrity",
                    "delta": 0.02,
                },
                {
                    "metric": "future_state_consistency.future_state_ranking_integrity",
                    "delta": 0.02,
                },
                {
                    "metric": "future_state_consistency.future_state_decision_brief_integrity",
                    "delta": 0.02,
                },
                {
                    "metric": "future_state_consistency.future_state_shift_tracking_integrity",
                    "delta": 0.04,
                },
                {
                    "metric": "future_state_consistency.future_state_simulation_integrity",
                    "delta": 0.03,
                },
                {
                    "metric": "future_state_consistency.future_state_fluid_trace_integrity",
                    "delta": 0.02,
                },
                {
                    "metric": "future_state_consistency.future_state_fluid_support_integrity",
                    "delta": 0.02,
                },
                {
                    "metric": "future_state_consistency.future_state_refinement_loop_integrity",
                    "delta": 0.02,
                },
                {
                    "metric": "future_state_consistency.future_state_adaptive_refinement",
                    "delta": 0.02,
                },
            ],
            "regressions": [],
            "unchanged": [],
            "new_metrics": [],
        },
        "focus_summary": {
            "few_shot": {"passed": True, "score": 1.0},
            "continual": {"passed": True, "score": 1.0},
            "retrieval_hygiene": {"passed": True, "score": 0.8},
            "adaptive_readiness": {
                "passed": True,
                "score": 1.0,
                "metrics": {
                    "task_switch_adaptation.meta_adaptation_parameter_integrity": 1.0,
                },
            },
            "predictive_readiness": {"passed": True, "score": 1.0},
            "efficiency_readiness": {
                "passed": True,
                "score": 0.95,
                "metrics": {
                    "energy_efficiency.energy_per_success_proxy": 1.0,
                    "energy_efficiency.performance_energy_ratio_proxy": 0.22,
                    "energy_efficiency.ann_cost_advantage_proxy": 12.0,
                    "energy_efficiency.sparse_event_cost_score": 1.0,
                    "energy_efficiency.brain_efficiency_alignment_proxy": 0.9,
                    "energy_efficiency.memory_per_success_proxy": 0.0,
                    "energy_efficiency.low_overhead_route_score": 1.0,
                    "energy_efficiency.bounded_latency_score": 0.8,
                    "energy_efficiency.stochastic_readout_integrity": 1.0,
                },
            },
            "parameter_efficiency": {
                "passed": True,
                "score": 0.9,
                "metrics": {
                    "parameter_efficiency.average_quality_per_kparam": 12.0,
                    "parameter_efficiency.average_quality_per_mb": 40.0,
                },
            },
            "consolidation_readiness": {
                "passed": True,
                "score": 1.0,
                "metrics": {
                    "continual_consolidation.replay_recovery_integrity": 1.0,
                    "continual_consolidation.long_horizon_consolidation_retention": 1.0,
                    "continual_consolidation.counterfactual_replay_selection_integrity": 1.0,
                    "continual_consolidation.replay_upgrade_reindex_integrity": 1.0,
                    "continual_consolidation.memory_health_index_integrity": 1.0,
                    "continual_consolidation.replay_noise_resilience_integrity": 1.0,
                    "continual_consolidation.astro_modulation_stability": 1.0,
                },
            },
                "cognitive_runtime_readiness": {
                    "passed": True,
                    "score": 1.0,
                    "metrics": {
                        "cognitive_runtime.common_spike_space_integrity": 1.0,
                    "cognitive_runtime.temporal_compression_efficiency": 1.0,
                    "cognitive_runtime.modality_temporal_budget_integrity": 1.0,
                    "cognitive_runtime.dendritic_context_gate_stability": 1.0,
                    "cognitive_runtime.spiking_hjepa_latent_transition": 1.0,
                    "cognitive_runtime.reverse_reasoning_trace_integrity": 1.0,
                    "cognitive_runtime.causal_candidate_trace_integrity": 1.0,
                    "cognitive_runtime.module_orchestration_integrity": 1.0,
                    "cognitive_runtime.counterfactual_lane_integrity": 1.0,
                    "cognitive_runtime.action_trace_observability": 1.0,
                        "cognitive_runtime.runtime_trace_replay_consistency": 1.0,
                    },
                    "plastic_submodel_observed_metrics": {
                        "cognitive_runtime.plastic_submodel_registry_integrity_observed": 1.0,
                        "cognitive_runtime.dynamic_submodel_route_integrity_observed": 1.0,
                        "cognitive_runtime.submodel_relearning_trace_integrity_observed": 1.0,
                        "cognitive_runtime.interpretable_submodel_concept_trace_observed": 1.0,
                        "cognitive_runtime.runtime_submodel_route_action_grounding_observed": 1.0,
                        "cognitive_runtime.runtime_submodel_counterfactual_route_separation_observed": 1.0,
                        "cognitive_runtime.runtime_submodel_concept_trace_observed": 1.0,
                        "cognitive_runtime.submodel_intervention_trace_integrity_observed": 1.0,
                        "cognitive_runtime.submodel_ablation_effect_observed": 1.0,
                        "cognitive_runtime.submodel_reactivation_recovery_observed": 1.0,
                        "cognitive_runtime.submodel_credit_assignment_trace_integrity_observed": 1.0,
                        "cognitive_runtime.submodel_credit_selectivity_observed": 1.0,
                        "cognitive_runtime.submodel_credit_state_budget_observed": 1.0,
                        "cognitive_runtime.runtime_submodel_local_credit_assignment_observed": 1.0,
                        "cognitive_runtime.runtime_submodel_feedback_trace_observed": 1.0,
                        "cognitive_runtime.submodel_structural_adaptation_trace_integrity_observed": 1.0,
                        "cognitive_runtime.submodel_structural_growth_bounded_observed": 1.0,
                        "cognitive_runtime.submodel_structural_pruning_observed": 1.0,
                        "cognitive_runtime.submodel_scientific_hypothesis_trace_integrity_observed": 1.0,
                        "cognitive_runtime.submodel_counterexample_revision_observed": 1.0,
                        "cognitive_runtime.submodel_scientific_model_budget_observed": 1.0,
                        "cognitive_runtime.submodel_hypothesis_bank_integrity_observed": 1.0,
                        "cognitive_runtime.submodel_open_ended_selection_observed": 1.0,
                        "cognitive_runtime.submodel_hypothesis_bank_budget_observed": 1.0,
                        "cognitive_runtime.micro_turn_event_budget_observed": 1.0,
                        "cognitive_runtime.foreground_background_context_handoff_observed": 1.0,
                        "cognitive_runtime.interrupt_recovery_trace_observed": 1.0,
                        "cognitive_runtime.simultaneous_stream_route_integrity_observed": 1.0,
                        "cognitive_runtime.time_aligned_backchannel_policy_observed": 1.0,
                        "cognitive_runtime.phase_assigned_submodel_route_observed": 1.0,
                        "cognitive_runtime.uncertainty_bucket_specialization_observed": 1.0,
                        "cognitive_runtime.denoising_correction_trace_integrity_observed": 1.0,
                        "cognitive_runtime.block_independent_local_update_budget_observed": 1.0,
                    },
                },
        },
        "phase3_completion": {
            "label": "Phase 3 Completion Gate",
            "passed": True,
            "completion_score": 1.0,
            "failed_checks": [],
            "checks": {
                "overall.score_at_least_0_95": True,
                "trend.zero_regressions": True,
                "stage_a.accepted": True,
                "stage_b.minimum_requirements_passed": True,
                "stage_c.minimum_requirements_passed": True,
                "stage_d.minimum_requirements_passed": True,
                "stage_e.minimum_requirements_passed": True,
                "focus.few_shot.passed": True,
                "focus.continual.passed": True,
                "focus.retrieval_hygiene.passed": True,
                "focus.adaptive_readiness.passed": True,
                "focus.predictive_readiness.passed": True,
                "focus.efficiency_readiness.passed": True,
                "focus.consolidation_readiness.passed": True,
                "focus.cognitive_runtime_readiness.passed": True,
            },
        },
        "stage_a_acceptance": {
            "passed": True,
            "label": "Evaluation First",
            "acc_target": 0.95,
            "overall_score": 1.0,
            "checks": {
                "trend.zero_regressions": True,
                "overall.acc_target_0_95": True,
            },
        },
            "stage_b_readiness": {
                "passed": True,
                "minimum_requirements_passed": True,
                "label": "Lightweight World Model Prototypes",
                "readiness_score": 1.0,
                "promotion_candidate_ready": True,
                "promotion_candidate_failure_count": 0,
                "promotion_candidate_promoted": True,
                "rlm_observation_candidate_ready": True,
                "rlm_observation_candidate_failure_count": 0,
                "rlm_observation_candidate_promoted": True,
                "rlm_observation_promotion_readiness": {
                    "consecutive_passes": 0,
                    "required_streak": 3,
                    "recommended": False,
                    "promoted_to_minimum": True,
                },
                    "minimum_checks": {
                    "metric.future_state_transition_integrity": True,
                    "metric.future_state_command_integrity": True,
                    "metric.future_state_predictor_snapshot_integrity": True,
                    "metric.future_state_runtime_tracking_integrity": True,
                    "metric.future_state_shift_tracking_integrity": True,
                    "metric.future_state_transition_operator_coverage": True,
                    "metric.future_state_transition_operator_consistency": True,
                    "metric.future_state_counterfactual_branch_viability": True,
                    "metric.future_state_fluid_trace_integrity": True,
                    "metric.future_state_fluid_support_integrity": True,
                    "metric.future_state_refinement_loop_integrity": True,
                    "metric.future_state_adaptive_refinement": True,
                    "metric.future_state_rewarded_action_selection_integrity": True,
                    "metric.future_state_policy_update_stability": True,
                    "metric.future_state_energy_aware_action_preference": True,
                    "metric.future_state_focused_retrieval_hit_ratio": True,
                    "metric.future_state_branch_level_decision_consistency": True,
                },
                "checks": {
                    "metric.future_state_branching_integrity": True,
                    "metric.future_state_simulation_integrity": True,
                    "metric.future_state_simulation_usefulness": True,
                    "metric.future_state_speculative_acceptance_ratio": True,
                    "metric.future_state_speculative_rollback_observability": True,
                    "metric.future_state_focused_retrieval_hit_ratio": True,
                    "metric.future_state_branch_level_decision_consistency": True,
                },
            },
        "stage_c_readiness": {
            "passed": True,
            "minimum_requirements_passed": True,
            "label": "Meta-Adaptation Experiments",
            "readiness_score": 1.0,
            "minimum_failure_count": 0,
            "minimum_checks": {
                "metric.meta_adaptation_loop": True,
                "metric.meta_adaptation_parameter_integrity": True,
                "metric.temporal_self_distillation_stability": True,
            },
        },
        "stage_d_readiness": {
            "passed": True,
            "minimum_requirements_passed": True,
            "label": "Continual Consolidation",
            "readiness_score": 1.0,
            "minimum_failure_count": 0,
            "delta_memory_candidate_ready": True,
            "delta_memory_candidate_failure_count": 0,
            "delta_memory_candidate_promoted": False,
            "acceptance_candidate_count": 16,
            "acceptance_candidate_ready_count": 16,
            "acceptance_candidates_ready": True,
            "acceptance_candidate_failure_count": 0,
            "acceptance_candidate_stability": {
                "consecutive_passes": 3,
                "required_streak": 3,
                "recommended": True,
            },
            "delta_memory_promotion_readiness": {
                "consecutive_passes": 3,
                "required_streak": 3,
                "recommended": True,
                "promoted_to_minimum": False,
            },
            "minimum_checks": {
                "metric.replay_recovery_integrity": True,
                "metric.long_horizon_consolidation_retention": True,
                "metric.counterfactual_replay_selection_integrity": True,
                "metric.replay_upgrade_reindex_integrity": True,
                "metric.memory_health_index_integrity": True,
                "metric.replay_noise_resilience_integrity": True,
                "metric.astro_modulation_stability": True,
            },
        },
        "stage_e_readiness": {
            "passed": True,
            "minimum_requirements_passed": True,
            "label": "Modular Cognitive Runtime",
            "readiness_score": 1.0,
            "minimum_failure_count": 0,
            "minimum_checks": {
                "metric.common_spike_space_integrity": True,
                "metric.temporal_compression_efficiency": True,
                "metric.modality_temporal_budget_integrity": True,
                "metric.dendritic_context_gate_stability": True,
                "metric.spiking_hjepa_latent_transition": True,
                "metric.reverse_reasoning_trace_integrity": True,
                "metric.causal_candidate_trace_integrity": True,
                "metric.module_orchestration_integrity": True,
                "metric.counterfactual_lane_integrity": True,
                "metric.action_trace_observability": True,
                "metric.runtime_trace_replay_consistency": True,
            },
        },
        "focus_trend": {
            "retrieval_hygiene": {"status": "UP", "delta": 0.1},
            "adaptive_readiness": {"status": "NEW", "delta": None},
            "predictive_readiness": {"status": "NEW", "delta": None},
            "efficiency_readiness": {"status": "NEW", "delta": None},
            "parameter_efficiency": {"status": "UP", "delta": 0.05},
            "consolidation_readiness": {"status": "NEW", "delta": None},
            "cognitive_runtime_readiness": {"status": "NEW", "delta": None},
        },
        "component_reports": {
            "agent_dialogue": {
                "passed": True,
                "overall_score": 0.9,
                "metrics": {"direction_shift_following": 1.0},
                "details": {
                    "test_results": [
                        {
                            "shift_from": "Pythonの関数とは？",
                            "user_input": "リスト内包表記のメリットは何ですか？",
                            "shift_following_score": 1.0,
                        }
                    ]
                },
            },
            "sara_inference": {"passed": True, "overall_score": 1.0},
            "spiking_llm": {"passed": True, "overall_score": 1.0},
            "task_switch_adaptation": {"passed": True, "overall_score": 1.0},
            "future_state_consistency": {"passed": True, "overall_score": 1.0},
            "energy_efficiency": {
                "passed": True,
                "overall_score": 0.95,
                "metrics": {
                    "edge_delta_state_persistence_observed": 1.0,
                    "edge_delta_state_budget_observed": 1.0,
                    "edge_delta_state_manifest_integrity_observed": 1.0,
                    "neuromorphic_ir_schema_integrity_observed": 1.0,
                    "neuromorphic_capability_manifest_integrity_observed": 1.0,
                    "neuromorphic_backend_profile_compatibility_observed": 1.0,
                    "neuromorphic_sparse_event_budget_observed": 1.0,
                    "neuromorphic_profile_report_integrity_observed": 1.0,
                    "neuromorphic_stage_e_state_trace_ir_observed": 1.0,
                    "neuromorphic_stage_e_routing_hint_coverage_observed": 1.0,
                    "neuromorphic_stage_e_online_update_policy_observed": 1.0,
                    "neuromorphic_stage_e_event_budget_observed": 1.0,
                    "neuromorphic_profile_history_regression_observed": 1.0,
                },
                "details": {
                    "average_state_units": 2.5,
                },
                "neuromorphic_profile_trend": {
                    "has_previous": True,
                    "regression_count": 0,
                    "policy_change_count": 0,
                    "new_profiles": [],
                    "missing_profiles": [],
                },
            },
            "parameter_efficiency": {
                "passed": True,
                "overall_score": 0.9,
            },
            "continual_consolidation": {
                "passed": True,
                "overall_score": 1.0,
                "metrics": {
                    "delta_memory_residual_write_integrity_observed": 1.0,
                    "delta_memory_retention_gate_stability_observed": 1.0,
                    "delta_memory_context_recall_without_text_reinjection_observed": 1.0,
                    "delta_memory_state_budget_integrity_observed": 1.0,
                    "delta_memory_interference_guard_observed": 1.0,
                    "synaptic_tag_integrity_observed": 1.0,
                    "synaptic_tag_importance_score_observed": 1.0,
                    "synaptic_tag_replay_priority_observed": 1.0,
                    "synaptic_tag_pruning_candidate_observed": 1.0,
                    "synaptic_tag_state_budget_observed": 1.0,
                    "memory_phase_transition_integrity_observed": 1.0,
                    "memory_phase_retention_protection_observed": 1.0,
                    "memory_phase_plasticity_guard_observed": 1.0,
                    "memory_phase_overfixation_guard_observed": 1.0,
                    "memory_phase_state_budget_observed": 1.0,
                    "metabolic_budget_integrity_observed": 1.0,
                    "plasticity_reserve_integrity_observed": 1.0,
                    "structural_growth_bounded_observed": 1.0,
                    "pruning_reason_trace_observed": 1.0,
                    "resource_pressure_observed": 1.0,
                    "sleep_consolidation_retention_observed": 1.0,
                    "latent_replay_noise_resilience_observed": 1.0,
                    "sleep_consolidation_memory_health_observed": 1.0,
                    "latent_replay_counterfactual_branch_observed": 1.0,
                    "sleep_consolidation_energy_budget_observed": 1.0,
                    "astro_structural_unlock_observed": 1.0,
                    "astro_structural_lock_observed": 1.0,
                    "astro_bounded_stdp_fallback_observed": 1.0,
                    "world_model_replay_policy_trace_observed": 1.0,
                    "astro_policy_state_budget_observed": 1.0,
                    "delta_memory_phase_retention_policy_observed": 1.0,
                    "delta_memory_crystal_retention_observed": 1.0,
                    "delta_memory_liquid_forget_observed": 1.0,
                    "delta_memory_astro_gate_alignment_observed": 1.0,
                    "delta_memory_policy_state_budget_observed": 1.0,
                    "delta_memory_multi_history_recall_observed": 1.0,
                    "delta_memory_multi_history_noise_resilience_observed": 1.0,
                    "delta_memory_multi_history_health_observed": 1.0,
                    "delta_memory_multi_history_manifold_guard_observed": 1.0,
                    "delta_memory_erase_write_decoupling_observed": 1.0,
                    "delta_memory_erase_preserves_stable_memory_observed": 1.0,
                    "delta_memory_write_commits_residual_observed": 1.0,
                },
            },
            "cognitive_runtime": {
                "passed": True,
                "overall_score": 1.0,
                "metrics": {
                    "manifold_trace_support_observed": 1.0,
                    "manifold_trace_recall_observed": 1.0,
                    "manifold_trace_scan_budget_observed": 1.0,
                    "manifold_trace_index_scan_reduction_observed": 1.0,
                    "manifold_trace_candidate_guard_observed": 1.0,
                    "delta_memory_steering_integrity_observed": 1.0,
                    "delta_memory_counterfactual_isolation_observed": 1.0,
                    "delta_memory_trace_observability_observed": 1.0,
                    "predictive_spike_entropy_reduction_observed": 1.0,
                    "phase_binding_coincidence_integrity_observed": 1.0,
                    "forward_only_local_update_stability_observed": 1.0,
                    "lejepa_linear_identifiability_proxy_observed": 1.0,
                    "lejepa_latent_whitening_health_observed": 1.0,
                    "lejepa_factor_disentanglement_observed": 1.0,
                    "lejepa_latent_planning_consistency_observed": 1.0,
                    "lejepa_positive_pair_alignment_observed": 1.0,
                },
            },
            "phase5_predictive_coding": {
                "passed": True,
                "overall_score": 1.0,
                "metrics": {
                    "manifold_candidate_miss_guard": 1.0,
                },
            },
        },
    }

    summary = module.format_phase3_accuracy_summary(report)

    assert "SARA Engine Phase 3 Accuracy Summary" in summary
    assert "overall_status: PASS" in summary
    assert "Phase 3 Completion" in summary
    assert "- phase3_completion_status: PASS" in summary
    assert "- phase3_completion_score: 1.000" in summary
    assert "Stage A Acceptance" in summary
    assert "- stage_a_status: PASS" in summary
    assert "- stage_a_acc_target: 0.950" in summary
    assert "- stage_a_overall_score: 1.000" in summary
    assert "- stage_a_zero_regressions: True" in summary
    assert "- stage_a_acc_target_met: True" in summary
    assert "Stage B Readiness" in summary
    assert "- stage_b_status: PASS" in summary
    assert "- stage_b_readiness_score: 1.000" in summary
    assert "- stage_b_minimum_requirements_passed: True" in summary
    assert "- stage_b_minimum_failure_count: 0" in summary
    assert "- stage_b_transition_ready: True" in summary
    assert "- stage_b_command_ready: True" in summary
    assert "- stage_b_predictor_snapshot_ready: True" in summary
    assert "- stage_b_runtime_tracking_ready: True" in summary
    assert "- stage_b_shift_tracking_ready: True" in summary
    assert "- stage_b_operator_coverage_ready: True" in summary
    assert "- stage_b_operator_consistency_ready: True" in summary
    assert "- stage_b_counterfactual_viability_ready: True" in summary
    assert "- stage_b_fluid_trace_ready: True" in summary
    assert "- stage_b_fluid_support_ready: True" in summary
    assert "- stage_b_refinement_loop_ready: True" in summary
    assert "- stage_b_adaptive_refinement_ready: True" in summary
    assert "- stage_b_rewarded_action_selection_ready: True" in summary
    assert "- stage_b_policy_update_stability_ready: True" in summary
    assert "- stage_b_energy_aware_preference_ready: True" in summary
    assert "- stage_b_focused_retrieval_observed: True" in summary
    assert "- stage_b_branch_decision_consistency_observed: True" in summary
    assert "- stage_b_rlm_observation_candidate_ready: True" in summary
    assert "- stage_b_rlm_observation_candidate_failure_count: 0" in summary
    assert "- stage_b_rlm_observation_candidate_promoted: True" in summary
    assert "- stage_b_rlm_observation_consecutive_passes: 0" in summary
    assert "- stage_b_rlm_observation_promotion_recommended: False" in summary
    assert "- stage_b_branching_ready: True" in summary
    assert "- stage_b_simulation_ready: True" in summary
    assert "- stage_b_simulation_useful: True" in summary
    assert "- stage_b_speculative_acceptance_ready: True" in summary
    assert "- stage_b_speculative_rollback_ready: True" in summary
    assert "- stage_b_promotion_candidate_ready: True" in summary
    assert "- stage_b_promotion_candidate_failure_count: 0" in summary
    assert "- stage_b_promotion_candidate_promoted: True" in summary
    assert "- stage_b_promotion_consecutive_passes: " in summary
    assert "- stage_b_promotion_required_streak: 3" in summary
    assert "- stage_b_promotion_recommended: False" in summary
    assert "Stage C Readiness" in summary
    assert "- stage_c_status: PASS" in summary
    assert "- stage_c_readiness_score: 1.000" in summary
    assert "- stage_c_minimum_requirements_passed: True" in summary
    assert "- stage_c_meta_adaptation_loop_ready: True" in summary
    assert "- stage_c_parameter_integrity_ready: True" in summary
    assert "- stage_c_temporal_self_distillation_ready: True" in summary
    assert "Stage D Readiness" in summary
    assert "- stage_d_status: PASS" in summary
    assert "- stage_d_readiness_score: 1.000" in summary
    assert "- stage_d_minimum_requirements_passed: True" in summary
    assert "- stage_d_replay_recovery_ready: True" in summary
    assert "- stage_d_long_horizon_retention_ready: True" in summary
    assert "- stage_d_counterfactual_replay_ready: True" in summary
    assert "- stage_d_reindex_ready: True" in summary
    assert "- stage_d_memory_health_ready: True" in summary
    assert "- stage_d_replay_noise_resilience_ready: True" in summary
    assert "- stage_d_astro_modulation_ready: True" in summary
    assert "- stage_d_delta_memory_candidate_ready: True" in summary
    assert "- stage_d_delta_memory_candidate_failure_count: 0" in summary
    assert "- stage_d_delta_memory_candidate_promoted: False" in summary
    assert "- stage_d_delta_memory_consecutive_passes: 3" in summary
    assert "- stage_d_delta_memory_required_streak: 3" in summary
    assert "- stage_d_delta_memory_promotion_recommended: True" in summary
    assert "- stage_d_acceptance_candidate_count: 16" in summary
    assert "- stage_d_acceptance_candidate_ready_count: 16" in summary
    assert "- stage_d_acceptance_candidates_ready: True" in summary
    assert "- stage_d_acceptance_candidate_failure_count: 0" in summary
    assert "- stage_d_acceptance_candidate_consecutive_passes: 3" in summary
    assert "- stage_d_acceptance_candidate_required_streak: 3" in summary
    assert "- stage_d_acceptance_candidate_stability_recommended: True" in summary
    assert "Stage E Readiness" in summary
    assert "- stage_e_status: PASS" in summary
    assert "- stage_e_readiness_score: 1.000" in summary
    assert "- stage_e_minimum_requirements_passed: True" in summary
    assert "- stage_e_common_spike_space_ready: True" in summary
    assert "- stage_e_temporal_compression_ready: True" in summary
    assert "- stage_e_modality_budget_ready: True" in summary
    assert "- stage_e_dendritic_gate_ready: True" in summary
    assert "- stage_e_spiking_hjepa_ready: True" in summary
    assert "- stage_e_reverse_reasoning_ready: True" in summary
    assert "- stage_e_causal_candidate_trace_ready: True" in summary
    assert "- stage_e_module_orchestration_ready: True" in summary
    assert "- stage_e_counterfactual_lane_ready: True" in summary
    assert "- stage_e_action_trace_ready: True" in summary
    assert "- stage_e_runtime_trace_replay_ready: True" in summary
    assert "- few_shot_status: PASS" in summary
    assert "- hierarchical_context_trend: UP" in summary
    assert "- hierarchical_context_delta: +0.020" in summary
    assert "- continual_status: PASS" in summary
    assert "- retrieval_hygiene_status: PASS" in summary
    assert "- retrieval_hygiene_trend: UP" in summary
    assert "- retrieval_hygiene_delta: +0.100" in summary
    assert "- adaptive_readiness_status: PASS" in summary
    assert "- adaptive_readiness_score: 1.000" in summary
    assert "- adaptive_readiness_trend: NEW" in summary
    assert "- adaptation_parameter_integrity: 1.000" in summary
    assert "- adaptation_parameter_integrity_trend: NEW" in summary
    assert "- adaptation_parameter_integrity_delta: +0.000" in summary
    assert "- direction_shift_following: 1.000" in summary
    assert "- direction_shift_trend: UP" in summary
    assert "- direction_shift_delta: +0.100" in summary
    assert "Dialogue Shift Detail" in summary
    assert "- shift_from: Pythonの関数とは？" in summary
    assert "- shift_query: リスト内包表記のメリットは何ですか？" in summary
    assert "- shift_following_score: 1.000" in summary
    assert "- predictive_readiness_status: PASS" in summary
    assert "- predictive_readiness_score: 1.000" in summary
    assert "- predictive_readiness_trend: NEW" in summary
    assert "- predictive_command_trend: UP" in summary
    assert "- predictive_command_delta: +0.050" in summary
    assert "- predictive_counterfactual_trend: UP" in summary
    assert "- predictive_counterfactual_delta: +0.030" in summary
    assert "- predictive_counterfactual_usefulness_trend: UP" in summary
    assert "- predictive_counterfactual_usefulness_delta: +0.020" in summary
    assert "- predictive_choice_trend: UP" in summary
    assert "- predictive_choice_delta: +0.010" in summary
    assert "- predictive_choice_reason_trend: UP" in summary
    assert "- predictive_choice_reason_delta: +0.010" in summary
    assert "- predictive_branching_trend: UP" in summary
    assert "- predictive_branching_delta: +0.020" in summary
    assert "- predictive_options_trend: UP" in summary
    assert "- predictive_options_delta: +0.020" in summary
    assert "- predictive_ranking_trend: UP" in summary
    assert "- predictive_ranking_delta: +0.020" in summary
    assert "- predictive_decision_brief_trend: UP" in summary
    assert "- predictive_decision_brief_delta: +0.020" in summary
    assert "- predictive_shift_trend: UP" in summary
    assert "- predictive_shift_delta: +0.040" in summary
    assert "- predictive_simulation_trend: UP" in summary
    assert "- predictive_simulation_delta: +0.030" in summary
    assert "- predictive_fluid_trace_trend: UP" in summary
    assert "- predictive_fluid_trace_delta: +0.020" in summary
    assert "- predictive_fluid_support_trend: UP" in summary
    assert "- predictive_fluid_support_delta: +0.020" in summary
    assert "- predictive_refinement_loop_trend: UP" in summary
    assert "- predictive_refinement_loop_delta: +0.020" in summary
    assert "- predictive_adaptive_refinement_trend: UP" in summary
    assert "- predictive_adaptive_refinement_delta: +0.020" in summary
    assert "- efficiency_readiness_status: PASS" in summary
    assert "- efficiency_readiness_score: 0.950" in summary
    assert "- energy_per_success_proxy: 1.000" in summary
    assert "- performance_energy_ratio_proxy: 0.220" in summary
    assert "- ann_cost_advantage_proxy: 12.000" in summary
    assert "- sparse_event_cost_score: 1.000" in summary
    assert "- brain_efficiency_alignment_proxy: 0.900" in summary
    assert "- memory_per_success_proxy: 0.000" in summary
    assert "- low_overhead_route_score: 1.000" in summary
    assert "- bounded_latency_score: 0.800" in summary
    assert "- stochastic_readout_integrity: 1.000" in summary
    assert "- edge_delta_state_persistence_observed: 1.000" in summary
    assert "- edge_delta_state_budget_observed: 1.000" in summary
    assert "- edge_delta_state_manifest_integrity_observed: 1.000" in summary
    assert "- neuromorphic_ir_schema_integrity_observed: 1.000" in summary
    assert "- neuromorphic_capability_manifest_integrity_observed: 1.000" in summary
    assert "- neuromorphic_backend_profile_compatibility_observed: 1.000" in summary
    assert "- neuromorphic_sparse_event_budget_observed: 1.000" in summary
    assert "- neuromorphic_profile_report_integrity_observed: 1.000" in summary
    assert "- neuromorphic_stage_e_state_trace_ir_observed: 1.000" in summary
    assert "- neuromorphic_stage_e_routing_hint_coverage_observed: 1.000" in summary
    assert "- neuromorphic_stage_e_online_update_policy_observed: 1.000" in summary
    assert "- neuromorphic_stage_e_event_budget_observed: 1.000" in summary
    assert "- neuromorphic_profile_history_regression_observed: 1.000" in summary
    assert "- neuromorphic_profile_trend_has_previous: True" in summary
    assert "- neuromorphic_profile_trend_regression_count: 0" in summary
    assert "- neuromorphic_profile_trend_policy_change_count: 0" in summary
    assert "- neuromorphic_profile_trend_regression_details: none" in summary
    assert "- neuromorphic_profile_trend_policy_change_details: none" in summary
    assert "- average_state_units: 2.500" in summary
    assert "- memory_per_success_trend: UP" in summary
    assert "- memory_per_success_delta: +0.030" in summary
    assert "- stochastic_readout_trend: UP" in summary
    assert "- stochastic_readout_delta: +0.040" in summary
    assert "- efficiency_readiness_trend: NEW" in summary
    assert "- parameter_efficiency_status: PASS" in summary
    assert "- parameter_efficiency_score: 0.900" in summary
    assert "- average_quality_per_kparam: 12.000" in summary
    assert "- average_quality_per_mb: 40.000" in summary
    assert "- parameter_efficiency_trend: UP" in summary
    assert "- parameter_efficiency_delta: +0.050" in summary
    assert "- consolidation_readiness_status: PASS" in summary
    assert "- consolidation_readiness_score: 1.000" in summary
    assert "- consolidation_replay_recovery_integrity: 1.000" in summary
    assert "- consolidation_replay_upgrade_reindex_integrity: 1.000" in summary
    assert "- consolidation_memory_health_index_integrity: 1.000" in summary
    assert "- consolidation_replay_noise_resilience_integrity: 1.000" in summary
    assert "- consolidation_astro_modulation_stability: 1.000" in summary
    assert "- consolidation_delta_memory_residual_write_integrity_observed: 1.000" in summary
    assert "- consolidation_delta_memory_retention_gate_stability_observed: 1.000" in summary
    assert "- consolidation_delta_memory_context_recall_without_text_reinjection_observed: 1.000" in summary
    assert "- consolidation_delta_memory_state_budget_integrity_observed: 1.000" in summary
    assert "- consolidation_delta_memory_interference_guard_observed: 1.000" in summary
    assert "- consolidation_synaptic_tag_integrity_observed: 1.000" in summary
    assert "- consolidation_synaptic_tag_importance_score_observed: 1.000" in summary
    assert "- consolidation_synaptic_tag_replay_priority_observed: 1.000" in summary
    assert "- consolidation_synaptic_tag_pruning_candidate_observed: 1.000" in summary
    assert "- consolidation_synaptic_tag_state_budget_observed: 1.000" in summary
    assert "- consolidation_memory_phase_transition_integrity_observed: 1.000" in summary
    assert "- consolidation_memory_phase_retention_protection_observed: 1.000" in summary
    assert "- consolidation_memory_phase_plasticity_guard_observed: 1.000" in summary
    assert "- consolidation_memory_phase_overfixation_guard_observed: 1.000" in summary
    assert "- consolidation_memory_phase_state_budget_observed: 1.000" in summary
    assert "- consolidation_metabolic_budget_integrity_observed: 1.000" in summary
    assert "- consolidation_plasticity_reserve_integrity_observed: 1.000" in summary
    assert "- consolidation_structural_growth_bounded_observed: 1.000" in summary
    assert "- consolidation_pruning_reason_trace_observed: 1.000" in summary
    assert "- consolidation_resource_pressure_observed: 1.000" in summary
    assert "- consolidation_sleep_retention_observed: 1.000" in summary
    assert "- consolidation_latent_replay_noise_resilience_observed: 1.000" in summary
    assert "- consolidation_sleep_memory_health_observed: 1.000" in summary
    assert "- consolidation_latent_replay_counterfactual_branch_observed: 1.000" in summary
    assert "- consolidation_sleep_energy_budget_observed: 1.000" in summary
    assert "- consolidation_astro_structural_unlock_observed: 1.000" in summary
    assert "- consolidation_astro_structural_lock_observed: 1.000" in summary
    assert "- consolidation_astro_bounded_stdp_fallback_observed: 1.000" in summary
    assert "- consolidation_world_model_replay_policy_trace_observed: 1.000" in summary
    assert "- consolidation_astro_policy_state_budget_observed: 1.000" in summary
    assert "- consolidation_delta_memory_phase_retention_policy_observed: 1.000" in summary
    assert "- consolidation_delta_memory_crystal_retention_observed: 1.000" in summary
    assert "- consolidation_delta_memory_liquid_forget_observed: 1.000" in summary
    assert "- consolidation_delta_memory_astro_gate_alignment_observed: 1.000" in summary
    assert "- consolidation_delta_memory_policy_state_budget_observed: 1.000" in summary
    assert "- consolidation_delta_memory_multi_history_recall_observed: 1.000" in summary
    assert "- consolidation_delta_memory_multi_history_noise_resilience_observed: 1.000" in summary
    assert "- consolidation_delta_memory_multi_history_health_observed: 1.000" in summary
    assert "- consolidation_delta_memory_multi_history_manifold_guard_observed: 1.000" in summary
    assert "- consolidation_delta_memory_erase_write_decoupling_observed: 1.000" in summary
    assert "- consolidation_delta_memory_erase_preserves_stable_memory_observed: 1.000" in summary
    assert "- consolidation_delta_memory_write_commits_residual_observed: 1.000" in summary
    assert "- consolidation_readiness_trend: NEW" in summary
    assert "- cognitive_runtime_readiness_status: PASS" in summary
    assert "- cognitive_runtime_readiness_score: 1.000" in summary
    assert "- common_spike_space_integrity: 1.000" in summary
    assert "- temporal_compression_efficiency: 1.000" in summary
    assert "- dendritic_context_gate_stability: 1.000" in summary
    assert "- reverse_reasoning_trace_integrity: 1.000" in summary
    assert "- causal_candidate_trace_integrity: 1.000" in summary
    assert "- module_orchestration_integrity: 1.000" in summary
    assert "- counterfactual_lane_integrity: 1.000" in summary
    assert "- action_trace_observability: 1.000" in summary
    assert "- runtime_trace_replay_consistency: 1.000" in summary
    assert "- cognitive_manifold_trace_support_observed: 1.000" in summary
    assert "- cognitive_manifold_trace_recall_observed: 1.000" in summary
    assert "- cognitive_manifold_trace_scan_budget_observed: 1.000" in summary
    assert "- cognitive_manifold_trace_index_scan_reduction_observed: 1.000" in summary
    assert "- cognitive_manifold_trace_candidate_guard_observed: 1.000" in summary
    assert "- cognitive_delta_memory_steering_integrity_observed: 1.000" in summary
    assert "- cognitive_delta_memory_counterfactual_isolation_observed: 1.000" in summary
    assert "- cognitive_delta_memory_trace_observability_observed: 1.000" in summary
    assert "- cognitive_linear_snn_fusion_observed_policy: excluded_from_score_and_release_gate" in summary
    assert "- cognitive_linear_snn_fusion_trend_has_previous: False" in summary
    assert "- cognitive_linear_snn_fusion_trend_regression_count: 0" in summary
    assert "- cognitive_linear_snn_fusion_trend_release_gate_blocking: False" in summary
    assert "- cognitive_predictive_spike_entropy_reduction_observed: 1.000" in summary
    assert "- cognitive_phase_binding_coincidence_integrity_observed: 1.000" in summary
    assert "- cognitive_forward_only_local_update_stability_observed: 1.000" in summary
    assert "- cognitive_lejepa_linear_identifiability_proxy_observed: 1.000" in summary
    assert "- cognitive_lejepa_latent_whitening_health_observed: 1.000" in summary
    assert "- cognitive_lejepa_factor_disentanglement_observed: 1.000" in summary
    assert "- cognitive_lejepa_latent_planning_consistency_observed: 1.000" in summary
    assert "- cognitive_lejepa_positive_pair_alignment_observed: 1.000" in summary
    assert "- cognitive_stage_e_architecture_integration_observed_policy: excluded_from_score_and_release_gate" in summary
    assert "- cognitive_stage_e_architecture_integration_trend_has_previous: False" in summary
    assert "- cognitive_stage_e_architecture_integration_trend_regression_count: 0" in summary
    assert "- cognitive_stage_e_architecture_integration_trend_release_gate_blocking: False" in summary
    assert "- cognitive_plastic_submodel_registry_integrity_observed: 1.000" in summary
    assert "- cognitive_dynamic_submodel_route_integrity_observed: 1.000" in summary
    assert "- cognitive_submodel_relearning_trace_integrity_observed: 1.000" in summary
    assert "- cognitive_interpretable_submodel_concept_trace_observed: 1.000" in summary
    assert "- cognitive_runtime_submodel_route_action_grounding_observed: 1.000" in summary
    assert "- cognitive_runtime_submodel_counterfactual_route_separation_observed: 1.000" in summary
    assert "- cognitive_runtime_submodel_concept_trace_observed: 1.000" in summary
    assert "- cognitive_submodel_intervention_trace_integrity_observed: 1.000" in summary
    assert "- cognitive_submodel_ablation_effect_observed: 1.000" in summary
    assert "- cognitive_submodel_reactivation_recovery_observed: 1.000" in summary
    assert "- cognitive_submodel_credit_assignment_trace_integrity_observed: 1.000" in summary
    assert "- cognitive_submodel_credit_selectivity_observed: 1.000" in summary
    assert "- cognitive_submodel_credit_state_budget_observed: 1.000" in summary
    assert "- cognitive_runtime_submodel_local_credit_assignment_observed: 1.000" in summary
    assert "- cognitive_runtime_submodel_feedback_trace_observed: 1.000" in summary
    assert "- cognitive_submodel_structural_adaptation_trace_integrity_observed: 1.000" in summary
    assert "- cognitive_submodel_structural_growth_bounded_observed: 1.000" in summary
    assert "- cognitive_submodel_structural_pruning_observed: 1.000" in summary
    assert "- cognitive_submodel_scientific_hypothesis_trace_integrity_observed: 1.000" in summary
    assert "- cognitive_submodel_counterexample_revision_observed: 1.000" in summary
    assert "- cognitive_submodel_scientific_model_budget_observed: 1.000" in summary
    assert "- cognitive_submodel_hypothesis_bank_integrity_observed: 1.000" in summary
    assert "- cognitive_submodel_open_ended_selection_observed: 1.000" in summary
    assert "- cognitive_submodel_hypothesis_bank_budget_observed: 1.000" in summary
    assert "- cognitive_micro_turn_event_budget_observed: 1.000" in summary
    assert "- cognitive_foreground_background_context_handoff_observed: 1.000" in summary
    assert "- cognitive_interrupt_recovery_trace_observed: 1.000" in summary
    assert "- cognitive_simultaneous_stream_route_integrity_observed: 1.000" in summary
    assert "- cognitive_time_aligned_backchannel_policy_observed: 1.000" in summary
    assert "- cognitive_phase_assigned_submodel_route_observed: 1.000" in summary
    assert "- cognitive_uncertainty_bucket_specialization_observed: 1.000" in summary
    assert "- cognitive_denoising_correction_trace_integrity_observed: 1.000" in summary
    assert "- cognitive_block_independent_local_update_budget_observed: 1.000" in summary
    assert "- cognitive_runtime_readiness_trend: NEW" in summary
    assert "- phase5_manifold_candidate_miss_guard_observed: 1.000" in summary
    assert "- sara_inference: PASS score=1.000" in summary
    assert "- continual_consolidation: PASS score=1.000" in summary


def test_phase3_completion_gate_validator_reports_failed_checks():
    module = _load_script("phase3_completion_gate.py")
    report = {
        "phase3_completion": {
            "passed": False,
            "failed_checks": [
                "stage_d.minimum_requirements_passed",
                "focus.consolidation_readiness.passed",
            ],
        }
    }

    errors = module.validate_phase3_completion(report)

    assert any("did not pass" in item for item in errors)
    assert any("score is below" in item for item in errors)
    assert any("failed checks" in item for item in errors)


def test_phase4_completion_gate_validator_reports_failed_checks():
    module = _load_script("phase4_completion_gate.py")
    errors = module.validate_phase4_completion(
        phase3_report={"phase3_completion": {"passed": False}},
        phase4_report={
            "evaluator_name": "Phase4ScaleContinualBenchmark",
            "passed": False,
            "metrics": {
                "structural_plasticity_stability": 0.0,
                "hippocampal_transfer_integrity": 0.0,
                "scale_out_retention_integrity": 0.0,
                "continual_drift_recovery_integrity": 0.0,
            },
        },
    )

    assert any("Phase 3 completion gate is not passed." in item for item in errors)
    assert any("Phase 4 benchmark did not pass." in item for item in errors)
    assert any("structural_plasticity_stability" in item for item in errors)
    assert any("quality_metrics" in item for item in errors)


def test_phase4_completion_gate_validator_rejects_quality_metric_regression():
    module = _load_script("phase4_completion_gate.py")
    errors = module.validate_phase4_completion(
        phase3_report={"phase3_completion": {"passed": True}},
        phase4_report={
            "evaluator_name": "Phase4ScaleContinualBenchmark",
            "passed": True,
            "metrics": {
                "structural_plasticity_stability": 1.0,
                "hippocampal_transfer_integrity": 1.0,
                "scale_out_retention_integrity": 1.0,
                "continual_drift_recovery_integrity": 1.0,
            },
            "threshold_results": {
                "structural_plasticity_stability": True,
                "hippocampal_transfer_integrity": True,
                "scale_out_retention_integrity": True,
                "continual_drift_recovery_integrity": True,
            },
            "quality_metrics": {
                "structural_synapse_ratio": 1.0,
                "structural_per_context_non_empty": 1.0,
                "hippocampal_after_top_score": 0.5,
                "hippocampal_score_retention_ratio": 1.0,
                "scale_out_retention_rate": 0.5,
                "scale_out_average_query_ms": 80.0,
                "continual_baseline_recovered": 1.0,
                "continual_drift_observed": 1.0,
            },
        },
    )

    assert any("retention rate" in item for item in errors)
    assert any("query latency" in item for item in errors)


def test_phase3_accuracy_suite_builds_retrieval_hygiene_focus_trend_against_previous_report():
    module = _load_script("phase3_accuracy_suite.py")
    current_focus = {
        "retrieval_hygiene": {"score": 0.82, "passed": True},
    }
    previous_report = {
        "focus_summary": {
            "retrieval_hygiene": {"score": 0.74, "passed": True},
        }
    }

    focus_trend = module._build_focus_trend(current_focus, previous_report)

    assert focus_trend["retrieval_hygiene"]["status"] == "UP"
    assert abs(focus_trend["retrieval_hygiene"]["delta"] - 0.08) < 1e-9


def test_phase3_tracking_detects_metric_regression():
    previous_report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 0.95,
        "component_reports": {
            "agent_dialogue": {
                "overall_score": 0.9,
                "metrics": {"response_keyword_recall": 0.8},
            },
        },
    }
    current_report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 0.90,
        "component_reports": {
            "agent_dialogue": {
                "overall_score": 0.85,
                "metrics": {"response_keyword_recall": 0.7},
            },
        },
    }

    trend = build_phase3_trend(current_report=current_report, previous_report=previous_report)

    assert trend["has_previous"] is True
    assert trend["regression_count"] >= 1
    assert any(
        item["metric"] == "agent_dialogue.response_keyword_recall"
        for item in trend["regressions"]
    )
    assert trend["gate_regression_count"] >= 1


def test_phase3_tracking_marks_raw_parameter_efficiency_as_observed_only_regression():
    previous_report = {
        "suite_name": "Phase3AccuracySuite",
        "component_reports": {
            "parameter_efficiency": {
                "metrics": {
                    "quality_per_mb_score": 1.0,
                    "average_quality_per_mb": 30.0,
                },
            },
        },
    }
    current_report = {
        "suite_name": "Phase3AccuracySuite",
        "component_reports": {
            "parameter_efficiency": {
                "metrics": {
                    "quality_per_mb_score": 1.0,
                    "average_quality_per_mb": 29.0,
                },
            },
        },
    }

    trend = build_phase3_trend(current_report=current_report, previous_report=previous_report)

    assert trend["regression_count"] == 1
    assert trend["gate_regression_count"] == 0
    assert trend["gate_regressions"] == []


def test_phase3_tracking_marks_latency_score_as_observed_only_regression():
    previous_report = {
        "suite_name": "Phase3AccuracySuite",
        "component_reports": {
            "energy_efficiency": {
                "metrics": {"bounded_latency_score": 0.96},
            },
        },
        "focus_summary": {
            "efficiency_readiness": {
                "metrics": {
                    "energy_efficiency.bounded_latency_score": 0.96,
                },
            },
        },
    }
    current_report = {
        "suite_name": "Phase3AccuracySuite",
        "component_reports": {
            "energy_efficiency": {
                "metrics": {"bounded_latency_score": 0.90},
            },
        },
        "focus_summary": {
            "efficiency_readiness": {
                "metrics": {
                    "energy_efficiency.bounded_latency_score": 0.90,
                },
            },
        },
    }

    trend = build_phase3_trend(current_report=current_report, previous_report=previous_report)

    assert trend["regression_count"] == 2
    assert trend["gate_regression_count"] == 0
    assert trend["gate_regressions"] == []


def test_phase3_tracking_marks_manifold_metrics_as_observed_only_regressions():
    previous_report = {
        "suite_name": "Phase3AccuracySuite",
        "component_reports": {
            "continual_consolidation": {
                "metrics": {
                    "manifold_capacity_pressure_recall_observed": 1.0,
                    "manifold_capacity_pressure_scan_reduction_observed": 0.889,
                    "manifold_replay_refresh_retention_observed": 1.0,
                    "manifold_replay_refresh_eviction_integrity_observed": 1.0,
                },
            },
            "phase5_predictive_coding": {
                "metrics": {
                    "manifold_index_scan_reduction": 1.0,
                    "causal_route_sparsity": 1.0,
                    "withheld_trajectory_recall": 1.0,
                },
            },
        },
    }
    current_report = {
        "suite_name": "Phase3AccuracySuite",
        "component_reports": {
            "continual_consolidation": {
                "metrics": {
                    "manifold_capacity_pressure_recall_observed": 0.90,
                    "manifold_capacity_pressure_scan_reduction_observed": 0.70,
                    "manifold_replay_refresh_retention_observed": 0.80,
                    "manifold_replay_refresh_eviction_integrity_observed": 0.75,
                },
            },
            "phase5_predictive_coding": {
                "metrics": {
                    "manifold_index_scan_reduction": 0.80,
                    "causal_route_sparsity": 0.75,
                    "withheld_trajectory_recall": 0.85,
                },
            },
        },
    }

    trend = build_phase3_trend(current_report=current_report, previous_report=previous_report)

    assert trend["regression_count"] == 7
    assert trend["gate_regression_count"] == 0
    assert trend["gate_regressions"] == []


def test_phase3_tracking_treats_small_deltas_as_unchanged():
    previous_report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 0.95,
        "component_reports": {
            "agent_dialogue": {
                "overall_score": 0.80,
                "metrics": {"retrieval_stability": 0.60},
            },
        },
        "focus_summary": {
            "retrieval_hygiene": {
                "score": 0.60,
                "metrics": {
                    "agent_dialogue.retrieval_stability": 0.60,
                },
            },
        },
    }
    current_report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 0.94,
        "component_reports": {
            "agent_dialogue": {
                "overall_score": 0.781,
                "metrics": {"retrieval_stability": 0.579},
            },
        },
        "focus_summary": {
            "retrieval_hygiene": {
                "score": 0.581,
                "metrics": {
                    "agent_dialogue.retrieval_stability": 0.579,
                },
            },
        },
    }

    trend = build_phase3_trend(current_report=current_report, previous_report=previous_report)

    assert trend["regression_count"] == 0
    assert "agent_dialogue.retrieval_stability" in trend["unchanged"]


def test_phase3_tracking_extracts_cognitive_manifold_trace_observations():
    report = {
        "component_reports": {
            "cognitive_runtime": {
                "metrics": {
                    "manifold_trace_support_observed": 1.0,
                    "manifold_trace_recall_observed": 0.75,
                    "manifold_trace_scan_budget_observed": 1.0,
                    "manifold_trace_index_scan_reduction_observed": 1.0,
                    "manifold_trace_candidate_guard_observed": 0.5,
                },
            },
        },
    }

    metrics = extract_cognitive_manifold_trace_metrics(report)

    assert tuple(metrics) == COGNITIVE_MANIFOLD_TRACE_METRIC_NAMES
    assert phase3_component_metrics(report, "cognitive_runtime")[
        "manifold_trace_recall_observed"
    ] == 0.75
    assert metrics == {
        "manifold_trace_support_observed": 1.0,
        "manifold_trace_recall_observed": 0.75,
        "manifold_trace_scan_budget_observed": 1.0,
        "manifold_trace_index_scan_reduction_observed": 1.0,
        "manifold_trace_candidate_guard_observed": 0.5,
    }


def test_phase3_tracking_defaults_missing_cognitive_manifold_trace_observations():
    assert extract_cognitive_manifold_trace_metrics({"component_reports": {}}) == {
        "manifold_trace_support_observed": 0.0,
        "manifold_trace_recall_observed": 0.0,
        "manifold_trace_scan_budget_observed": 0.0,
        "manifold_trace_index_scan_reduction_observed": 0.0,
        "manifold_trace_candidate_guard_observed": 0.0,
    }


def test_phase3_tracking_extracts_cognitive_delta_memory_observations():
    report = {
        "component_reports": {
            "cognitive_runtime": {
                "metrics": {
                    "delta_memory_steering_integrity_observed": 1.0,
                    "delta_memory_counterfactual_isolation_observed": 0.75,
                    "delta_memory_trace_observability_observed": 0.5,
                },
            },
        },
    }

    metrics = extract_cognitive_delta_memory_metrics(report)

    assert tuple(metrics) == COGNITIVE_DELTA_MEMORY_METRIC_NAMES
    assert metrics == {
        "delta_memory_steering_integrity_observed": 1.0,
        "delta_memory_counterfactual_isolation_observed": 0.75,
        "delta_memory_trace_observability_observed": 0.5,
    }


def test_phase3_tracking_defaults_missing_cognitive_delta_memory_observations():
    assert extract_cognitive_delta_memory_metrics({"component_reports": {}}) == {
        "delta_memory_steering_integrity_observed": 0.0,
        "delta_memory_counterfactual_isolation_observed": 0.0,
        "delta_memory_trace_observability_observed": 0.0,
    }


def test_phase3_tracking_extracts_cognitive_linear_snn_fusion_observations():
    report = {
        "component_reports": {
            "cognitive_runtime": {
                "metrics": {
                    "predictive_spike_entropy_reduction_observed": 1.0,
                    "phase_binding_coincidence_integrity_observed": 0.75,
                    "forward_only_local_update_stability_observed": 0.5,
                    "lejepa_linear_identifiability_proxy_observed": 1.0,
                    "lejepa_latent_whitening_health_observed": 1.0,
                    "lejepa_factor_disentanglement_observed": 1.0,
                    "lejepa_latent_planning_consistency_observed": 1.0,
                    "lejepa_positive_pair_alignment_observed": 1.0,
                },
            },
        },
    }

    metrics = extract_cognitive_linear_snn_fusion_metrics(report)

    assert tuple(metrics) == COGNITIVE_LINEAR_SNN_FUSION_METRIC_NAMES
    assert metrics == {
        "predictive_spike_entropy_reduction_observed": 1.0,
        "phase_binding_coincidence_integrity_observed": 0.75,
        "forward_only_local_update_stability_observed": 0.5,
        "lejepa_linear_identifiability_proxy_observed": 1.0,
        "lejepa_latent_whitening_health_observed": 1.0,
        "lejepa_factor_disentanglement_observed": 1.0,
        "lejepa_latent_planning_consistency_observed": 1.0,
        "lejepa_positive_pair_alignment_observed": 1.0,
    }


def test_phase3_tracking_defaults_missing_cognitive_linear_snn_fusion_observations():
    assert extract_cognitive_linear_snn_fusion_metrics({"component_reports": {}}) == {
        "predictive_spike_entropy_reduction_observed": 0.0,
        "phase_binding_coincidence_integrity_observed": 0.0,
        "forward_only_local_update_stability_observed": 0.0,
        "lejepa_linear_identifiability_proxy_observed": 0.0,
        "lejepa_latent_whitening_health_observed": 0.0,
        "lejepa_factor_disentanglement_observed": 0.0,
        "lejepa_latent_planning_consistency_observed": 0.0,
        "lejepa_positive_pair_alignment_observed": 0.0,
    }


def test_phase3_tracking_extracts_stage_e_architecture_integration_observations():
    report = {
        "component_reports": {
            "cognitive_runtime": {
                "metrics": {
                    "micro_turn_event_budget_observed": 1.0,
                    "foreground_background_context_handoff_observed": 0.9,
                    "interrupt_recovery_trace_observed": 0.8,
                    "simultaneous_stream_route_integrity_observed": 0.7,
                    "time_aligned_backchannel_policy_observed": 0.6,
                    "phase_assigned_submodel_route_observed": 1.0,
                    "uncertainty_bucket_specialization_observed": 0.95,
                    "denoising_correction_trace_integrity_observed": 0.85,
                    "block_independent_local_update_budget_observed": 0.75,
                },
            },
        },
    }

    metrics = extract_cognitive_stage_e_architecture_integration_metrics(report)

    assert tuple(metrics) == COGNITIVE_STAGE_E_ARCHITECTURE_INTEGRATION_METRIC_NAMES
    assert metrics == {
        "micro_turn_event_budget_observed": 1.0,
        "foreground_background_context_handoff_observed": 0.9,
        "interrupt_recovery_trace_observed": 0.8,
        "simultaneous_stream_route_integrity_observed": 0.7,
        "time_aligned_backchannel_policy_observed": 0.6,
        "phase_assigned_submodel_route_observed": 1.0,
        "uncertainty_bucket_specialization_observed": 0.95,
        "denoising_correction_trace_integrity_observed": 0.85,
        "block_independent_local_update_budget_observed": 0.75,
    }


def test_phase3_tracking_defaults_missing_stage_e_architecture_integration_observations():
    assert extract_cognitive_stage_e_architecture_integration_metrics({"component_reports": {}}) == {
        "micro_turn_event_budget_observed": 0.0,
        "foreground_background_context_handoff_observed": 0.0,
        "interrupt_recovery_trace_observed": 0.0,
        "simultaneous_stream_route_integrity_observed": 0.0,
        "time_aligned_backchannel_policy_observed": 0.0,
        "phase_assigned_submodel_route_observed": 0.0,
        "uncertainty_bucket_specialization_observed": 0.0,
        "denoising_correction_trace_integrity_observed": 0.0,
        "block_independent_local_update_budget_observed": 0.0,
    }


def test_phase3_tracking_detects_stage_e_architecture_integration_observed_regression_without_gate_block():
    previous_report = {
        "component_reports": {
            "cognitive_runtime": {
                "metrics": {
                    "micro_turn_event_budget_observed": 1.0,
                    "foreground_background_context_handoff_observed": 1.0,
                    "interrupt_recovery_trace_observed": 1.0,
                    "simultaneous_stream_route_integrity_observed": 1.0,
                    "time_aligned_backchannel_policy_observed": 1.0,
                    "phase_assigned_submodel_route_observed": 1.0,
                    "uncertainty_bucket_specialization_observed": 1.0,
                    "denoising_correction_trace_integrity_observed": 1.0,
                    "block_independent_local_update_budget_observed": 1.0,
                },
            },
        },
    }
    current_report = {
        "component_reports": {
            "cognitive_runtime": {
                "metrics": {
                    "micro_turn_event_budget_observed": 1.0,
                    "foreground_background_context_handoff_observed": 0.5,
                    "interrupt_recovery_trace_observed": 1.0,
                    "simultaneous_stream_route_integrity_observed": 1.0,
                    "time_aligned_backchannel_policy_observed": 1.0,
                    "phase_assigned_submodel_route_observed": 1.0,
                    "uncertainty_bucket_specialization_observed": 1.0,
                    "denoising_correction_trace_integrity_observed": 0.75,
                    "block_independent_local_update_budget_observed": 1.0,
                },
            },
        },
    }

    trend = build_cognitive_stage_e_architecture_integration_observed_trend(
        current_report=current_report,
        previous_report=previous_report,
    )

    assert trend["observed_only"] is True
    assert trend["release_gate_blocking"] is False
    assert trend["regression_count"] == 2
    assert {
        item["metric"] for item in trend["regressions"]
    } == {
        "foreground_background_context_handoff_observed",
        "denoising_correction_trace_integrity_observed",
    }


def test_phase3_tracking_compacts_neuromorphic_profile_trend_details():
    trend = {
        "missing_profiles": ["spinnaker"],
        "regressions": [
            {"profile": "akida", "kind": "compatibility_regression"},
            {
                "profile": "akida",
                "kind": "check_regression",
                "check": "low_precision_weight_ok",
            },
        ],
        "policy_changes": [
            {
                "profile": "lava",
                "previous": "native_online_update",
                "current": "freeze_state_for_inference_profile",
            }
        ],
    }

    compact = compact_neuromorphic_profile_trend(trend)

    assert compact["regression_details"] == [
        "akida:compatibility_regression",
        "akida:check_regression:low_precision_weight_ok",
        "spinnaker:missing_profile",
    ]
    assert compact["policy_change_details"] == [
        "lava:native_online_update->freeze_state_for_inference_profile"
    ]
    assert compact["regression_detail_line"] == (
        "akida:compatibility_regression,"
        "akida:check_regression:low_precision_weight_ok,"
        "spinnaker:missing_profile"
    )


def test_phase3_tracking_detects_linear_snn_fusion_observed_regression_without_gate_block():
    previous_report = {
        "component_reports": {
            "cognitive_runtime": {
                "metrics": {
                    "predictive_spike_entropy_reduction_observed": 1.0,
                    "phase_binding_coincidence_integrity_observed": 1.0,
                    "forward_only_local_update_stability_observed": 1.0,
                },
            },
        },
    }
    current_report = {
        "component_reports": {
            "cognitive_runtime": {
                "metrics": {
                    "predictive_spike_entropy_reduction_observed": 0.5,
                    "phase_binding_coincidence_integrity_observed": 1.0,
                    "forward_only_local_update_stability_observed": 0.75,
                },
            },
        },
    }

    trend = build_cognitive_linear_snn_fusion_observed_trend(
        current_report=current_report,
        previous_report=previous_report,
    )

    assert trend["observed_only"] is True
    assert trend["release_gate_blocking"] is False
    assert trend["regression_count"] == 2
    assert {
        item["metric"] for item in trend["regressions"]
    } == {
        "predictive_spike_entropy_reduction_observed",
        "forward_only_local_update_stability_observed",
    }


def test_phase3_accuracy_suite_focus_trend_treats_small_delta_as_flat():
    module = _load_script("phase3_accuracy_suite.py")
    current_focus = {
        "retrieval_hygiene": {
            "score": 0.59,
        }
    }
    previous_report = {
        "focus_summary": {
            "retrieval_hygiene": {
                "score": 0.60,
            }
        }
    }

    focus_trend = module._build_focus_trend(current_focus, previous_report)

    assert focus_trend["retrieval_hygiene"]["status"] == "FLAT"
    assert abs(focus_trend["retrieval_hygiene"]["delta"] + 0.01) < 1e-9


def test_phase3_tracking_flattens_focus_summary_metrics():
    from sara_engine.evaluation.phase3_tracking import flatten_phase3_metrics

    report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 1.0,
        "component_reports": {},
        "focus_summary": {
            "few_shot": {
                "score": 1.0,
                "metrics": {
                    "sara_inference.few_shot_accuracy": 1.0,
                },
            },
            "continual": {
                "score": 0.9,
                "metrics": {
                    "spiking_llm.long_horizon_memory_retention": 0.9,
                },
            },
        },
    }

    flattened = flatten_phase3_metrics(report)

    assert flattened["focus.few_shot.score"] == 1.0
    assert flattened["focus.few_shot.sara_inference.few_shot_accuracy"] == 1.0
    assert flattened["focus.continual.score"] == 0.9
    assert flattened["focus.continual.spiking_llm.long_horizon_memory_retention"] == 0.9


def test_phase3_accuracy_suite_persists_managed_history():
    module = _load_script("phase3_accuracy_suite.py")
    history_path = workspace_path("tests", "phase3_accuracy_history_test.json")
    if os.path.exists(history_path):
        os.remove(history_path)

    first_report = module.run_phase3_accuracy_suite(
        history_path=history_path,
        persist_history=True,
        history_limit=2,
    )
    second_report = module.run_phase3_accuracy_suite(
        history_path=history_path,
        persist_history=True,
        history_limit=2,
    )
    history = load_phase3_history(history_path)

    assert first_report["history_length"] == 1
    assert second_report["history_length"] == 2
    assert len(history) == 2
    assert history[-1]["suite_name"] == "Phase3AccuracySuite"


def test_phase3_history_limit_trims_old_entries():
    history_path = workspace_path("tests", "phase3_accuracy_history_trim_test.json")
    if os.path.exists(history_path):
        os.remove(history_path)

    for idx in range(3):
        append_phase3_history(
            history_path=history_path,
            report={"suite_name": "Phase3AccuracySuite", "overall_score": float(idx)},
            max_entries=2,
        )

    history = load_phase3_history(history_path)

    assert len(history) == 2
    assert history[0]["overall_score"] == 1.0
    assert history[1]["overall_score"] == 2.0


def test_stage_b_promotion_readiness_recommends_after_required_streak():
    module = _load_script("phase3_accuracy_suite.py")
    current = {"promotion_candidate_ready": True}
    history = [
        {"stage_b_readiness": {"promotion_candidate_ready": True}},
        {"stage_b_readiness": {"promotion_candidate_ready": True}},
    ]
    readiness = module._build_stage_b_promotion_readiness(
        current,
        history=history,
        required_streak=3,
    )
    assert readiness["consecutive_passes"] == 3
    assert readiness["required_streak"] == 3
    assert readiness["recommended"] is True


def test_stage_b_promotion_readiness_respects_custom_required_streak():
    module = _load_script("phase3_accuracy_suite.py")
    current = {"promotion_candidate_ready": True}
    history = [
        {"stage_b_readiness": {"promotion_candidate_ready": True}},
        {"stage_b_readiness": {"promotion_candidate_ready": True}},
    ]
    readiness = module._build_stage_b_promotion_readiness(
        current,
        history=history,
        required_streak=4,
    )
    assert readiness["consecutive_passes"] == 3
    assert readiness["required_streak"] == 4
    assert readiness["recommended"] is False


def test_stage_b_rlm_observation_promotion_readiness_recommends_after_required_streak():
    module = _load_script("phase3_accuracy_suite.py")
    current = {"rlm_observation_candidate_ready": True}
    history = [
        {"stage_b_readiness": {"rlm_observation_candidate_ready": True}},
        {"stage_b_readiness": {"rlm_observation_candidate_ready": True}},
    ]
    readiness = module._build_stage_b_rlm_observation_promotion_readiness(
        current,
        history=history,
        required_streak=3,
    )
    assert readiness["consecutive_passes"] == 3
    assert readiness["required_streak"] == 3
    assert readiness["recommended"] is True
    assert readiness["promoted_to_minimum"] is False


def test_stage_d_delta_memory_promotion_readiness_recommends_after_required_streak():
    module = _load_script("phase3_accuracy_suite.py")
    current = {"delta_memory_candidate_ready": True}
    history = [
        {"stage_d_readiness": {"delta_memory_candidate_ready": True}},
        {"stage_d_readiness": {"delta_memory_candidate_ready": True}},
    ]
    readiness = module._build_stage_d_delta_memory_promotion_readiness(
        current,
        history=history,
        required_streak=3,
    )
    assert readiness["consecutive_passes"] == 3
    assert readiness["required_streak"] == 3
    assert readiness["recommended"] is True
    assert readiness["promoted_to_minimum"] is False


def test_stage_d_acceptance_candidate_stability_recommends_after_required_streak():
    module = _load_script("phase3_accuracy_suite.py")
    current = {"acceptance_candidates_ready": True}
    history = [
        {"stage_d_readiness": {"acceptance_candidates_ready": True}},
        {"stage_d_readiness": {"acceptance_candidates_ready": True}},
    ]
    readiness = module._build_stage_d_acceptance_candidate_stability(
        current,
        history=history,
        required_streak=3,
    )
    assert readiness["consecutive_passes"] == 3
    assert readiness["required_streak"] == 3
    assert readiness["recommended"] is True


def test_stage_e_observed_acceptance_candidate_stability_recommends_after_required_streak():
    module = _load_script("phase3_accuracy_suite.py")
    current = {"observed_acceptance_candidates_ready": True}
    history = [
        {"stage_e_readiness": {"observed_acceptance_candidates_ready": True}},
        {"stage_e_readiness": {"observed_acceptance_candidates_ready": True}},
    ]
    readiness = module._build_stage_e_acceptance_candidate_stability(
        current,
        history=history,
        required_streak=3,
    )
    assert readiness["consecutive_passes"] == 3
    assert readiness["required_streak"] == 3
    assert readiness["recommended"] is True


def test_stage_d_delta_memory_candidate_failures_are_structured():
    module = _load_script("phase3_accuracy_suite.py")
    report = {
        "component_reports": {
            "continual_consolidation": {
                "metrics": {
                    "replay_recovery_integrity": 1.0,
                    "long_horizon_consolidation_retention": 1.0,
                    "counterfactual_replay_selection_integrity": 1.0,
                    "replay_upgrade_reindex_integrity": 1.0,
                    "memory_health_index_integrity": 1.0,
                    "replay_noise_resilience_integrity": 1.0,
                    "astro_modulation_stability": 1.0,
                    "delta_memory_phase_retention_policy_observed": 1.0,
                },
            },
        },
    }

    readiness = module._build_stage_d_readiness(report)

    assert readiness["minimum_requirements_passed"] is True
    assert readiness["delta_memory_candidate_ready"] is False
    assert readiness["delta_memory_candidate_failure_count"] >= 1
    assert readiness["acceptance_candidates_ready"] is False
    assert readiness["acceptance_candidate_failure_count"] >= 1
    assert readiness["acceptance_candidate_failures"][0]["check"].startswith("metric.")
    assert readiness["acceptance_candidate_failures"][0]["threshold"] == 1.0
    assert any(
        item["metric"] == "delta_memory_erase_write_decoupling_observed"
        for item in readiness["acceptance_candidate_failures"]
    )
    assert readiness["delta_memory_candidate_failures"][0]["check"].startswith("metric.delta_memory_")
    assert readiness["delta_memory_candidate_failures"][0]["threshold"] == 1.0


def test_stage_e_observed_acceptance_candidate_failures_are_structured():
    module = _load_script("phase3_accuracy_suite.py")
    metrics = {
        metric_name: 1.0
        for metric_name in STAGE_E_OBSERVED_ACCEPTANCE_CANDIDATE_METRIC_NAMES
    }
    metrics.update({
        "common_spike_space_integrity": 1.0,
        "temporal_compression_efficiency": 1.0,
        "modality_temporal_budget_integrity": 1.0,
        "dendritic_context_gate_stability": 1.0,
        "spiking_hjepa_latent_transition": 1.0,
        "reverse_reasoning_trace_integrity": 1.0,
        "causal_candidate_trace_integrity": 1.0,
        "module_orchestration_integrity": 1.0,
        "counterfactual_lane_integrity": 1.0,
        "action_trace_observability": 1.0,
        "runtime_trace_replay_consistency": 1.0,
        "micro_turn_event_budget_observed": 0.0,
    })
    report = {"component_reports": {"cognitive_runtime": {"metrics": metrics}}}

    readiness = module._build_stage_e_readiness(report)

    assert readiness["minimum_requirements_passed"] is True
    assert readiness["observed_acceptance_candidates_ready"] is False
    assert readiness["observed_acceptance_candidate_failure_count"] == 1
    failure = readiness["observed_acceptance_candidate_failures"][0]
    assert failure["check"] == "metric.micro_turn_event_budget_observed"
    assert failure["metric"] == "micro_turn_event_budget_observed"
    assert failure["value"] == 0.0
    assert failure["threshold"] == 1.0
    assert failure["description"] == "micro turn event budget observed"
