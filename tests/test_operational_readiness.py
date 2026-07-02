import importlib.util
import json
import os


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


def _build_phase3_report(passed: bool = True):
    return {
        "suite_name": "Phase3AccuracySuite",
        "passed": bool(passed),
        "overall_score": 0.98 if passed else 0.20,
        "component_reports": {
            "agent_dialogue": {"passed": bool(passed), "overall_score": 1.0},
            "sara_inference": {"passed": bool(passed), "overall_score": 1.0},
            "spiking_llm": {"passed": bool(passed), "overall_score": 1.0},
            "task_switch_adaptation": {"passed": bool(passed), "overall_score": 1.0},
            "future_state_consistency": {"passed": bool(passed), "overall_score": 1.0},
            "energy_efficiency": {
                "passed": bool(passed),
                "overall_score": 1.0,
                "metrics": {
                    "neuromorphic_profile_history_regression_observed": 1.0 if passed else 0.0,
                    "neuromorphic_profile_report_integrity_observed": 1.0 if passed else 0.0,
                    "neuromorphic_backend_profile_compatibility_observed": 1.0 if passed else 0.0,
                    "neuromorphic_stage_e_state_trace_ir_observed": 1.0 if passed else 0.0,
                    "neuromorphic_stage_e_routing_hint_coverage_observed": 1.0 if passed else 0.0,
                    "neuromorphic_stage_e_online_update_policy_observed": 1.0 if passed else 0.0,
                    "neuromorphic_stage_e_event_budget_observed": 1.0 if passed else 0.0,
                },
                "neuromorphic_profile_trend": {
                    "has_previous": True,
                    "regression_count": 0 if passed else 1,
                    "policy_change_count": 0,
                    "new_profiles": [],
                    "missing_profiles": [],
                },
            },
            "continual_consolidation": {
                "passed": bool(passed),
                "overall_score": 1.0,
                "metrics": {
                    "manifold_continual_retention_observed": 1.0 if passed else 0.0,
                    "manifold_trajectory_case_coverage_observed": 1.0 if passed else 0.0,
                    "manifold_average_case_recall_observed": 1.0 if passed else 0.0,
                    "manifold_scan_budget_integrity_observed": 1.0 if passed else 0.0,
                    "manifold_indexed_candidate_integrity_observed": 1.0 if passed else 0.0,
                    "manifold_index_scan_reduction_observed": 1.0 if passed else 0.0,
                    "manifold_capacity_pressure_recall_observed": 1.0 if passed else 0.0,
                    "manifold_capacity_pressure_scan_reduction_observed": 0.889 if passed else 0.0,
                    "manifold_replay_refresh_retention_observed": 1.0 if passed else 0.0,
                    "manifold_replay_refresh_eviction_integrity_observed": 1.0 if passed else 0.0,
                    "synaptic_tag_integrity_observed": 1.0 if passed else 0.0,
                    "memory_phase_transition_integrity_observed": 1.0 if passed else 0.0,
                    "metabolic_budget_integrity_observed": 1.0 if passed else 0.0,
                    "sleep_consolidation_retention_observed": 1.0 if passed else 0.0,
                    "astro_structural_lock_observed": 1.0 if passed else 0.0,
                    "delta_memory_phase_retention_policy_observed": 1.0 if passed else 0.0,
                    "delta_memory_crystal_retention_observed": 1.0 if passed else 0.0,
                    "delta_memory_multi_history_recall_observed": 1.0 if passed else 0.0,
                    "delta_memory_multi_history_health_observed": 1.0 if passed else 0.0,
                    "delta_memory_erase_write_decoupling_observed": 1.0 if passed else 0.0,
                    "delta_memory_erase_preserves_stable_memory_observed": 1.0 if passed else 0.0,
                    "delta_memory_write_commits_residual_observed": 1.0 if passed else 0.0,
                },
            },
            "cognitive_runtime": {
                "passed": bool(passed),
                "overall_score": 1.0,
                "metrics": {
                    "manifold_trace_support_observed": 1.0 if passed else 0.0,
                    "manifold_trace_recall_observed": 1.0 if passed else 0.0,
                    "manifold_trace_scan_budget_observed": 1.0 if passed else 0.0,
                    "manifold_trace_index_scan_reduction_observed": 1.0 if passed else 0.0,
                    "manifold_trace_candidate_guard_observed": 1.0 if passed else 0.0,
                    "delta_memory_steering_integrity_observed": 1.0 if passed else 0.0,
                    "delta_memory_counterfactual_isolation_observed": 1.0 if passed else 0.0,
                    "delta_memory_trace_observability_observed": 1.0 if passed else 0.0,
                    "predictive_spike_entropy_reduction_observed": 1.0 if passed else 0.0,
                    "phase_binding_coincidence_integrity_observed": 1.0 if passed else 0.0,
                    "forward_only_local_update_stability_observed": 1.0 if passed else 0.0,
                    "lejepa_linear_identifiability_proxy_observed": 1.0 if passed else 0.0,
                    "lejepa_latent_whitening_health_observed": 1.0 if passed else 0.0,
                    "lejepa_factor_disentanglement_observed": 1.0 if passed else 0.0,
                    "lejepa_latent_planning_consistency_observed": 1.0 if passed else 0.0,
                    "lejepa_positive_pair_alignment_observed": 1.0 if passed else 0.0,
                    "plastic_submodel_registry_integrity_observed": 1.0 if passed else 0.0,
                    "dynamic_submodel_route_integrity_observed": 1.0 if passed else 0.0,
                    "submodel_relearning_trace_integrity_observed": 1.0 if passed else 0.0,
                    "interpretable_submodel_concept_trace_observed": 1.0 if passed else 0.0,
                    "runtime_submodel_route_action_grounding_observed": 1.0 if passed else 0.0,
                    "runtime_submodel_counterfactual_route_separation_observed": 1.0 if passed else 0.0,
                    "runtime_submodel_concept_trace_observed": 1.0 if passed else 0.0,
                    "submodel_intervention_trace_integrity_observed": 1.0 if passed else 0.0,
                    "submodel_ablation_effect_observed": 1.0 if passed else 0.0,
                    "submodel_reactivation_recovery_observed": 1.0 if passed else 0.0,
                    "submodel_credit_assignment_trace_integrity_observed": 1.0 if passed else 0.0,
                    "submodel_credit_selectivity_observed": 1.0 if passed else 0.0,
                    "submodel_credit_state_budget_observed": 1.0 if passed else 0.0,
                    "runtime_submodel_local_credit_assignment_observed": 1.0 if passed else 0.0,
                    "runtime_submodel_feedback_trace_observed": 1.0 if passed else 0.0,
                    "submodel_structural_adaptation_trace_integrity_observed": 1.0 if passed else 0.0,
                    "submodel_structural_growth_bounded_observed": 1.0 if passed else 0.0,
                    "submodel_structural_pruning_observed": 1.0 if passed else 0.0,
                    "submodel_scientific_hypothesis_trace_integrity_observed": 1.0 if passed else 0.0,
                    "submodel_counterexample_revision_observed": 1.0 if passed else 0.0,
                    "submodel_scientific_model_budget_observed": 1.0 if passed else 0.0,
                    "submodel_hypothesis_bank_integrity_observed": 1.0 if passed else 0.0,
                    "submodel_open_ended_selection_observed": 1.0 if passed else 0.0,
                    "submodel_hypothesis_bank_budget_observed": 1.0 if passed else 0.0,
                    "micro_turn_event_budget_observed": 1.0 if passed else 0.0,
                    "foreground_background_context_handoff_observed": 1.0 if passed else 0.0,
                    "interrupt_recovery_trace_observed": 1.0 if passed else 0.0,
                    "simultaneous_stream_route_integrity_observed": 1.0 if passed else 0.0,
                    "time_aligned_backchannel_policy_observed": 1.0 if passed else 0.0,
                    "phase_assigned_submodel_route_observed": 1.0 if passed else 0.0,
                    "uncertainty_bucket_specialization_observed": 1.0 if passed else 0.0,
                    "denoising_correction_trace_integrity_observed": 1.0 if passed else 0.0,
                    "block_independent_local_update_budget_observed": 1.0 if passed else 0.0,
                },
            },
            "phase5_predictive_coding": {
                "passed": bool(passed),
                "overall_score": 1.0,
                "metrics": {
                    "manifold_candidate_miss_guard": 1.0 if passed else 0.0,
                },
            },
        },
        "trend": {"regression_count": 0},
        "linear_snn_fusion_observed_trend": {
            "has_previous": bool(passed),
            "regression_count": 0 if passed else 1,
            "release_gate_blocking": False,
        },
        "stage_e_architecture_integration_observed_trend": {
            "has_previous": bool(passed),
            "regression_count": 0 if passed else 1,
            "release_gate_blocking": False,
        },
        "focus_summary": {
            "few_shot": {"passed": bool(passed)},
            "continual": {"passed": bool(passed)},
            "retrieval_hygiene": {"passed": bool(passed)},
            "adaptive_readiness": {"passed": bool(passed)},
            "predictive_readiness": {"passed": bool(passed)},
            "efficiency_readiness": {"passed": bool(passed)},
            "consolidation_readiness": {"passed": bool(passed)},
            "cognitive_runtime_readiness": {"passed": bool(passed)},
            "phase5_entry_readiness": {
                "passed": bool(passed),
                "score": 1.0 if passed else 0.0,
                "metrics": {
                    "phase5_predictive_coding.latent_transition_alignment": 1.0 if passed else 0.0,
                    "phase5_predictive_coding.prediction_error_observability": 1.0 if passed else 0.0,
                    "phase5_predictive_coding.correction_event_coverage": 1.0 if passed else 0.0,
                    "phase5_predictive_coding.anti_collapse_event_diversity": 1.0 if passed else 0.0,
                    "phase5_predictive_coding.counterfactual_transition_separation": 1.0 if passed else 0.0,
                    "phase5_predictive_coding.multi_step_latent_chain_integrity": 1.0 if passed else 0.0,
                    "phase5_predictive_coding.long_horizon_error_correction_convergence": 1.0 if passed else 0.0,
                    "phase5_predictive_coding.horizon_bucket_stability": 1.0 if passed else 0.0,
                    "phase5_predictive_coding.macro_action_effectiveness": 1.0 if passed else 0.0,
                    "phase5_predictive_coding.subgoal_decomposition_integrity": 1.0 if passed else 0.0,
                    "phase5_predictive_coding.depth_selective_routing_integrity": 1.0 if passed else 0.0,
                    "phase5_predictive_coding.micro_es_policy_refinement_integrity": 1.0 if passed else 0.0,
                },
            },
        },
        "stage_a_acceptance": {"passed": bool(passed)},
        "stage_b_readiness": {
            "passed": bool(passed),
            "minimum_requirements_passed": bool(passed),
            "minimum_failure_count": 0 if passed else 1,
            "readiness_score": 1.0 if passed else 0.0,
            "promotion_candidate_ready": bool(passed),
            "promotion_candidate_failure_count": 0 if passed else 3,
            "promotion_readiness": {
                "consecutive_passes": 3 if passed else 0,
                "required_streak": 3,
                "recommended": bool(passed),
            },
            "rlm_observation_candidate_ready": bool(passed),
            "rlm_observation_candidate_failure_count": 0 if passed else 2,
            "rlm_observation_candidate_promoted": bool(passed),
            "rlm_observation_promotion_readiness": {
                "consecutive_passes": 0,
                "required_streak": 3,
                "recommended": False,
                "promoted_to_minimum": bool(passed),
            },
            "minimum_checks": {
                "metric.future_state_transition_integrity": bool(passed),
                "metric.future_state_command_integrity": bool(passed),
                "metric.future_state_predictor_snapshot_integrity": bool(passed),
                "metric.future_state_runtime_tracking_integrity": bool(passed),
                "metric.future_state_shift_tracking_integrity": bool(passed),
                "metric.future_state_transition_operator_coverage": bool(passed),
                "metric.future_state_transition_operator_consistency": bool(passed),
                "metric.future_state_counterfactual_branch_viability": bool(passed),
                "metric.future_state_fluid_trace_integrity": bool(passed),
                "metric.future_state_fluid_support_integrity": bool(passed),
                "metric.future_state_refinement_loop_integrity": bool(passed),
                "metric.future_state_adaptive_refinement": bool(passed),
                "metric.future_state_rewarded_action_selection_integrity": bool(passed),
                "metric.future_state_policy_update_stability": bool(passed),
                "metric.future_state_energy_aware_action_preference": bool(passed),
                "metric.future_state_focused_retrieval_hit_ratio": bool(passed),
                "metric.future_state_branch_level_decision_consistency": bool(passed),
            },
            "metrics": {
                "future_state_transition_integrity": 1.0 if passed else 0.0,
                "future_state_command_integrity": 1.0 if passed else 0.0,
                "future_state_predictor_snapshot_integrity": 1.0 if passed else 0.0,
                "future_state_runtime_tracking_integrity": 1.0 if passed else 0.0,
                "future_state_shift_tracking_integrity": 1.0 if passed else 0.0,
                "future_state_transition_operator_coverage": 1.0 if passed else 0.0,
                "future_state_transition_operator_consistency": 1.0 if passed else 0.0,
                "future_state_counterfactual_branch_viability": 1.0 if passed else 0.0,
                "future_state_fluid_trace_integrity": 1.0 if passed else 0.0,
                "future_state_fluid_support_integrity": 1.0 if passed else 0.0,
                "future_state_refinement_loop_integrity": 1.0 if passed else 0.0,
                "future_state_adaptive_refinement": 1.0 if passed else 0.0,
                "future_state_rewarded_action_selection_integrity": 1.0 if passed else 0.0,
                "future_state_policy_update_stability": 1.0 if passed else 0.0,
                "future_state_energy_aware_action_preference": 1.0 if passed else 0.0,
                "future_state_focused_retrieval_hit_ratio": 1.0 if passed else 0.0,
                "future_state_branch_level_decision_consistency": 1.0 if passed else 0.0,
            },
        },
        "stage_c_readiness": {
            "passed": bool(passed),
            "minimum_requirements_passed": bool(passed),
            "minimum_checks": {
                "metric.meta_adaptation_loop": bool(passed),
                "metric.meta_adaptation_parameter_integrity": bool(passed),
                "metric.temporal_self_distillation_stability": bool(passed),
            },
            "metrics": {
                "meta_adaptation_loop": 1.0 if passed else 0.0,
                "meta_adaptation_parameter_integrity": 1.0 if passed else 0.0,
                "temporal_self_distillation_stability": 1.0 if passed else 0.0,
            },
        },
        "stage_d_readiness": {
            "passed": bool(passed),
            "minimum_requirements_passed": bool(passed),
            "minimum_failure_count": 0 if passed else 1,
            "minimum_failures": [] if passed else [
                {
                    "check": "metric.replay_noise_resilience_integrity",
                    "metric": "replay_noise_resilience_integrity",
                    "value": 0.0,
                    "threshold": 1.0,
                }
            ],
            "readiness_score": 1.0 if passed else 0.0,
            "acceptance_candidate_count": 16 if passed else 16,
            "acceptance_candidate_ready_count": 16 if passed else 0,
            "acceptance_candidates_ready": bool(passed),
            "acceptance_candidate_failure_count": 0 if passed else 16,
            "acceptance_candidates": [] if passed else [
                {
                    "check": "metric.delta_memory_multi_history_recall_observed",
                    "metric": "delta_memory_multi_history_recall_observed",
                    "ready": False,
                    "value": 0.0,
                    "threshold": 1.0,
                }
            ],
            "acceptance_candidate_stability": {
                "consecutive_passes": 3 if passed else 0,
                "required_streak": 3,
                "recommended": bool(passed),
            },
            "delta_memory_candidate_ready": bool(passed),
            "delta_memory_candidate_failure_count": 0 if passed else 4,
            "delta_memory_candidate_failures": [] if passed else [
                {
                    "check": "metric.delta_memory_multi_history_recall_observed",
                    "metric": "delta_memory_multi_history_recall_observed",
                    "value": 0.0,
                    "threshold": 1.0,
                }
            ],
            "delta_memory_candidate_promoted": False,
            "delta_memory_promotion_readiness": {
                "consecutive_passes": 3 if passed else 0,
                "required_streak": 3,
                "recommended": bool(passed),
                "promoted_to_minimum": False,
            },
            "minimum_checks": {
                "metric.replay_recovery_integrity": bool(passed),
                "metric.long_horizon_consolidation_retention": bool(passed),
                "metric.counterfactual_replay_selection_integrity": bool(passed),
                "metric.replay_upgrade_reindex_integrity": bool(passed),
                "metric.memory_health_index_integrity": bool(passed),
                "metric.replay_noise_resilience_integrity": bool(passed),
                "metric.astro_modulation_stability": bool(passed),
            },
            "metrics": {
                "replay_recovery_integrity": 1.0 if passed else 0.0,
                "long_horizon_consolidation_retention": 1.0 if passed else 0.0,
                "counterfactual_replay_selection_integrity": 1.0 if passed else 0.0,
                "replay_upgrade_reindex_integrity": 1.0 if passed else 0.0,
                "memory_health_index_integrity": 1.0 if passed else 0.0,
                "replay_noise_resilience_integrity": 1.0 if passed else 0.0,
                "astro_modulation_stability": 1.0 if passed else 0.0,
                "delta_memory_multi_history_recall_observed": 1.0 if passed else 0.0,
                "delta_memory_multi_history_health_observed": 1.0 if passed else 0.0,
                "delta_memory_erase_write_decoupling_observed": 1.0 if passed else 0.0,
                "delta_memory_erase_preserves_stable_memory_observed": 1.0 if passed else 0.0,
                "delta_memory_write_commits_residual_observed": 1.0 if passed else 0.0,
            },
        },
        "stage_e_readiness": {
            "passed": bool(passed),
            "minimum_requirements_passed": bool(passed),
            "minimum_failure_count": 0 if passed else 1,
            "minimum_failures": [] if passed else [
                {
                    "check": "metric.module_orchestration_integrity",
                    "metric": "module_orchestration_integrity",
                    "value": 0.0,
                    "threshold": 1.0,
                }
            ],
            "readiness_score": 1.0 if passed else 0.0,
            "observed_acceptance_candidate_count": 49 if passed else 49,
            "observed_acceptance_candidate_ready_count": 49 if passed else 0,
            "observed_acceptance_candidates_ready": bool(passed),
            "observed_acceptance_candidate_failure_count": 0 if passed else 49,
            "observed_acceptance_candidate_stability": {
                "consecutive_passes": 3 if passed else 0,
                "required_streak": 3,
                "recommended": bool(passed),
            },
            "minimum_checks": {
                "metric.common_spike_space_integrity": bool(passed),
                "metric.temporal_compression_efficiency": bool(passed),
                "metric.modality_temporal_budget_integrity": bool(passed),
                "metric.dendritic_context_gate_stability": bool(passed),
                "metric.spiking_hjepa_latent_transition": bool(passed),
                "metric.reverse_reasoning_trace_integrity": bool(passed),
                "metric.causal_candidate_trace_integrity": bool(passed),
                "metric.module_orchestration_integrity": bool(passed),
                "metric.counterfactual_lane_integrity": bool(passed),
                "metric.action_trace_observability": bool(passed),
                "metric.runtime_trace_replay_consistency": bool(passed),
            },
            "metrics": {
                "common_spike_space_integrity": 1.0 if passed else 0.0,
                "temporal_compression_efficiency": 1.0 if passed else 0.0,
                "modality_temporal_budget_integrity": 1.0 if passed else 0.0,
                "dendritic_context_gate_stability": 1.0 if passed else 0.0,
                "spiking_hjepa_latent_transition": 1.0 if passed else 0.0,
                "reverse_reasoning_trace_integrity": 1.0 if passed else 0.0,
                "causal_candidate_trace_integrity": 1.0 if passed else 0.0,
                "module_orchestration_integrity": 1.0 if passed else 0.0,
                "counterfactual_lane_integrity": 1.0 if passed else 0.0,
                "action_trace_observability": 1.0 if passed else 0.0,
                "runtime_trace_replay_consistency": 1.0 if passed else 0.0,
            },
        },
        "phase3_completion": {
            "passed": bool(passed),
            "completion_score": 1.0 if passed else 0.0,
            "checks": {
                "overall.score_at_least_0_95": bool(passed),
                "trend.zero_regressions": bool(passed),
                "stage_a.accepted": bool(passed),
                "stage_b.minimum_requirements_passed": bool(passed),
                "stage_c.minimum_requirements_passed": bool(passed),
                "stage_d.minimum_requirements_passed": bool(passed),
                "stage_e.minimum_requirements_passed": bool(passed),
                "focus.few_shot.passed": bool(passed),
                "focus.continual.passed": bool(passed),
                "focus.retrieval_hygiene.passed": bool(passed),
                "focus.adaptive_readiness.passed": bool(passed),
                "focus.predictive_readiness.passed": bool(passed),
                "focus.efficiency_readiness.passed": bool(passed),
                "focus.consolidation_readiness.passed": bool(passed),
                "focus.cognitive_runtime_readiness.passed": bool(passed),
            },
            "failed_checks": [] if passed else [
                "stage_d.minimum_requirements_passed",
                "stage_e.minimum_requirements_passed",
            ],
        },
    }


def _build_phase4_report(passed: bool = True):
    return {
        "evaluator_name": "Phase4ScaleContinualBenchmark",
        "passed": bool(passed),
        "overall_score": 1.0 if passed else 0.0,
        "metrics": {
            "structural_plasticity_stability": 1.0 if passed else 0.0,
            "hippocampal_transfer_integrity": 1.0 if passed else 0.0,
            "scale_out_retention_integrity": 1.0 if passed else 0.0,
            "continual_drift_recovery_integrity": 1.0 if passed else 0.0,
        },
        "threshold_results": {
            "structural_plasticity_stability": bool(passed),
            "hippocampal_transfer_integrity": bool(passed),
            "scale_out_retention_integrity": bool(passed),
            "continual_drift_recovery_integrity": bool(passed),
        },
        "quality_metrics": {
            "structural_synapse_ratio": 1.0 if passed else 2.0,
            "structural_per_context_non_empty": 1.0 if passed else 0.0,
            "hippocampal_after_top_score": 0.5 if passed else 0.0,
            "hippocampal_score_retention_ratio": 1.0 if passed else 0.0,
            "scale_out_retention_rate": 1.0 if passed else 0.0,
            "scale_out_average_query_ms": 1.0 if passed else 999.0,
            "continual_baseline_recovered": 1.0 if passed else 0.0,
            "continual_drift_observed": 1.0 if passed else 0.0,
        },
    }


def _build_release_report(passed: bool = True):
    return {
        "duration_seconds": 6.0 if passed else 1.0,
        "criteria": {
            "min_agent_turns": 24,
            "min_inference_iterations": 32,
            "min_pattern_count": 1,
            "min_duration_seconds": 5.0,
            "require_phase3_accuracy": False,
        },
        "agent": {
            "history_bounded": bool(passed),
            "issue_count": 0 if passed else 1,
            "turns": 32 if passed else 3,
        },
        "inference": {
            "roundtrip_ok": bool(passed),
            "tuple_keys_only": bool(passed),
            "iterations": 40 if passed else 5,
            "pattern_count": 2 if passed else 0,
        },
        "gate_feedback": {
            "stage_b_promotion_next_step_hint": (
                "promote_stage_b_reward_policy_metrics_to_minimum_gate"
                if passed
                else ""
            ),
            "stage_b_promotion_actions": (
                [
                    "review stage_b_contract minimum list and add the three promotion-candidate metrics"
                ]
                if passed
                else []
            ),
            "stage_b_rlm_observation_next_step_hint": "",
            "stage_b_rlm_observation_actions": [],
        },
    }


def _build_phase5_entry_gate_report(passed: bool = True):
    value = 1.0 if passed else 0.0
    return {
        "suite_name": "Phase5EntryGate",
        "passed": bool(passed),
        "failed_checks": [] if passed else ["metric.counterfactual_transition_separation"],
        "error_count": 0 if passed else 1,
        "errors": [] if passed else ["Phase 5 required metric 'counterfactual_transition_separation' did not satisfy the minimum threshold."],
        "check_count": 25,
        "pass_count": 25 if passed else 22,
        "phase5_overall_score": value,
        "checks": {
            "suite_name": {"passed": True},
            "benchmark_passed": {"passed": bool(passed)},
            "metrics_present": {"passed": True},
            "thresholds_present": {"passed": True},
            "primary_trace_complete": {"passed": bool(passed)},
            "counterfactual_branch_separable": {"passed": bool(passed)},
            "multi_step_trace_complete": {"passed": bool(passed)},
            "metric.latent_transition_alignment": {"passed": bool(passed)},
            "threshold.latent_transition_alignment": {"passed": bool(passed)},
            "metric.prediction_error_observability": {"passed": bool(passed)},
            "threshold.prediction_error_observability": {"passed": bool(passed)},
            "metric.correction_event_coverage": {"passed": bool(passed)},
            "threshold.correction_event_coverage": {"passed": bool(passed)},
            "metric.anti_collapse_event_diversity": {"passed": bool(passed)},
            "threshold.anti_collapse_event_diversity": {"passed": bool(passed)},
            "metric.counterfactual_transition_separation": {"passed": bool(passed)},
            "threshold.counterfactual_transition_separation": {"passed": bool(passed)},
            "metric.multi_step_latent_chain_integrity": {"passed": bool(passed)},
            "threshold.multi_step_latent_chain_integrity": {"passed": bool(passed)},
            "metric.long_horizon_error_correction_convergence": {"passed": bool(passed)},
            "threshold.long_horizon_error_correction_convergence": {"passed": bool(passed)},
            "metric.horizon_bucket_stability": {"passed": bool(passed)},
            "threshold.horizon_bucket_stability": {"passed": bool(passed)},
            "metric.macro_action_effectiveness": {"passed": bool(passed)},
            "threshold.macro_action_effectiveness": {"passed": bool(passed)},
            "metric.subgoal_decomposition_integrity": {"passed": bool(passed)},
            "threshold.subgoal_decomposition_integrity": {"passed": bool(passed)},
            "metric.depth_selective_routing_integrity": {"passed": bool(passed)},
            "threshold.depth_selective_routing_integrity": {"passed": bool(passed)},
            "metric.micro_es_policy_refinement_integrity": {"passed": bool(passed)},
            "threshold.micro_es_policy_refinement_integrity": {"passed": bool(passed)},
        },
    }


def _build_phase5_completion_gate_report(passed: bool = True):
    value = 1.0 if passed else 0.0
    checks = {
        "phase4_prerequisite_passed": {"passed": bool(passed)},
        "phase5_benchmark_passed": {"passed": bool(passed)},
        "phase5_entry_gate_passed": {"passed": bool(passed)},
        "primary_trace_complete": {"passed": bool(passed)},
        "multi_step_trace_complete": {"passed": bool(passed)},
        "counterfactual_branch_separable": {"passed": bool(passed)},
        "macro_step_reduction": {"passed": bool(passed), "details": {"value": value * 3.0, "required_min": 2.0}},
        "macro_cost_reduction": {"passed": bool(passed), "details": {"value": value * 0.42, "required_min": 0.30}},
        "subgoal_coverage_ratio": {"passed": bool(passed), "details": {"value": value, "required_min": 1.0}},
        "micro_es_low_rank_trace_complete": {"passed": bool(passed)},
        "micro_es_fitness_improvement": {"passed": bool(passed), "details": {"value": value * 0.249, "required_gt": 0.05}},
        "micro_es_event_cost_reduction": {"passed": bool(passed), "details": {"value": value * 0.090, "required_min": 0.04}},
        "micro_es_population_event_budget": {
            "passed": bool(passed),
            "details": {"value": 0.160 if passed else 0.400, "event_budget": 0.250},
        },
        "sparse_diffusion_block_readiness_passed": {"passed": bool(passed)},
    }
    sparse_metrics = [
        "sparse_diffusion_partition_integrity",
        "sparse_diffusion_independent_block_integrity",
        "sparse_diffusion_denoise_accuracy",
        "sparse_diffusion_event_cost_advantage",
        "sparse_diffusion_block_ablation_integrity",
        "sparse_diffusion_single_pass_recurrent_integrity",
        "sparse_diffusion_policy_compatibility",
    ]
    for name in sparse_metrics:
        checks[f"sparse_diffusion.{name}"] = {"passed": bool(passed)}
    entry_metrics = [
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
    for name in entry_metrics:
        checks[f"metric.{name}"] = {"passed": bool(passed)}
        checks[f"threshold.{name}"] = {"passed": bool(passed)}
    return {
        "suite_name": "Phase5CompletionGate",
        "passed": bool(passed),
        "failed_checks": [] if passed else ["multi_step_trace_complete"],
        "error_count": 0 if passed else 1,
        "errors": [] if passed else ["Phase 5 multi-step latent transition trace is incomplete."],
        "check_count": len(checks),
        "pass_count": len(checks) if passed else max(len(checks) - 2, 0),
        "phase5_overall_score": value,
        "checks": checks,
    }


def _build_external_validity_report(passed: bool = True):
    value = 1.0 if passed else 0.0
    ratio = 4.0 if passed else 1.0
    return {
        "suite_name": "RealDataExternalValidity",
        "passed": bool(passed),
        "checks": {
            "real_data_task_count": bool(passed),
            "sparse_accuracy_floor": bool(passed),
            "sparse_matches_dense_accuracy": bool(passed),
            "summary_keyword_coverage_floor": bool(passed),
            "continual_memory_hit_rate_floor": bool(passed),
            "ann_cost_advantage_proxy": bool(passed),
            "performance_energy_ratio_proxy": bool(passed),
            "trend.no_regressions": bool(passed),
        },
        "metrics": {
            "real_data_qa_accuracy": value,
            "ann_proxy_qa_accuracy": value,
            "real_data_summary_keyword_coverage": value,
            "continual_memory_hit_rate": value,
            "performance_energy_ratio_proxy": ratio,
            "ann_cost_advantage_proxy": ratio,
        },
        "check_details": {
            "performance_energy_ratio_proxy": {
                "passed": bool(passed),
                "value": ratio,
                "required_min": 2.0,
            },
            "ann_cost_advantage_proxy": {
                "passed": bool(passed),
                "value": ratio,
                "required_min": 2.0,
            },
            "trend.no_regressions": {
                "passed": bool(passed),
                "value": 0 if passed else 1,
                "required_max": 0,
            },
        },
    }


def _build_external_validity_ladder_report(passed: bool = True):
    value = 1.0 if passed else 0.0
    ratio = 4.0 if passed else 1.0
    profile_count = 3
    return {
        "suite_name": "RealDataExternalValidityLadder",
        "passed": bool(passed),
        "checks": {
            "all_profiles_passed": bool(passed),
            "profile_count_matches_plan": True,
            "scale_doc_counts_monotonic": True,
            "large_profile_present": True,
            "ann_cost_advantage_all_profiles": bool(passed),
            "performance_energy_ratio_all_profiles": bool(passed),
            "no_trend_regressions_all_profiles": bool(passed),
        },
        "metrics": {
            "profile_count": profile_count,
            "passed_profile_count": profile_count if passed else profile_count - 1,
            "min_real_data_qa_accuracy": value,
            "min_ann_cost_advantage_proxy": ratio,
            "min_performance_energy_ratio_proxy": ratio,
        },
    }


def _build_adaptive_credit_field_report(passed: bool = True):
    return {
        "schema": "sara-adaptive-credit-field-benchmark-v1",
        "passed": bool(passed),
        "observed_only": True,
        "metrics": {
            "decision_integrity": 1.0 if passed else 0.5,
            "harmful_update_suppression": 1.0 if passed else 0.5,
            "quantized_behavior_match": 1.0 if passed else 0.5,
        },
    }


def _build_adaptive_credit_event_memory_report(passed: bool = True):
    return {
        "schema": "sara-adaptive-credit-event-memory-benchmark-v1",
        "passed": bool(passed),
        "observed_only": True,
        "metrics": {
            "harmful_block_preserved_count": 1 if passed else 0,
            "credit_strong_entry_present": bool(passed),
            "credit_weak_entry_evicted": bool(passed),
        },
    }


def test_operational_readiness_evaluation_passes_with_all_gates_green():
    module = _load_script("operational_readiness.py")
    passed, summary = module._evaluate_operational_readiness(
        phase3_report=_build_phase3_report(True),
        phase4_report=_build_phase4_report(True),
        release_report=_build_release_report(True),
        phase5_entry_gate_report=_build_phase5_entry_gate_report(True),
        phase5_completion_gate_report=_build_phase5_completion_gate_report(True),
        external_validity_report=_build_external_validity_report(True),
        external_validity_ladder_report=_build_external_validity_ladder_report(True),
        adaptive_credit_field_report=_build_adaptive_credit_field_report(True),
        adaptive_credit_event_memory_report=_build_adaptive_credit_event_memory_report(True),
    )
    assert passed is True
    assert summary["error_count"] == 0
    assert summary["readiness_score"] == 1.0
    assert summary["checks"]["phase3_accuracy"]["passed"] is True
    assert summary["checks"]["phase3_completion"]["passed"] is True
    assert summary["checks"]["phase4_completion"]["passed"] is True
    assert summary["checks"]["phase5_entry_gate"]["passed"] is True
    assert summary["checks"]["phase5_completion_gate"]["passed"] is True
    assert summary["checks"]["external_validity"]["passed"] is True
    assert summary["checks"]["external_validity_ladder"]["passed"] is True
    assert summary["checks"]["adaptive_credit_field"]["passed"] is True
    assert summary["checks"]["adaptive_credit_event_memory"]["passed"] is True
    assert summary["checks"]["release_gate"]["passed"] is True
    assert summary["checks"]["production_profile"]["passed"] is True
    assert summary["stage_b_promotion"]["stage_b_passed"] is True
    assert summary["stage_b_promotion"]["promotion_recommended"] is True
    assert summary["stage_b_promotion"]["promotion_consecutive_passes"] == 3
    assert summary["stage_b_promotion"]["promotion_required_streak"] == 3
    assert summary["stage_d_readiness"]["passed"] is True
    assert summary["stage_d_readiness"]["minimum_requirements_passed"] is True
    assert summary["stage_d_readiness"]["replay_noise_resilience_integrity"] == 1.0
    assert summary["stage_d_readiness"]["manifold_capacity_pressure_recall_observed"] == 1.0
    assert summary["stage_d_readiness"]["manifold_capacity_pressure_scan_reduction_observed"] == 0.889
    assert summary["stage_d_readiness"]["manifold_replay_refresh_retention_observed"] == 1.0
    assert summary["stage_d_readiness"]["manifold_replay_refresh_eviction_integrity_observed"] == 1.0
    assert summary["stage_d_readiness"]["synaptic_tag_integrity_observed"] == 1.0
    assert summary["stage_d_readiness"]["memory_phase_transition_integrity_observed"] == 1.0
    assert summary["stage_d_readiness"]["metabolic_budget_integrity_observed"] == 1.0
    assert summary["stage_d_readiness"]["sleep_consolidation_retention_observed"] == 1.0
    assert summary["stage_d_readiness"]["astro_structural_lock_observed"] == 1.0
    assert summary["stage_d_readiness"]["delta_memory_phase_retention_policy_observed"] == 1.0
    assert summary["stage_d_readiness"]["delta_memory_crystal_retention_observed"] == 1.0
    assert summary["stage_d_readiness"]["delta_memory_multi_history_recall_observed"] == 1.0
    assert summary["stage_d_readiness"]["delta_memory_multi_history_health_observed"] == 1.0
    assert summary["stage_d_readiness"]["delta_memory_erase_write_decoupling_observed"] == 1.0
    assert summary["stage_d_readiness"]["delta_memory_erase_preserves_stable_memory_observed"] == 1.0
    assert summary["stage_d_readiness"]["delta_memory_write_commits_residual_observed"] == 1.0
    assert summary["stage_d_readiness"]["delta_memory_candidate_ready"] is True
    assert summary["stage_d_readiness"]["delta_memory_candidate_failure_count"] == 0
    assert summary["stage_d_readiness"]["delta_memory_candidate_failures"] == []
    assert summary["stage_d_readiness"]["acceptance_candidate_count"] == 16
    assert summary["stage_d_readiness"]["acceptance_candidate_ready_count"] == 16
    assert summary["stage_d_readiness"]["acceptance_candidates_ready"] is True
    assert summary["stage_d_readiness"]["acceptance_candidate_failure_count"] == 0
    assert summary["stage_d_readiness"]["acceptance_candidate_stability"]["consecutive_passes"] == 3
    assert summary["stage_d_readiness"]["acceptance_candidate_stability"]["recommended"] is True
    assert (
        summary["stage_d_readiness"]["acceptance_candidate_next_step_hint"]
        == "review_stage_d_acceptance_candidates_for_minimum_promotion"
    )
    assert summary["stage_d_readiness"]["acceptance_candidate_actions"][0] == (
        "review stage_d_contract acceptance candidates and choose minimum promotion scope"
    )
    assert summary["stage_d_readiness"]["acceptance_candidate_action_count"] == 3
    assert summary["stage_d_readiness"]["delta_memory_promotion_readiness"]["recommended"] is True
    assert summary["iterative_repair_plan"]["completed"] is True
    assert summary["iterative_repair_plan"]["stop_reason"] == "auto_stopped_completed"
    assert summary["repair_plan"]["estimated_steps"] >= 1
    assert summary["repair_plan"]["coverage_ratio"] == 1.0
    assert summary["error_details"] == []
    assert summary["error_details_summary"]["total"] == 0
    assert summary["failure_focus"]["primary_category"] == ""
    assert any(
        isinstance(action, dict)
        and action.get("priority") == "medium"
        and "stage_b_promotion" in action.get("affected_checks", [])
        for action in summary["recovery_actions"]
    )
    assert summary["stage_e_readiness"]["passed"] is True
    assert summary["stage_e_readiness"]["minimum_requirements_passed"] is True
    assert summary["stage_e_readiness"]["causal_candidate_trace_integrity"] == 1.0
    assert summary["stage_e_readiness"]["module_orchestration_integrity"] == 1.0
    assert summary["stage_e_readiness"]["runtime_trace_replay_consistency"] == 1.0
    assert summary["stage_e_readiness"]["manifold_trace_support_observed"] == 1.0
    assert summary["stage_e_readiness"]["manifold_trace_recall_observed"] == 1.0
    assert summary["stage_e_readiness"]["manifold_trace_scan_budget_observed"] == 1.0
    assert summary["stage_e_readiness"]["manifold_trace_index_scan_reduction_observed"] == 1.0
    assert summary["stage_e_readiness"]["manifold_trace_candidate_guard_observed"] == 1.0
    assert summary["stage_e_readiness"]["delta_memory_steering_integrity_observed"] == 1.0
    assert summary["stage_e_readiness"]["delta_memory_counterfactual_isolation_observed"] == 1.0
    assert summary["stage_e_readiness"]["delta_memory_trace_observability_observed"] == 1.0
    assert summary["stage_e_readiness"]["linear_snn_fusion_observed_policy"] == "excluded_from_score_and_release_gate"
    assert summary["stage_e_readiness"]["linear_snn_fusion_trend_has_previous"] is True
    assert summary["stage_e_readiness"]["linear_snn_fusion_trend_regression_count"] == 0
    assert summary["stage_e_readiness"]["linear_snn_fusion_trend_release_gate_blocking"] is False
    assert summary["stage_e_readiness"]["architecture_integration_observed_policy"] == "excluded_from_score_and_release_gate"
    assert summary["stage_e_readiness"]["architecture_integration_trend_has_previous"] is True
    assert summary["stage_e_readiness"]["architecture_integration_trend_regression_count"] == 0
    assert summary["stage_e_readiness"]["architecture_integration_trend_release_gate_blocking"] is False
    assert summary["stage_e_readiness"]["predictive_spike_entropy_reduction_observed"] == 1.0
    assert summary["stage_e_readiness"]["phase_binding_coincidence_integrity_observed"] == 1.0
    assert summary["stage_e_readiness"]["forward_only_local_update_stability_observed"] == 1.0
    assert summary["stage_e_readiness"]["lejepa_linear_identifiability_proxy_observed"] == 1.0
    assert summary["stage_e_readiness"]["lejepa_latent_whitening_health_observed"] == 1.0
    assert summary["stage_e_readiness"]["lejepa_factor_disentanglement_observed"] == 1.0
    assert summary["stage_e_readiness"]["lejepa_latent_planning_consistency_observed"] == 1.0
    assert summary["stage_e_readiness"]["lejepa_positive_pair_alignment_observed"] == 1.0
    assert summary["stage_e_readiness"]["plastic_submodel_registry_integrity_observed"] == 1.0
    assert summary["stage_e_readiness"]["runtime_submodel_route_action_grounding_observed"] == 1.0
    assert summary["stage_e_readiness"]["runtime_submodel_counterfactual_route_separation_observed"] == 1.0
    assert summary["stage_e_readiness"]["submodel_intervention_trace_integrity_observed"] == 1.0
    assert summary["stage_e_readiness"]["submodel_ablation_effect_observed"] == 1.0
    assert summary["stage_e_readiness"]["submodel_credit_assignment_trace_integrity_observed"] == 1.0
    assert summary["stage_e_readiness"]["runtime_submodel_local_credit_assignment_observed"] == 1.0
    assert summary["stage_e_readiness"]["submodel_structural_growth_bounded_observed"] == 1.0
    assert summary["stage_e_readiness"]["submodel_structural_pruning_observed"] == 1.0
    assert summary["stage_e_readiness"]["submodel_scientific_hypothesis_trace_integrity_observed"] == 1.0
    assert summary["stage_e_readiness"]["submodel_counterexample_revision_observed"] == 1.0
    assert summary["stage_e_readiness"]["submodel_hypothesis_bank_integrity_observed"] == 1.0
    assert summary["stage_e_readiness"]["submodel_open_ended_selection_observed"] == 1.0
    assert summary["stage_e_readiness"]["micro_turn_event_budget_observed"] == 1.0
    assert summary["stage_e_readiness"]["foreground_background_context_handoff_observed"] == 1.0
    assert summary["stage_e_readiness"]["interrupt_recovery_trace_observed"] == 1.0
    assert summary["stage_e_readiness"]["simultaneous_stream_route_integrity_observed"] == 1.0
    assert summary["stage_e_readiness"]["time_aligned_backchannel_policy_observed"] == 1.0
    assert summary["stage_e_readiness"]["phase_assigned_submodel_route_observed"] == 1.0
    assert summary["stage_e_readiness"]["uncertainty_bucket_specialization_observed"] == 1.0
    assert summary["stage_e_readiness"]["denoising_correction_trace_integrity_observed"] == 1.0
    assert summary["stage_e_readiness"]["block_independent_local_update_budget_observed"] == 1.0
    assert summary["phase5_entry_readiness"]["passed"] is True
    assert summary["phase5_entry_readiness"]["correction_event_coverage"] == 1.0


def test_operational_readiness_evaluation_detects_gate_failures():
    module = _load_script("operational_readiness.py")
    passed, summary = module._evaluate_operational_readiness(
        phase3_report=_build_phase3_report(False),
        phase4_report=_build_phase4_report(False),
        release_report=_build_release_report(False),
        phase5_entry_gate_report=_build_phase5_entry_gate_report(False),
        phase5_completion_gate_report=_build_phase5_completion_gate_report(False),
        adaptive_credit_field_report=_build_adaptive_credit_field_report(False),
        adaptive_credit_event_memory_report=_build_adaptive_credit_event_memory_report(False),
    )
    assert passed is False
    assert summary["error_count"] > 0
    assert summary["checks"]["phase3_accuracy"]["passed"] is False
    assert summary["checks"]["phase3_completion"]["passed"] is False
    assert summary["checks"]["phase4_completion"]["passed"] is False
    assert summary["checks"]["phase5_entry_gate"]["passed"] is False
    assert summary["checks"]["phase5_completion_gate"]["passed"] is False
    assert summary["checks"]["adaptive_credit_field"]["passed"] is False
    assert summary["checks"]["adaptive_credit_event_memory"]["passed"] is False
    assert summary["checks"]["release_gate"]["passed"] is False
    assert summary["checks"]["production_profile"]["passed"] is True
    assert summary["stage_b_promotion"]["stage_b_passed"] is False
    assert summary["stage_b_promotion"]["promotion_recommended"] is False
    assert summary["stage_b_promotion"]["promotion_candidate_failure_count"] == 3
    assert summary["stage_d_readiness"]["passed"] is False
    assert summary["stage_d_readiness"]["minimum_failure_count"] == 1
    assert summary["iterative_repair_plan"]["completed"] is False
    assert summary["iterative_repair_plan"]["stop_reason"] == "pending_actions"
    assert summary["iterative_repair_plan"]["failed_checks"]
    assert isinstance(summary["repair_plan"]["selected_actions"], list)
    assert summary["repair_plan"]["estimated_steps"] >= 1
    assert summary["error_details_summary"]["total"] > 0
    assert summary["failure_focus"]["primary_category"] != ""
    assert summary["stage_e_readiness"]["passed"] is False
    assert summary["stage_e_readiness"]["minimum_failure_count"] == 1
    assert summary["phase5_entry_readiness"]["passed"] is False
    assert summary["phase5_entry_readiness"]["counterfactual_transition_separation"] == 0.0
    formatted = module.format_operational_summary(summary)
    assert "- adaptive_credit_field: FAIL" in formatted
    assert "- adaptive_credit_event_memory: FAIL" in formatted
    assert "- stage_d_delta_memory_candidate_failure: metric.delta_memory_multi_history_recall_observed value=0.000 required>=1.000 description=delta-memory multi-history recall" in formatted
    assert "- stage_d_acceptance_candidate_failure: metric.delta_memory_multi_history_recall_observed value=0.000 required>=1.000 description=delta-memory multi-history recall" in formatted
    assert len(summary["recovery_actions"]) >= 1
    commands = [
        str(action.get("command", ""))
        for action in summary["recovery_actions"]
        if isinstance(action, dict)
    ]
    assert any("--record-repair-source stage_d_acceptance_candidate_repair" in command for command in commands)
    assert any("repair_stage_d_acceptance_candidates" in command for command in commands)
    assert any(
        isinstance(action, dict)
        and "delta_memory_multi_history_recall_observed" in action.get("affected_checks", [])
        for action in summary["recovery_actions"]
    )
    assert any(
        isinstance(action, dict)
        and action.get("priority") == "high"
        and isinstance(action.get("expected_effect"), str)
        and action.get("expected_effect")
        for action in summary["recovery_actions"]
    )


def test_operational_readiness_rejects_failed_phase5_entry_gate_artifact():
    module = _load_script("operational_readiness.py")

    passed, summary = module._evaluate_operational_readiness(
        phase3_report=_build_phase3_report(True),
        phase4_report=_build_phase4_report(True),
        release_report=_build_release_report(True),
        phase5_entry_gate_report=_build_phase5_entry_gate_report(False),
        phase5_completion_gate_report=_build_phase5_completion_gate_report(True),
    )

    assert passed is False
    assert summary["checks"]["phase5_entry_gate"]["passed"] is False
    assert any("failed checks" in error.lower() for error in summary["checks"]["phase5_entry_gate"]["errors"])


def test_operational_readiness_rejects_failed_phase5_completion_gate_artifact():
    module = _load_script("operational_readiness.py")

    passed, summary = module._evaluate_operational_readiness(
        phase3_report=_build_phase3_report(True),
        phase4_report=_build_phase4_report(True),
        release_report=_build_release_report(True),
        phase5_entry_gate_report=_build_phase5_entry_gate_report(True),
        phase5_completion_gate_report=_build_phase5_completion_gate_report(False),
    )

    assert passed is False
    assert summary["checks"]["phase5_completion_gate"]["passed"] is False
    assert any("failed checks" in error.lower() for error in summary["checks"]["phase5_completion_gate"]["errors"])


def test_operational_readiness_rejects_failed_external_validity_artifact():
    module = _load_script("operational_readiness.py")

    passed, summary = module._evaluate_operational_readiness(
        phase3_report=_build_phase3_report(True),
        phase4_report=_build_phase4_report(True),
        release_report=_build_release_report(True),
        phase5_entry_gate_report=_build_phase5_entry_gate_report(True),
        phase5_completion_gate_report=_build_phase5_completion_gate_report(True),
        external_validity_report=_build_external_validity_report(False),
    )

    assert passed is False
    assert summary["checks"]["external_validity"]["passed"] is False
    assert any("external validity" in error.lower() for error in summary["checks"]["external_validity"]["errors"])
    assert any(
        action.get("command") == "python scripts/eval/real_data_external_validity.py"
        for action in summary["recovery_actions"]
    )


def test_operational_readiness_rejects_failed_external_validity_ladder_artifact():
    module = _load_script("operational_readiness.py")

    passed, summary = module._evaluate_operational_readiness(
        phase3_report=_build_phase3_report(True),
        phase4_report=_build_phase4_report(True),
        release_report=_build_release_report(True),
        phase5_entry_gate_report=_build_phase5_entry_gate_report(True),
        phase5_completion_gate_report=_build_phase5_completion_gate_report(True),
        external_validity_report=_build_external_validity_report(True),
        external_validity_ladder_report=_build_external_validity_ladder_report(False),
    )

    assert passed is False
    assert summary["checks"]["external_validity_ladder"]["passed"] is False
    assert any(
        "external validity ladder" in error.lower()
        for error in summary["checks"]["external_validity_ladder"]["errors"]
    )
    assert any(
        action.get("command") == "python scripts/eval/real_data_external_validity_ladder.py"
        for action in summary["recovery_actions"]
    )


def test_operational_readiness_rejects_failed_adaptive_credit_artifacts():
    module = _load_script("operational_readiness.py")

    passed, summary = module._evaluate_operational_readiness(
        phase3_report=_build_phase3_report(True),
        phase4_report=_build_phase4_report(True),
        release_report=_build_release_report(True),
        phase5_entry_gate_report=_build_phase5_entry_gate_report(True),
        phase5_completion_gate_report=_build_phase5_completion_gate_report(True),
        adaptive_credit_field_report=_build_adaptive_credit_field_report(False),
        adaptive_credit_event_memory_report=_build_adaptive_credit_event_memory_report(False),
    )

    assert passed is False
    assert summary["checks"]["adaptive_credit_field"]["passed"] is False
    assert summary["checks"]["adaptive_credit_event_memory"]["passed"] is False
    categories = {
        item.get("category")
        for item in summary["error_details"]
        if isinstance(item, dict)
    }
    assert "adaptive_credit_field_validation" in categories
    assert "adaptive_credit_event_memory_validation" in categories
    commands = [
        str(action.get("command", ""))
        for action in summary["recovery_actions"]
        if isinstance(action, dict)
    ]
    assert "python scripts/eval/adaptive_credit_field_benchmark.py" in commands
    assert "python scripts/eval/adaptive_credit_event_memory_benchmark.py" in commands
    assert any(
        isinstance(action, dict)
        and action.get("command") == "python scripts/eval/adaptive_credit_event_memory_benchmark.py"
        and "event_memory_ingest_pipeline" in action.get("affected_checks", [])
        for action in summary["recovery_actions"]
    )
    assert summary["repair_plan"]["coverage_ratio"] == 1.0
    assert summary["failure_focus"]["primary_category"].startswith("adaptive_credit_")


def test_operational_readiness_rejects_phase5_completion_gate_missing_required_checks():
    module = _load_script("operational_readiness.py")
    incomplete = _build_phase5_completion_gate_report(True)
    incomplete["checks"] = {"phase5_entry_gate_passed": {"passed": True}}

    passed, summary = module._evaluate_operational_readiness(
        phase3_report=_build_phase3_report(True),
        phase4_report=_build_phase4_report(True),
        release_report=_build_release_report(True),
        phase5_entry_gate_report=_build_phase5_entry_gate_report(True),
        phase5_completion_gate_report=incomplete,
    )

    assert passed is False
    assert summary["checks"]["phase5_completion_gate"]["passed"] is False
    assert any("missing required checks" in error.lower() for error in summary["checks"]["phase5_completion_gate"]["errors"])


def test_operational_readiness_summary_includes_stage_e_snapshot():
    module = _load_script("operational_readiness.py")
    phase3_report = _build_phase3_report(True)
    release_report = _build_release_report(True)
    _, report = module._evaluate_operational_readiness(
        phase3_report=phase3_report,
        phase4_report=_build_phase4_report(True),
        release_report=release_report,
        phase5_entry_gate_report=_build_phase5_entry_gate_report(True),
        phase5_completion_gate_report=_build_phase5_completion_gate_report(True),
    )
    report["research_review"] = module.build_operational_research_review(
        phase3_report=phase3_report,
        release_report=release_report,
        operational_report=report,
    )

    summary = module.format_operational_summary(report)

    assert "- stage_e_passed: True" in summary
    assert "- stage_e_minimum_requirements_passed: True" in summary
    assert "- stage_e_minimum_failure_count: 0" in summary
    assert "- stage_e_readiness_score: 1.000" in summary
    assert "- stage_e_observed_acceptance_candidate_count: 49" in summary
    assert "- stage_e_observed_acceptance_candidate_ready_count: 49" in summary
    assert "- stage_e_observed_acceptance_candidates_ready: True" in summary
    assert "- stage_e_observed_acceptance_candidate_failure_count: 0" in summary
    assert "- stage_e_observed_acceptance_candidate_consecutive_passes: 3" in summary
    assert "- stage_e_observed_acceptance_candidate_required_streak: 3" in summary
    assert "- stage_e_observed_acceptance_candidate_stability_recommended: True" in summary
    assert "- stage_e_causal_candidate_trace_integrity: 1.000" in summary
    assert "- stage_e_module_orchestration_integrity: 1.000" in summary
    assert "- stage_e_counterfactual_lane_integrity: 1.000" in summary
    assert "- stage_e_action_trace_observability: 1.000" in summary
    assert "- stage_e_runtime_trace_replay_consistency: 1.000" in summary
    assert "- stage_e_manifold_trace_support_observed: 1.000" in summary
    assert "- stage_e_manifold_trace_recall_observed: 1.000" in summary
    assert "- stage_e_manifold_trace_scan_budget_observed: 1.000" in summary
    assert "- stage_e_manifold_trace_index_scan_reduction_observed: 1.000" in summary
    assert "- stage_e_manifold_trace_candidate_guard_observed: 1.000" in summary
    assert "- stage_e_delta_memory_steering_integrity_observed: 1.000" in summary
    assert "- stage_e_delta_memory_counterfactual_isolation_observed: 1.000" in summary
    assert "- stage_e_delta_memory_trace_observability_observed: 1.000" in summary
    assert "- stage_e_linear_snn_fusion_observed_policy: excluded_from_score_and_release_gate" in summary
    assert "- stage_e_linear_snn_fusion_trend_has_previous: True" in summary
    assert "- stage_e_linear_snn_fusion_trend_regression_count: 0" in summary
    assert "- stage_e_linear_snn_fusion_trend_release_gate_blocking: False" in summary
    assert "- stage_e_architecture_integration_observed_policy: excluded_from_score_and_release_gate" in summary
    assert "- stage_e_architecture_integration_trend_has_previous: True" in summary
    assert "- stage_e_architecture_integration_trend_regression_count: 0" in summary
    assert "- stage_e_architecture_integration_trend_release_gate_blocking: False" in summary
    assert "- stage_e_predictive_spike_entropy_reduction_observed: 1.000" in summary
    assert "- stage_e_phase_binding_coincidence_integrity_observed: 1.000" in summary
    assert "- stage_e_forward_only_local_update_stability_observed: 1.000" in summary
    assert "- stage_e_lejepa_linear_identifiability_proxy_observed: 1.000" in summary
    assert "- stage_e_lejepa_latent_whitening_health_observed: 1.000" in summary
    assert "- stage_e_lejepa_factor_disentanglement_observed: 1.000" in summary
    assert "- stage_e_lejepa_latent_planning_consistency_observed: 1.000" in summary
    assert "- stage_e_lejepa_positive_pair_alignment_observed: 1.000" in summary
    assert "- stage_e_plastic_submodel_registry_integrity_observed: 1.000" in summary
    assert "- stage_e_runtime_submodel_route_action_grounding_observed: 1.000" in summary
    assert "- stage_e_runtime_submodel_counterfactual_route_separation_observed: 1.000" in summary
    assert "- stage_e_submodel_intervention_trace_integrity_observed: 1.000" in summary
    assert "- stage_e_submodel_ablation_effect_observed: 1.000" in summary
    assert "- stage_e_submodel_credit_assignment_trace_integrity_observed: 1.000" in summary
    assert "- stage_e_runtime_submodel_local_credit_assignment_observed: 1.000" in summary
    assert "- stage_e_submodel_structural_adaptation_trace_integrity_observed: 1.000" in summary
    assert "- stage_e_submodel_structural_growth_bounded_observed: 1.000" in summary
    assert "- stage_e_submodel_scientific_hypothesis_trace_integrity_observed: 1.000" in summary
    assert "- stage_e_submodel_counterexample_revision_observed: 1.000" in summary
    assert "- stage_e_submodel_hypothesis_bank_integrity_observed: 1.000" in summary
    assert "- stage_e_submodel_open_ended_selection_observed: 1.000" in summary
    assert "- stage_e_micro_turn_event_budget_observed: 1.000" in summary
    assert "- stage_e_foreground_background_context_handoff_observed: 1.000" in summary
    assert "- stage_e_interrupt_recovery_trace_observed: 1.000" in summary
    assert "- stage_e_simultaneous_stream_route_integrity_observed: 1.000" in summary
    assert "- stage_e_time_aligned_backchannel_policy_observed: 1.000" in summary
    assert "- stage_e_phase_assigned_submodel_route_observed: 1.000" in summary
    assert "- stage_e_uncertainty_bucket_specialization_observed: 1.000" in summary
    assert "- stage_e_denoising_correction_trace_integrity_observed: 1.000" in summary
    assert "- stage_e_block_independent_local_update_budget_observed: 1.000" in summary
    assert "- neuromorphic_profile_history_regression_observed: 1.000" in summary
    assert "- neuromorphic_profile_report_integrity_observed: 1.000" in summary
    assert "- neuromorphic_backend_profile_compatibility_observed: 1.000" in summary
    assert "- neuromorphic_stage_e_state_trace_ir_observed: 1.000" in summary
    assert "- neuromorphic_stage_e_routing_hint_coverage_observed: 1.000" in summary
    assert "- neuromorphic_stage_e_online_update_policy_observed: 1.000" in summary
    assert "- neuromorphic_stage_e_event_budget_observed: 1.000" in summary
    assert "- neuromorphic_profile_trend_has_previous: True" in summary
    assert "- neuromorphic_profile_trend_regression_count: 0" in summary
    assert "- neuromorphic_profile_trend_policy_change_count: 0" in summary
    assert "- neuromorphic_profile_trend_regression_details: none" in summary
    assert "- neuromorphic_profile_trend_policy_change_details: none" in summary
    assert "- neuromorphic_profile_recovery_hint: No neuromorphic profile repair required." in summary
    assert "- phase5_entry_gate: PASS" in summary
    assert "- phase5_completion_gate: PASS" in summary
    assert "- research_review_passed: True" in summary
    assert "- research_review_score: 1.000" in summary
    assert "- research_review_release_gate_blocking: False" in summary
    assert "- research_review_requires_human_approval: True" in summary
    assert "- research_review_next_hypothesis_count: 0" in summary
    assert "- research_review_regression_watchlist_count: 0" in summary
    assert "- research_review_bounded_experiment_graph_node_count: 4" in summary
    assert "- research_review_bounded_experiment_graph_edge_count: 0" in summary
    assert "- research_review_sara_policy_dimension_count: 5" in summary
    assert "- research_review_sara_policy_needs_review_count: 0" in summary
    assert "- research_review_experiment_adoption_candidate_count: 4" in summary
    assert "- research_review_experiment_regressing_item_count: 0" in summary
    assert "- research_review_experiment_falsified_item_count: 0" in summary
    assert "- research_review_experiment_human_review_pending_count: 0" in summary
    assert "- research_journal_entry_count: 0" in summary
    assert "- research_journal_total_seen_count: 0" in summary
    assert "- phase5_completion_micro_es_fitness_improvement_value: 0.249 required_gt=0.050" in summary
    assert "- phase5_completion_micro_es_population_event_budget_value: 0.160 event_budget=0.250" in summary
    assert "- phase5_entry_passed: True" in summary
    assert "- phase5_entry_readiness_score: 1.000" in summary
    assert "- phase5_latent_transition_alignment: 1.000" in summary
    assert "- phase5_correction_event_coverage: 1.000" in summary
    assert "- phase5_counterfactual_transition_separation: 1.000" in summary
    assert "- phase5_multi_step_latent_chain_integrity: 1.000" in summary
    assert "- phase5_long_horizon_error_correction_convergence: 1.000" in summary
    assert "- phase5_macro_action_effectiveness: 1.000" in summary
    assert "- phase5_subgoal_decomposition_integrity: 1.000" in summary
    assert "- phase5_depth_selective_routing_integrity: 1.000" in summary
    assert "- phase5_micro_es_policy_refinement_integrity: 1.000" in summary
    assert "- iterative_completed: True" in summary
    assert "- iterative_stop_reason: auto_stopped_completed" in summary
    assert "- iterative_next_step_hint: No further action required. Re-run operational_readiness only for verification." in summary
    assert "- repair_plan_steps: " in summary
    assert "- repair_plan_coverage: " in summary
    assert "- failure_focus_primary_category: " in summary
    assert "- failure_focus_confidence: " in summary
    assert "- stage_b_passed: True" in summary
    assert "- stage_b_promotion_candidate_ready: True" in summary
    assert "- stage_b_promotion_consecutive_passes: 3" in summary
    assert "- stage_b_promotion_required_streak: 3" in summary
    assert "- stage_b_promotion_recommended: True" in summary
    assert "- stage_b_promotion_next_step_hint: promote_stage_b_reward_policy_metrics_to_minimum_gate" in summary
    assert "- stage_b_rlm_observation_candidate_ready: True" in summary
    assert "- stage_b_rlm_observation_candidate_failure_count: 0" in summary
    assert "- stage_b_rlm_observation_candidate_promoted: True" in summary
    assert "- stage_b_rlm_observation_consecutive_passes: 0" in summary
    assert "- stage_b_rlm_observation_required_streak: 3" in summary
    assert "- stage_b_rlm_observation_promotion_recommended: False" in summary
    assert "- stage_b_rlm_observation_next_step_hint: " in summary
    assert "- stage_d_passed: True" in summary
    assert "- stage_d_minimum_requirements_passed: True" in summary
    assert "- stage_d_minimum_failure_count: 0" in summary
    assert "- stage_d_readiness_score: 1.000" in summary
    assert "- stage_d_acceptance_candidate_count: 16" in summary
    assert "- stage_d_acceptance_candidate_ready_count: 16" in summary
    assert "- stage_d_acceptance_candidates_ready: True" in summary
    assert "- stage_d_acceptance_candidate_failure_count: 0" in summary
    assert "- stage_d_acceptance_candidate_consecutive_passes: 3" in summary
    assert "- stage_d_acceptance_candidate_required_streak: 3" in summary
    assert "- stage_d_acceptance_candidate_stability_recommended: True" in summary
    assert "- stage_d_acceptance_candidate_next_step_hint: review_stage_d_acceptance_candidates_for_minimum_promotion" in summary
    assert "- stage_d_acceptance_candidate_action_count: 3" in summary
    assert "- stage_d_delta_memory_candidate_ready: True" in summary
    assert "- stage_d_delta_memory_candidate_failure_count: 0" in summary
    assert "- stage_d_delta_memory_candidate_promoted: False" in summary
    assert "- stage_d_delta_memory_consecutive_passes: 3" in summary
    assert "- stage_d_delta_memory_required_streak: 3" in summary
    assert "- stage_d_delta_memory_promotion_recommended: True" in summary
    assert "- stage_d_replay_noise_resilience_integrity: 1.000" in summary
    assert "- stage_d_astro_modulation_stability: 1.000" in summary
    assert "- stage_d_manifold_continual_retention_observed: 1.000" in summary
    assert "- stage_d_manifold_capacity_pressure_recall_observed: 1.000" in summary
    assert "- stage_d_manifold_capacity_pressure_scan_reduction_observed: 0.889" in summary
    assert "- stage_d_manifold_replay_refresh_retention_observed: 1.000" in summary
    assert "- stage_d_manifold_replay_refresh_eviction_integrity_observed: 1.000" in summary
    assert "- stage_d_synaptic_tag_integrity_observed: 1.000" in summary
    assert "- stage_d_memory_phase_transition_integrity_observed: 1.000" in summary
    assert "- stage_d_metabolic_budget_integrity_observed: 1.000" in summary
    assert "- stage_d_sleep_consolidation_retention_observed: 1.000" in summary
    assert "- stage_d_astro_structural_lock_observed: 1.000" in summary
    assert "- stage_d_delta_memory_phase_retention_policy_observed: 1.000" in summary
    assert "- stage_d_delta_memory_crystal_retention_observed: 1.000" in summary
    assert "- stage_d_delta_memory_multi_history_recall_observed: 1.000" in summary
    assert "- stage_d_delta_memory_multi_history_health_observed: 1.000" in summary
    assert "- stage_d_delta_memory_erase_write_decoupling_observed: 1.000" in summary
    assert "- stage_d_delta_memory_erase_preserves_stable_memory_observed: 1.000" in summary
    assert "- stage_d_delta_memory_write_commits_residual_observed: 1.000" in summary
    assert "- phase5_manifold_candidate_miss_guard_observed: 1.000" in summary
    assert "- stage_b_promotion_action: review stage_b_contract minimum list and add the three promotion-candidate metrics" in summary


def test_operational_readiness_summary_includes_stage_e_minimum_failures():
    module = _load_script("operational_readiness.py")
    _, report = module._evaluate_operational_readiness(
        phase3_report=_build_phase3_report(False),
        phase4_report=_build_phase4_report(True),
        release_report=_build_release_report(True),
    )

    summary = module.format_operational_summary(report)

    assert "- stage_e_passed: False" in summary
    assert "- stage_e_minimum_failure_count: 1" in summary
    assert "- stage_e_minimum_failure: metric.module_orchestration_integrity value=0.000 required>=1.000" in summary


def test_operational_readiness_summary_includes_stage_e_observed_candidate_failures():
    module = _load_script("operational_readiness.py")
    phase3_report = _build_phase3_report(True)
    phase3_report["stage_e_readiness"]["observed_acceptance_candidate_failure_count"] = 1
    phase3_report["stage_e_readiness"]["observed_acceptance_candidates_ready"] = False
    phase3_report["stage_e_readiness"]["observed_acceptance_candidate_failures"] = [
        {
            "check": "metric.micro_turn_event_budget_observed",
            "metric": "micro_turn_event_budget_observed",
            "value": 0.0,
            "threshold": 1.0,
        }
    ]
    _, report = module._evaluate_operational_readiness(
        phase3_report=phase3_report,
        phase4_report=_build_phase4_report(True),
        release_report=_build_release_report(True),
    )

    summary = module.format_operational_summary(report)

    assert "- stage_e_observed_acceptance_candidate_failure_count: 1" in summary
    assert (
        "- stage_e_observed_acceptance_candidate_failure: "
        "metric.micro_turn_event_budget_observed value=0.000 required>=1.000 "
        "description=micro turn event budget observed"
    ) in summary


def test_operational_readiness_strict_production_requires_extended_profile():
    module = _load_script("operational_readiness.py")
    passed, summary = module._evaluate_operational_readiness(
        phase3_report=_build_phase3_report(True),
        phase4_report=_build_phase4_report(True),
        release_report=_build_release_report(True),
        strict_production=True,
    )
    assert passed is False
    assert summary["checks"]["production_profile"]["passed"] is False
    assert any("shipping_ready=true" in error for error in summary["checks"]["production_profile"]["errors"])
    assert any("strict mode" in action.get("command", "") or "--strict-production" in action.get("command", "") for action in summary["recovery_actions"])


def test_operational_recovery_actions_include_stage_b_promotion_followup_commands():
    module = _load_script("operational_readiness.py")
    _, summary = module._evaluate_operational_readiness(
        phase3_report=_build_phase3_report(True),
        phase4_report=_build_phase4_report(True),
        release_report=_build_release_report(True),
    )
    commands = [
        str(action.get("command", ""))
        for action in summary["recovery_actions"]
        if isinstance(action, dict)
    ]
    assert any("--record-repair-source stage_b_promotion" in command for command in commands)
    assert any(
        "promote_stage_b_reward_policy_metrics_to_minimum_gate" in command
        for command in commands
    )
    assert any("--record-repair-source stage_d_delta_memory_promotion" in command for command in commands)
    assert any(
        "promote_stage_d_delta_memory_metrics_to_minimum_gate" in command
        for command in commands
    )
    assert any("--record-repair-source stage_d_acceptance_candidate_stability" in command for command in commands)
    assert any(
        "review_stage_d_acceptance_candidates_for_minimum_promotion" in command
        for command in commands
    )


def test_operational_summary_lists_stage_d_acceptance_candidate_actions():
    module = _load_script("operational_readiness.py")
    _, report = module._evaluate_operational_readiness(
        phase3_report=_build_phase3_report(True),
        phase4_report=_build_phase4_report(True),
        release_report=_build_release_report(True),
    )
    summary = module.format_operational_summary(report)
    assert "- stage_d_acceptance_candidate_action: review stage_d_contract acceptance candidates and choose minimum promotion scope" in summary


def test_operational_summary_includes_iterative_actions_when_gate_fails():
    module = _load_script("operational_readiness.py")
    _, report = module._evaluate_operational_readiness(
        phase3_report=_build_phase3_report(False),
        phase4_report=_build_phase4_report(False),
        release_report=_build_release_report(False),
    )
    summary = module.format_operational_summary(report)
    assert "- iterative_completed: False" in summary
    assert "- iterative_stop_reason: pending_actions" in summary
    assert "- iterative_action_count: " in summary
    assert "- auto_dispatch_requested: 0" in summary
    assert "- auto_dispatch_selection_mode: priority" in summary
    assert "- repair_retry_queue_count: " in summary
    assert "- error_detail_count: " in summary
    assert "- efficiency_kpi_failure_count: " in summary
    assert "- error_detail_total: " in summary
    assert "- error_detail_type_count: " in summary
    assert "- error_detail_category_count: " in summary


def test_operational_iterative_plan_excludes_successful_logged_commands():
    module = _load_script("operational_readiness.py")
    _, summary = module._evaluate_operational_readiness(
        phase3_report=_build_phase3_report(False),
        phase4_report=_build_phase4_report(False),
        release_report=_build_release_report(False),
        execution_log=[
            {
                "command": "python scripts/eval/phase3_accuracy_suite.py",
                "status": "success",
            }
        ],
    )
    iterative = summary["iterative_repair_plan"]
    assert iterative["iteration"] == 2
    assert iterative["executed_steps"] == 1
    assert iterative["successful_steps"] == 1
    commands = [
        str(item.get("command", ""))
        for item in iterative.get("next_actions", [])
        if isinstance(item, dict)
    ]
    assert "python scripts/eval/phase3_accuracy_suite.py" not in commands


def test_collect_operational_readiness_artifacts_returns_expected_sections():
    module = _load_script("operational_readiness.py")
    checks = {
        "phase3_accuracy": {"passed": False, "errors": ["phase3 failed"]},
        "phase3_completion": {"passed": True, "errors": []},
        "phase4_completion": {"passed": True, "errors": []},
        "release_gate": {"passed": False, "errors": ["release failed"]},
        "production_profile": {"passed": True, "errors": []},
    }
    artifacts = module.collect_operational_readiness_artifacts(
        checks,
        stage_b_promotion={
            "promotion_recommended": True,
            "promotion_next_step_hint": "promote_stage_b_reward_policy_metrics_to_minimum_gate",
            "promotion_actions": ["review stage_b_contract minimum list and add the three promotion-candidate metrics"],
        },
        execution_log=[],
        strict_production=False,
    )
    assert "recovery_actions" in artifacts
    assert "repair_plan" in artifacts
    assert "error_details" in artifacts
    assert "error_details_summary" in artifacts
    assert "failure_focus" in artifacts
    assert "iterative_repair_plan" in artifacts
    assert artifacts["error_details_summary"]["total"] >= 1


def test_operational_readiness_efficiency_failure_focus_and_recovery_action():
    module = _load_script("operational_readiness.py")
    checks = {
        "phase3_accuracy": {
            "passed": False,
            "errors": [
                "Phase 3 efficiency_readiness did not satisfy performance-per-energy ratio proxy "
                "(energy_efficiency.performance_energy_ratio_proxy, value=0.150, required>=0.200)."
            ],
        },
        "phase3_completion": {"passed": False, "errors": ["Phase 3 completion required check failed: focus.efficiency_readiness.passed"]},
        "phase4_completion": {"passed": True, "errors": []},
        "release_gate": {"passed": True, "errors": []},
        "production_profile": {"passed": True, "errors": []},
    }
    artifacts = module.collect_operational_readiness_artifacts(
        checks,
        stage_b_promotion={},
        execution_log=[],
        strict_production=False,
    )
    assert artifacts["failure_focus"]["primary_category"] == "phase3_efficiency_kpi"
    assert any(
        isinstance(action, dict)
        and "energy_efficiency_benchmark.py" in str(action.get("command", ""))
        for action in artifacts["recovery_actions"]
    )


def test_operational_recovery_actions_include_neuromorphic_profile_regression_hint():
    module = _load_script("operational_readiness.py")
    checks = {
        "phase3_accuracy": {"passed": True, "errors": []},
        "phase3_completion": {"passed": True, "errors": []},
        "phase4_completion": {"passed": True, "errors": []},
        "release_gate": {"passed": True, "errors": []},
        "production_profile": {"passed": True, "errors": []},
    }

    artifacts = module.collect_operational_readiness_artifacts(
        checks,
        stage_b_promotion={},
        neuromorphic_profile={
            "trend_regression_count": 2,
            "trend_policy_change_count": 0,
            "trend_missing_profiles": ["akida"],
        },
        execution_log=[],
        strict_production=False,
    )

    commands = [
        str(action.get("command", ""))
        for action in artifacts["recovery_actions"]
        if isinstance(action, dict)
    ]
    titles = [
        str(action.get("title", ""))
        for action in artifacts["recovery_actions"]
        if isinstance(action, dict)
    ]
    assert any("energy_efficiency_benchmark.py --no-history-update" in command for command in commands)
    assert "Inspect Neuromorphic Profile Regression" in titles
    assert any(
        isinstance(action, dict)
        and "missing profiles" in str(action.get("expected_effect", ""))
        for action in artifacts["recovery_actions"]
    )


def test_neuromorphic_profile_operational_snapshot_compacts_regression_details():
    module = _load_script("operational_readiness.py")
    phase3_report = _build_phase3_report(True)
    energy = phase3_report["component_reports"]["energy_efficiency"]
    energy["neuromorphic_profile_trend"] = {
        "has_previous": True,
        "regression_count": 3,
        "policy_change_count": 1,
        "new_profiles": [],
        "missing_profiles": ["spinnaker"],
        "regressions": [
            {"profile": "akida", "kind": "compatibility_regression"},
            {
                "profile": "akida",
                "kind": "check_regression",
                "check": "low_precision_weight_ok",
            },
            {"profile": "spinnaker", "kind": "missing_profile"},
        ],
        "policy_changes": [
            {
                "profile": "akida",
                "previous": "freeze_state_for_inference_profile",
                "current": "native_online_update",
            }
        ],
    }

    _, report = module._evaluate_operational_readiness(
        phase3_report=phase3_report,
        phase4_report=_build_phase4_report(True),
        release_report=_build_release_report(True),
        phase5_entry_gate_report=_build_phase5_entry_gate_report(True),
        phase5_completion_gate_report=_build_phase5_completion_gate_report(True),
    )
    summary = module.format_operational_summary(report)

    assert report["neuromorphic_profile_readiness"]["trend_regression_details"] == [
        "akida:compatibility_regression",
        "akida:check_regression:low_precision_weight_ok",
        "spinnaker:missing_profile",
    ]
    assert "- neuromorphic_profile_trend_regression_count: 3" in summary
    assert (
        "- neuromorphic_profile_trend_regression_details: "
        "akida:compatibility_regression,akida:check_regression:low_precision_weight_ok,spinnaker:missing_profile"
        in summary
    )
    assert (
        "- neuromorphic_profile_trend_policy_change_details: "
        "akida:freeze_state_for_inference_profile->native_online_update"
        in summary
    )
    assert (
        "- neuromorphic_profile_recovery_hint: "
        "Re-run edge neuromorphic export validation and inspect missing backend profiles."
        in summary
    )


def test_append_operational_repair_entry_finalizes_pending():
    module = _load_script("operational_readiness.py")
    entries = [
        {
            "command": "python scripts/eval/phase3_accuracy_suite.py",
            "status": "pending",
            "covered_checks": ["phase3_accuracy"],
            "timestamp": 1.0,
        }
    ]
    updated = module.append_operational_repair_execution_entry(
        entries,
        command="python scripts/eval/phase3_accuracy_suite.py",
        status="success",
        covered_checks=["phase3_completion"],
        source="test",
    )
    assert updated is True
    assert entries[0]["status"] == "success"
    assert sorted(entries[0]["covered_checks"]) == ["phase3_accuracy", "phase3_completion"]
    assert entries[0]["source"] == "test"
    assert "resolved_timestamp" in entries[0]


def test_operational_finalize_pending_entries_returns_zero_when_no_match():
    module = _load_script("operational_readiness.py")
    entries = [
        {
            "command": "python scripts/eval/release_gate.py",
            "status": "pending",
            "timestamp": 1.0,
        }
    ]
    completed = module._finalize_pending_operational_repair_entries(
        entries,
        command="python scripts/eval/phase3_accuracy_suite.py",
        status="success",
        covered_checks=["phase3_accuracy"],
        source="test_completion",
    )
    assert completed == 0
    assert entries[0]["status"] == "pending"


def test_append_operational_repair_entry_appends_new_entry():
    module = _load_script("operational_readiness.py")
    entries = []
    updated = module.append_operational_repair_execution_entry(
        entries,
        command="python scripts/eval/release_gate.py",
        status="pending",
        covered_checks=["release_gate"],
        source="manual",
    )
    assert updated is True
    assert len(entries) == 1
    assert entries[0]["command"] == "python scripts/eval/release_gate.py"
    assert entries[0]["status"] == "pending"
    assert entries[0]["covered_checks"] == ["release_gate"]


def test_record_roadmap_patch_review_decision_finalizes_pending_and_suppresses_action():
    module = _load_script("operational_readiness.py")
    entries = [
        {
            "command": "python scripts/eval/research_automation_benchmark.py --append-journal",
            "status": "pending",
            "covered_checks": ["research_review", "roadmap_patch_suggestion"],
            "source": "runbook_action:roadmap_patch_review",
            "timestamp": 100.0,
        }
    ]

    recorded = module.record_roadmap_patch_review_decision(
        entries,
        decision="rejected",
        reason="Needs stronger evidence.",
    )

    assert recorded >= 1
    assert entries[0]["status"] == "skipped"
    assert entries[0]["source"] == "roadmap_patch_review"
    assert entries[0]["roadmap_patch_review_decision"] == "rejected"
    assert entries[0]["roadmap_patch_review_reason"] == "Needs stronger evidence."

    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "execution_log": entries,
        "research_review": {
            "compact": {
                "passed": False,
                "next_hypothesis_count": 1,
                "regression_watchlist_count": 0,
                "negative_result_count": 1,
            },
            "report": {"generated_at": 1.0},
        },
    }
    actions = module.build_operational_runbook_actions(report)
    assert all(action["source"] != "roadmap_patch_review" for action in actions)


def test_record_roadmap_patch_review_decision_allows_newer_review_action():
    module = _load_script("operational_readiness.py")
    entries = [
        {
            "command": "python scripts/eval/research_automation_benchmark.py --append-journal",
            "status": "skipped",
            "covered_checks": ["research_review", "roadmap_patch_suggestion"],
            "source": "roadmap_patch_review",
            "timestamp": 100.0,
            "resolved_timestamp": 100.0,
            "roadmap_patch_review_decision": "rejected",
        }
    ]
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "execution_log": entries,
        "research_review": {
            "compact": {
                "passed": False,
                "next_hypothesis_count": 1,
                "regression_watchlist_count": 0,
                "negative_result_count": 0,
            },
            "report": {"generated_at": 200.0},
        },
    }
    actions = module.build_operational_runbook_actions(report)
    assert any(action["source"] == "roadmap_patch_review" for action in actions)


def test_completed_evidence_collection_keys_surface_in_runbook_and_actions():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "execution_log": [],
        "research_journal_summary": {
            "completed_roadmap_patch_evidence_collection_count": 1,
            "completed_roadmap_patch_evidence_collection_keys": [
                "predictive_spike_entropy_reduction_observed:real_data_fixture"
            ],
            "roadmap_patch_refreshed_items": [],
        },
    }

    actions = module.build_operational_runbook_actions(report)
    runbook = module.build_operational_runbook({**report, "runbook_actions": actions})

    assert any(action["source"] == "roadmap_patch_evidence_review" for action in actions)
    assert any(
        "predictive_spike_entropy_reduction_observed:real_data_fixture"
        in ",".join(action.get("affected_checks", []))
        for action in actions
    )
    assert "- Completed evidence collection pending review count: 1" in runbook
    assert (
        "- completed_evidence_pending_review_key: "
        "predictive_spike_entropy_reduction_observed:real_data_fixture"
    ) in runbook


def test_build_operational_runbook_actions_adds_observed_trend_long_run_validation():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "execution_log": [],
        "stage_e_readiness": {
            "linear_snn_fusion_trend_has_previous": False,
            "delta_memory_steering_integrity_observed": 1.0,
            "delta_memory_counterfactual_isolation_observed": 1.0,
            "delta_memory_trace_observability_observed": 1.0,
        },
    }

    actions = module.build_operational_runbook_actions(report)

    action = next(
        item for item in actions
        if item["source"] == "observed_trend_long_run_validation"
    )
    assert action["priority"] == "medium"
    assert "--soak-profile extended" in action["command"]
    assert "linear_snn_fusion_observed_trend" in action["affected_checks"]
    assert "stage_e_architecture_integration_observed_trend" in action["affected_checks"]
    assert "delta_memory_trace_observability_observed" in action["affected_checks"]
    assert "real_data_external_validity" in action["affected_checks"]


def test_build_operational_runbook_actions_adds_stage_e_architecture_regression_validation():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "execution_log": [],
        "stage_e_readiness": {
            "linear_snn_fusion_trend_has_previous": True,
            "architecture_integration_trend_has_previous": True,
            "architecture_integration_trend_regression_count": 2,
        },
    }

    actions = module.build_operational_runbook_actions(report)

    action = next(
        item for item in actions
        if item["source"] == "observed_trend_long_run_validation"
    )
    assert action["priority"] == "medium"
    assert "stage_e_architecture_integration_observed_trend" in action["affected_checks"]
    assert "stage_e_architecture_integration_regression_count=2" in action["reason"]


def test_build_operational_runbook_actions_adds_stage_e_acceptance_candidate_review():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "execution_log": [],
        "stage_e_readiness": {
            "linear_snn_fusion_trend_has_previous": True,
            "architecture_integration_trend_has_previous": True,
            "architecture_integration_trend_regression_count": 0,
            "observed_acceptance_candidate_stability_recommended": True,
            "observed_acceptance_candidate_consecutive_passes": 3,
            "observed_acceptance_candidate_required_streak": 3,
        },
    }

    actions = module.build_operational_runbook_actions(report)

    action = next(
        item for item in actions
        if item["source"] == "stage_e_observed_acceptance_candidate_stability"
    )
    assert action["priority"] == "medium"
    assert "review_stage_e_observed_acceptance_candidates_for_minimum_promotion" in action["command"]
    assert "consecutive_passes=3" in action["reason"]
    assert action["affected_checks"] == [
        "stage_e_observed_acceptance_candidate_stability",
        "stage_e_readiness",
    ]


def test_build_operational_runbook_actions_adds_stage_e_acceptance_candidate_repair():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "execution_log": [],
        "stage_e_readiness": {
            "linear_snn_fusion_trend_has_previous": True,
            "architecture_integration_trend_has_previous": True,
            "architecture_integration_trend_regression_count": 0,
            "observed_acceptance_candidate_failure_count": 1,
            "observed_acceptance_candidate_failures": [
                {
                    "metric": "micro_turn_event_budget_observed",
                    "check": "metric.micro_turn_event_budget_observed",
                }
            ],
        },
    }

    actions = module.build_operational_runbook_actions(report)

    action = next(
        item for item in actions
        if item["source"] == "stage_e_observed_acceptance_candidate_repair"
    )
    assert action["priority"] == "medium"
    assert "repair_stage_e_observed_acceptance_candidates:micro_turn_event_budget_observed" in action["command"]
    assert "stage_e_observed_acceptance_candidate_failure_count=1" in action["reason"]
    assert "micro_turn_event_budget_observed" in action["affected_checks"]


def test_operational_retry_queue_builds_from_failed_entries():
    module = _load_script("operational_readiness.py")
    now = 200.0
    entries = [
        {
            "command": "python scripts/eval/release_gate.py",
            "status": "failed",
            "covered_checks": ["release_gate"],
            "timestamp": now - 50.0,
        }
    ]
    queue = module.build_operational_retry_queue_from_repair_log(
        entries,
        max_attempts=2,
        cooldown_seconds=0.0,
        now_timestamp=now,
    )
    assert len(queue) == 1
    assert queue[0]["command"] == "python scripts/eval/release_gate.py"
    assert queue[0]["next_attempt"] == 2


def test_operational_retry_cooldown_blocked_respects_window():
    module = _load_script("operational_readiness.py")
    now = 300.0
    entries = [
        {
            "command": "python scripts/eval/release_gate.py",
            "status": "timeout",
            "covered_checks": ["release_gate"],
            "timestamp": now - 5.0,
        }
    ]
    blocked = module.build_operational_retry_cooldown_blocked_from_repair_log(
        entries,
        max_attempts=2,
        cooldown_seconds=10.0,
        now_timestamp=now,
    )
    assert len(blocked) == 1
    assert blocked[0]["command"] == "python scripts/eval/release_gate.py"
    assert blocked[0]["cooldown_remaining_seconds"] > 0.0


def test_operational_dispatch_retry_queue_to_pending_with_report():
    module = _load_script("operational_readiness.py")
    entries = []
    retry_queue = [
        {
            "command": "python scripts/eval/release_gate.py",
            "covered_checks": ["release_gate"],
        }
    ]
    report = module.dispatch_operational_retry_queue_to_pending_with_report(
        entries,
        retry_queue,
        max_dispatch=1,
    )
    assert report["dispatched"] == 1
    assert report["dispatched_commands"] == ["python scripts/eval/release_gate.py"]
    assert len(entries) == 1
    assert entries[0]["status"] == "pending"


def test_operational_dispatch_retry_queue_skips_existing_pending_command():
    module = _load_script("operational_readiness.py")
    entries = [
        {
            "command": "python scripts/eval/release_gate.py",
            "status": "pending",
            "covered_checks": ["release_gate"],
        }
    ]
    retry_queue = [
        {
            "command": "python scripts/eval/release_gate.py",
            "covered_checks": ["release_gate"],
        }
    ]
    report = module.dispatch_operational_retry_queue_to_pending_with_report(
        entries,
        retry_queue,
        max_dispatch=1,
    )
    assert report["dispatched"] == 0
    assert report["skipped_pending_commands"] == ["python scripts/eval/release_gate.py"]


def test_operational_select_dispatch_batch_with_priority_threshold():
    module = _load_script("operational_readiness.py")
    retry_queue = [
        {"command": "cmd_high", "priority_tier": "high", "covered_checks": ["a"]},
        {"command": "cmd_medium", "priority_tier": "medium", "covered_checks": ["b"]},
        {"command": "cmd_low", "priority_tier": "low", "covered_checks": ["c"]},
    ]
    batch = module.select_operational_retry_dispatch_batch(
        retry_queue,
        max_dispatch=3,
        min_priority_tier="medium",
    )
    selected_commands = [item["command"] for item in batch["selected"]]
    assert "cmd_high" in selected_commands
    assert "cmd_medium" in selected_commands
    assert "cmd_low" not in selected_commands
    assert batch["skipped_low_priority_count"] == 1


def test_prioritize_operational_retry_queue_boosts_critical_energy_measurement_repair():
    module = _load_script("operational_readiness.py")
    retry_queue = [
        {
            "command": "python scripts/eval/phase3_accuracy_suite.py",
            "reason": "failed",
            "covered_checks": ["phase3_accuracy"],
            "attempts_used": 0,
            "max_attempts": 2,
            "source": "manual",
            "severity": "",
        },
        {
            "command": "python scripts/sara_cli.py run-physical-energy-pair --pair-id p1",
            "reason": "failed",
            "covered_checks": ["ann_efficiency_roadmap", "energy_measurement"],
            "attempts_used": 0,
            "max_attempts": 2,
            "source": "runbook_action:ann_efficiency_next_evidence",
            "severity": "critical",
        },
    ]

    prioritized = module.prioritize_operational_retry_queue(
        retry_queue,
        iterative_plan={"remaining_checks": ["energy_measurement"]},
    )

    assert prioritized[0]["command"] == "python scripts/sara_cli.py run-physical-energy-pair --pair-id p1"
    assert prioritized[0]["priority_tier"] == "high"
    assert prioritized[0]["priority_urgency_bonus"] >= 2.75


def test_operational_summary_includes_auto_dispatch_command_breakdown():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "error_count": 1,
        "readiness_score": 0.0,
        "strict_production": False,
        "checks": {
            "phase3_accuracy": {"passed": False, "errors": ["e"]},
            "phase3_completion": {"passed": True, "errors": []},
            "phase4_completion": {"passed": True, "errors": []},
            "release_gate": {"passed": False, "errors": ["e2"]},
            "production_profile": {"passed": True, "errors": []},
        },
        "repair_retry_queue_count": 1,
        "repair_retry_queue": [
            {
                "command": "python scripts/eval/release_gate.py",
                "reason": "failed",
                "next_attempt": 2,
                "max_attempts": 3,
                "priority_tier": "high",
                "priority_score": 5.5,
                "covered_checks": ["release_gate"],
            }
        ],
        "repair_retry_cooldown_blocked_count": 0,
        "repair_retry_cooldown_blocked": [],
        "repair_retry_cooldown_seconds": 0.0,
        "repair_pending_count": 0,
        "repair_timeout_count": 0,
        "iterative_repair_plan": {"completed": False, "stalled": False, "stop_reason": "pending_actions", "next_step_hint": "x", "next_actions": []},
        "repair_plan": {"estimated_steps": 0, "covered_checks": [], "uncovered_checks": [], "fallback_actions": []},
        "stage_b_promotion": {},
        "stage_e_readiness": {},
        "failure_focus": {"primary_category": "", "secondary_category": "", "confidence": 0.0},
        "repair_auto_dispatch": {
            "requested": 2,
            "candidate_count": 3,
            "eligible_count": 2,
            "selected_count": 2,
            "selected_unique_check_count": 2,
            "min_priority_tier": "medium",
            "selection_mode": "priority_diversified",
            "max_per_check": 1,
            "dispatched": 1,
            "dispatched_commands": ["python scripts/eval/release_gate.py"],
            "skipped_pending_commands": ["python scripts/eval/phase3_accuracy_suite.py"],
            "skipped_limit_commands": ["python scripts/eval/release_soak.py --profile release --include-accuracy"],
            "skipped_low_priority_commands": ["python scripts/eval/future_state_consistency.py"],
            "skipped_check_quota_commands": ["python scripts/eval/operational_readiness.py"],
            "skipped_low_priority_count": 1,
            "skipped_check_quota_count": 1,
        },
    }
    summary = module.format_operational_summary(report)
    assert "- auto_dispatch_command: python scripts/eval/release_gate.py" in summary
    assert "- auto_dispatch_skipped_pending_command: python scripts/eval/phase3_accuracy_suite.py" in summary
    assert "- auto_dispatch_skipped_limit_command: python scripts/eval/release_soak.py --profile release --include-accuracy" in summary
    assert "- auto_dispatch_skipped_low_priority_command: python scripts/eval/future_state_consistency.py" in summary
    assert "- auto_dispatch_skipped_check_quota_command: python scripts/eval/operational_readiness.py" in summary


def test_collect_operational_checklist_status_marks_managed_paths():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": True,
        "checks": {},
        "repair_plan": {},
        "iterative_repair_plan": {},
    }
    checklist = module.collect_operational_checklist_status(
        report,
        report_path="workspace/release/operational_readiness_report.json",
        summary_path="workspace/release/operational_readiness_summary.txt",
        repair_plan_path="workspace/release/operational_repair_plan.json",
        runbook_path="workspace/release/operational_readiness_runbook.md",
        runbook_actions_path="workspace/release/operational_readiness_runbook_actions.json",
    )
    assert checklist["managed_output_paths_ok"] is True
    assert checklist["report_summary_review_ready"] is True
    assert checklist["runbook_manifest_hygiene_ok"] is True
    assert checklist["runbook_drop_rate_ok"] is True
    assert checklist["runbook_drop_rate_threshold"] == 0.9
    assert checklist["efficiency_shortcut_action_count"] == 0
    assert checklist["efficiency_shortcut_action_threshold"] == 3
    assert checklist["efficiency_shortcut_action_ok"] is True
    assert checklist["passed"] is True


def test_build_operational_repair_artifact_includes_checklist_snapshot():
    module = _load_script("operational_readiness.py")
    output = {
        "checks": {"phase3_accuracy": {"passed": True, "errors": []}},
        "repair_plan": {"estimated_steps": 0},
        "iterative_repair_plan": {"completed": True},
        "operational_checklist": {"passed": True},
        "runbook_actions": [{"step": 1, "command": "python scripts/eval/release_gate.py"}],
        "runbook_actions_path": "/tmp/actions.json",
        "refresh_results": [],
        "generated_at": 1.0,
    }
    artifact = module._build_operational_repair_artifact(output)
    assert artifact["checks"] == output["checks"]
    assert artifact["repair_plan"] == output["repair_plan"]
    assert artifact["iterative_repair_plan"] == output["iterative_repair_plan"]
    assert artifact["operational_checklist"] == output["operational_checklist"]
    assert artifact["runbook_actions"] == output["runbook_actions"]


def test_collect_operational_checklist_status_fails_for_unmanaged_paths():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": True,
        "checks": {},
        "repair_plan": {},
        "iterative_repair_plan": {},
    }
    checklist = module.collect_operational_checklist_status(
        report,
        report_path="/tmp/operational_readiness_report.json",
        summary_path="/tmp/operational_readiness_summary.txt",
        repair_plan_path="/tmp/operational_repair_plan.json",
        runbook_path="/tmp/operational_readiness_runbook.md",
        runbook_actions_path="/tmp/operational_readiness_runbook_actions.json",
    )
    assert checklist["managed_output_paths_ok"] is False
    assert checklist["report_summary_review_ready"] is False
    assert checklist["runbook_manifest_hygiene_ok"] is True
    assert checklist["runbook_drop_rate_ok"] is True
    assert checklist["runbook_drop_rate_threshold"] == 0.9
    assert checklist["efficiency_shortcut_action_ok"] is True
    assert checklist["passed"] is False


def test_collect_operational_checklist_status_flags_high_drop_rate():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": True,
        "checks": {},
        "repair_plan": {},
        "iterative_repair_plan": {},
        "runbook_action_build_rates": {"drop_rate": 0.95},
    }
    checklist = module.collect_operational_checklist_status(
        report,
        report_path="workspace/release/operational_readiness_report.json",
        summary_path="workspace/release/operational_readiness_summary.txt",
        repair_plan_path="workspace/release/operational_repair_plan.json",
        runbook_path="workspace/release/operational_readiness_runbook.md",
        runbook_actions_path="workspace/release/operational_readiness_runbook_actions.json",
    )
    assert checklist["runbook_drop_rate_ok"] is False
    assert checklist["runbook_drop_rate_threshold"] == 0.9
    assert checklist["passed"] is True


def test_collect_operational_checklist_status_respects_custom_drop_rate_threshold():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": True,
        "checks": {},
        "repair_plan": {},
        "iterative_repair_plan": {},
        "runbook_action_build_rates": {"drop_rate": 0.75},
    }
    checklist = module.collect_operational_checklist_status(
        report,
        report_path="workspace/release/operational_readiness_report.json",
        summary_path="workspace/release/operational_readiness_summary.txt",
        repair_plan_path="workspace/release/operational_repair_plan.json",
        runbook_path="workspace/release/operational_readiness_runbook.md",
        runbook_actions_path="workspace/release/operational_readiness_runbook_actions.json",
        runbook_drop_rate_threshold=0.7,
    )
    assert checklist["runbook_drop_rate_ok"] is False
    assert checklist["runbook_drop_rate_threshold"] == 0.7


def test_collect_operational_checklist_status_flags_efficiency_shortcut_overuse():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": True,
        "checks": {},
        "repair_plan": {},
        "iterative_repair_plan": {},
        "runbook_action_summary": {
            "total_actions": 4,
            "source_counts": {"efficiency_incident_shortcut": 4},
            "priority_counts": {"high": 4},
        },
    }
    checklist = module.collect_operational_checklist_status(
        report,
        report_path="workspace/release/operational_readiness_report.json",
        summary_path="workspace/release/operational_readiness_summary.txt",
        repair_plan_path="workspace/release/operational_repair_plan.json",
        runbook_path="workspace/release/operational_readiness_runbook.md",
        runbook_actions_path="workspace/release/operational_readiness_runbook_actions.json",
        efficiency_shortcut_action_threshold=3,
    )
    assert checklist["efficiency_shortcut_action_count"] == 4
    assert checklist["efficiency_shortcut_action_threshold"] == 3
    assert checklist["efficiency_shortcut_action_ok"] is False


def test_collect_operational_checklist_status_flags_efficiency_shortcut_overuse_rate():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": True,
        "checks": {},
        "repair_plan": {},
        "iterative_repair_plan": {},
        "runbook_action_summary": {
            "total_actions": 4,
            "source_counts": {"efficiency_incident_shortcut": 1},
            "priority_counts": {"high": 1},
        },
        "efficiency_shortcut_overuse_timeline": [
            {"overuse_active": True},
            {"overuse_active": True},
            {"overuse_active": False},
        ],
    }
    checklist = module.collect_operational_checklist_status(
        report,
        report_path="workspace/release/operational_readiness_report.json",
        summary_path="workspace/release/operational_readiness_summary.txt",
        repair_plan_path="workspace/release/operational_repair_plan.json",
        runbook_path="workspace/release/operational_readiness_runbook.md",
        runbook_actions_path="workspace/release/operational_readiness_runbook_actions.json",
        efficiency_shortcut_action_threshold=0,
        efficiency_shortcut_overuse_window=4,
        efficiency_shortcut_overuse_rate_threshold=0.5,
    )
    assert checklist["efficiency_shortcut_overuse_observed_window_size"] == 4
    assert checklist["efficiency_shortcut_overuse_count_in_window"] == 3
    assert checklist["efficiency_shortcut_overuse_rate"] == 0.75
    assert checklist["efficiency_shortcut_overuse_rate_ok"] is False


def test_operational_summary_includes_checklist_section():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": True,
        "error_count": 0,
        "readiness_score": 1.0,
        "strict_production": False,
        "checks": {
            "phase3_accuracy": {"passed": True, "errors": []},
            "phase3_completion": {"passed": True, "errors": []},
            "phase4_completion": {"passed": True, "errors": []},
            "release_gate": {"passed": True, "errors": []},
            "production_profile": {"passed": True, "errors": []},
        },
        "stage_b_promotion": {},
        "stage_e_readiness": {},
        "repair_plan": {"estimated_steps": 0, "covered_checks": [], "uncovered_checks": [], "fallback_actions": []},
        "iterative_repair_plan": {"completed": True, "stalled": False, "stop_reason": "auto_stopped_completed", "next_step_hint": "", "next_actions": []},
        "failure_focus": {},
        "repair_retry_queue_count": 0,
        "repair_retry_queue": [],
        "repair_retry_cooldown_seconds": 0.0,
        "repair_retry_cooldown_blocked_count": 0,
        "repair_retry_cooldown_blocked": [],
        "repair_pending_count": 0,
        "repair_timeout_count": 0,
        "repair_auto_dispatch": {"requested": 0, "candidate_count": 0, "eligible_count": 0, "selected_count": 0, "selected_unique_check_count": 0, "min_priority_tier": "low", "selection_mode": "priority", "max_per_check": 0, "dispatched": 0, "skipped_pending_commands": [], "skipped_limit_commands": [], "skipped_low_priority_count": 0, "skipped_check_quota_count": 0, "dispatched_commands": []},
        "operational_checklist": {
            "passed": True,
            "managed_output_paths_ok": True,
            "report_summary_review_ready": True,
            "runbook_manifest_hygiene_ok": True,
            "runbook_drop_rate_ok": True,
            "runbook_drop_rate_threshold": 0.9,
            "efficiency_shortcut_action_ok": True,
            "efficiency_shortcut_action_count": 0,
            "efficiency_shortcut_action_threshold": 3,
        },
    }
    summary = module.format_operational_summary(report)
    assert "Checklist" in summary
    assert "- managed_output_paths_ok: True" in summary
    assert "- report_summary_review_ready: True" in summary
    assert "- runbook_manifest_hygiene_ok: True" in summary
    assert "- runbook_drop_rate_ok: True" in summary
    assert "- runbook_drop_rate_threshold: 0.900" in summary
    assert "- efficiency_shortcut_action_ok: True" in summary
    assert "- efficiency_shortcut_action_count: 0" in summary
    assert "- efficiency_shortcut_action_threshold: 3" in summary
    assert "- efficiency_shortcut_overuse_rate_ok: True" in summary


def test_operational_append_iterative_next_actions_avoids_duplicates():
    module = _load_script("operational_readiness.py")
    entries = []
    iterative_plan = {
        "next_actions": [
            {
                "command": "python scripts/eval/release_gate.py",
                "title": "rerun release gate",
                "affected_checks": ["release_gate"],
            }
        ]
    }
    first = module.append_operational_iterative_next_actions_to_repair_log(entries, iterative_plan)
    second = module.append_operational_iterative_next_actions_to_repair_log(entries, iterative_plan)
    assert first == 1
    assert second == 0
    assert len(entries) == 1
    assert entries[0]["status"] == "pending"
    assert entries[0]["source"] == "iterative_next_action"


def test_append_operational_runbook_actions_to_repair_log_applies_priority_and_limit():
    module = _load_script("operational_readiness.py")
    entries = [
        {
            "command": "python scripts/eval/release_gate.py --skip-accuracy",
            "status": "pending",
            "covered_checks": ["release_gate"],
            "source": "manual",
        }
    ]
    runbook_actions = [
        {
            "command": "python scripts/eval/release_gate.py --skip-accuracy",
            "priority": "high",
            "source": "iterative_next_action",
            "affected_checks": ["release_gate"],
        },
        {
            "command": "python scripts/eval/release_soak.py --profile extended",
            "priority": "medium",
            "source": "retry_queue",
            "affected_checks": ["release_gate"],
        },
        {
            "command": "python scripts/eval/phase3_accuracy_suite.py",
            "priority": "low",
            "source": "fallback_action",
            "affected_checks": ["phase3_accuracy"],
        },
    ]
    appended = module.append_operational_runbook_actions_to_repair_log(
        entries,
        runbook_actions,
        max_append=1,
        min_priority="medium",
    )
    assert appended == 1
    assert len(entries) == 2
    assert entries[1]["command"] == "python scripts/eval/release_soak.py --profile extended"
    assert entries[1]["status"] == "pending"
    assert entries[1]["source"] == "runbook_action:retry_queue"


def test_append_operational_runbook_actions_to_repair_log_preserves_severity_metadata():
    module = _load_script("operational_readiness.py")
    entries = []
    runbook_actions = [
        {
            "command": "python scripts/sara_cli.py run-physical-energy-pair --pair-id p1",
            "priority": "high",
            "source": "ann_efficiency_next_evidence",
            "reason": (
                "category=weak_joule_pair; task=real_data_external_validity; "
                "severity=critical; ratio_gap=1.200; relative_ratio=0.400"
            ),
            "affected_checks": ["ann_efficiency_roadmap", "energy_measurement"],
        }
    ]

    appended = module.append_operational_runbook_actions_to_repair_log(
        entries,
        runbook_actions,
        max_append=1,
        min_priority="medium",
    )

    assert appended == 1
    assert entries[0]["severity"] == "critical"
    assert entries[0]["priority_hint"] == "high"
    assert "weak_joule_pair" in entries[0]["reason_hint"]


def test_append_efficiency_incident_repair_shortcut_appends_three_commands():
    module = _load_script("operational_readiness.py")
    entries = []
    appended = module.append_efficiency_incident_repair_shortcut(entries)
    assert appended == 3
    commands = [str(item.get("command", "")) for item in entries if isinstance(item, dict)]
    assert "python scripts/eval/energy_efficiency_benchmark.py" in commands
    assert "python scripts/eval/phase3_accuracy_suite.py" in commands
    assert "python scripts/eval/release_gate.py" in commands
    assert all(str(item.get("status", "")) == "pending" for item in entries if isinstance(item, dict))
    assert all(str(item.get("source", "")) == "efficiency_incident_shortcut" for item in entries if isinstance(item, dict))


def test_append_efficiency_incident_runbook_actions_appends_missing_commands():
    module = _load_script("operational_readiness.py")
    actions = [
        {
            "command": "python scripts/eval/release_gate.py",
            "priority": "medium",
            "source": "existing",
            "affected_checks": ["release_gate"],
        }
    ]
    appended = module.append_efficiency_incident_runbook_actions(actions)
    assert appended == 2
    commands = [str(item.get("command", "")) for item in actions if isinstance(item, dict)]
    assert "python scripts/eval/energy_efficiency_benchmark.py" in commands
    assert "python scripts/eval/phase3_accuracy_suite.py" in commands
    assert "python scripts/eval/release_gate.py" in commands
    appended_again = module.append_efficiency_incident_runbook_actions(actions)
    assert appended_again == 0


def test_main_record_efficiency_incident_repair_updates_log_and_manifest(tmp_path, monkeypatch):
    module = _load_script("operational_readiness.py")
    scoped_name = str(tmp_path.name).strip() or "pytest"
    output_dir = os.path.join(module.PROJECT_ROOT, "workspace", "tests", scoped_name)
    os.makedirs(output_dir, exist_ok=True)
    repair_log_path = os.path.join(output_dir, "operational_repair_execution_log.json")
    runbook_actions_path = os.path.join(output_dir, "operational_readiness_runbook_actions.json")
    with open(runbook_actions_path, "w", encoding="utf-8") as handle:
        handle.write(
        json.dumps(
            [
                {
                    "command": "python scripts/eval/release_gate.py",
                    "priority": "medium",
                    "source": "existing",
                    "affected_checks": ["release_gate"],
                }
            ],
            ensure_ascii=False,
        )
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "operational_readiness.py",
            "--record-efficiency-incident-repair",
            "--repair-log-path",
            repair_log_path,
            "--runbook-actions-path",
            runbook_actions_path,
        ],
    )

    rc = module.main()
    assert rc == 0

    with open(repair_log_path, "r", encoding="utf-8") as handle:
        saved_log = json.load(handle)
    assert isinstance(saved_log, list)
    log_commands = [str(item.get("command", "")) for item in saved_log if isinstance(item, dict)]
    assert "python scripts/eval/energy_efficiency_benchmark.py" in log_commands
    assert "python scripts/eval/phase3_accuracy_suite.py" in log_commands
    assert "python scripts/eval/release_gate.py" in log_commands

    with open(runbook_actions_path, "r", encoding="utf-8") as handle:
        saved_actions = json.load(handle)
    assert isinstance(saved_actions, list)
    action_commands = [str(item.get("command", "")) for item in saved_actions if isinstance(item, dict)]
    assert "python scripts/eval/energy_efficiency_benchmark.py" in action_commands
    assert "python scripts/eval/phase3_accuracy_suite.py" in action_commands
    assert "python scripts/eval/release_gate.py" in action_commands
    assert action_commands.count("python scripts/eval/release_gate.py") == 1


def test_append_tool_verification_trace_updates_trace_and_repair_log():
    module = _load_script("operational_readiness.py")
    traces = []
    repair_log = []

    result = module.append_tool_verification_trace(
        traces,
        repair_log,
        command="python -m pytest tests/test_sparse_verifier.py",
        status="success",
        covered_checks=["sparse_verifier"],
        source="tool_verification_trace",
        summary="Sparse verifier tests passed.",
        stdout_excerpt="1 passed",
        artifact_path=module.workspace_path("evaluation", "tool_verification_trace.json"),
    )

    assert result["appended"] is True
    assert traces[0]["schema"] == "sara-tool-verification-trace-v1"
    assert traces[0]["passed"] is True
    assert traces[0]["managed_output_policy"]["records_result_only"] is True
    assert repair_log[0]["covered_checks"] == ["sparse_verifier", "tool_verification_trace"]
    assert repair_log[0]["tool_verification_trace"]["passed"] is True
    assert repair_log[0]["tool_verification_trace"]["summary"] == "Sparse verifier tests passed."


def test_main_record_tool_verification_writes_managed_trace_and_repair_log(tmp_path, monkeypatch):
    module = _load_script("operational_readiness.py")
    scoped_name = str(tmp_path.name).strip() or "pytest"
    output_dir = os.path.join(module.PROJECT_ROOT, "workspace", "tests", scoped_name)
    os.makedirs(output_dir, exist_ok=True)
    repair_log_path = os.path.join(output_dir, "operational_repair_execution_log.json")
    trace_path = os.path.join(output_dir, "tool_verification_trace.json")
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "operational_readiness.py",
            "--record-tool-verification-command",
            "python -m pytest tests/test_sparse_verifier.py",
            "--record-tool-verification-status",
            "success",
            "--record-tool-verification-checks",
            "sparse_verifier",
            "--record-tool-verification-source",
            "tool_verification_trace",
            "--record-tool-verification-summary",
            "Sparse verifier tests passed.",
            "--tool-verification-trace-path",
            trace_path,
            "--repair-log-path",
            repair_log_path,
        ],
    )

    rc = module.main()
    assert rc == 0

    with open(trace_path, "r", encoding="utf-8") as handle:
        saved_trace = json.load(handle)
    with open(repair_log_path, "r", encoding="utf-8") as handle:
        saved_log = json.load(handle)

    assert saved_trace["schema"] == "sara-tool-verification-trace-log-v1"
    assert saved_trace["traces"][0]["summary"] == "Sparse verifier tests passed."
    assert saved_log[0]["source"] == "tool_verification_trace"
    assert saved_log[0]["tool_verification_trace"]["schema"] == "sara-tool-verification-trace-v1"


def test_operational_expire_pending_entries_marks_timeout():
    module = _load_script("operational_readiness.py")
    entries = [
        {"command": "python scripts/eval/release_gate.py", "status": "pending", "timestamp": 10.0},
        {"command": "python scripts/eval/phase3_accuracy_suite.py", "status": "pending", "timestamp": 95.0},
    ]
    expired = module.expire_pending_operational_repair_entries(
        entries,
        ttl_seconds=20.0,
        now_timestamp=100.0,
    )
    assert expired == 1
    assert entries[0]["status"] == "timeout"
    assert entries[0]["source"] == "pending_ttl_timeout"
    assert entries[1]["status"] == "pending"


def test_operational_evaluation_counts_pending_and_timeout_entries():
    module = _load_script("operational_readiness.py")
    _, summary = module._evaluate_operational_readiness(
        phase3_report=_build_phase3_report(False),
        phase4_report=_build_phase4_report(False),
        release_report=_build_release_report(False),
        execution_log=[
            {"command": "a", "status": "pending"},
            {"command": "b", "status": "timeout"},
            {"command": "c", "status": "failed"},
        ],
    )
    assert summary["repair_pending_count"] == 1
    assert summary["repair_timeout_count"] == 1


def test_build_operational_runbook_contains_focus_iterative_and_retry_sections():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "error_count": 2,
        "readiness_score": 0.25,
        "strict_production": True,
        "checks": {
            "phase3_accuracy": {"passed": False, "errors": ["x"]},
            "release_gate": {"passed": False, "errors": ["y"]},
        },
        "failure_focus": {
            "primary_category": "validation_error",
            "secondary_category": "gate_error",
            "primary_action": "rerun release gate",
            "confidence": 0.91,
        },
        "iterative_repair_plan": {
            "next_step_hint": "rerun phase3",
            "next_actions": [
                {
                    "title": "rerun release gate",
                    "command": "python scripts/eval/release_gate.py --skip-accuracy",
                    "affected_checks": ["release_gate"],
                }
            ],
        },
        "repair_retry_queue_count": 1,
        "repair_retry_queue": [
            {
                "command": "python scripts/eval/release_soak.py --profile extended",
                "reason": "timeout",
                "next_attempt": 2,
                "max_attempts": 3,
                "priority_tier": "high",
                "priority_score": 5.2,
            }
        ],
        "runbook_action_build_stats": {
            "considered_count": 4,
            "skipped_duplicate_count": 1,
            "skipped_duplicate_by_source": {"iterative_next_action": 1},
            "skipped_empty_command_count": 1,
            "skipped_empty_command_by_source": {"iterative_next_action": 1},
            "skipped_source_cap_count": 1,
            "skipped_source_cap_by_source": {"retry_queue": 1},
            "skipped_max_actions_count": 1,
            "skipped_max_actions_by_source": {"fallback_action": 1},
            "skipped_remeasure_command_history_quota_count": 1,
            "skipped_remeasure_command_history_quota_by_command": {
                "python scripts/eval/cognitive_runtime_benchmark.py": 1
            },
            "max_actions": 25,
            "max_per_source": 1,
        },
        "runbook_max_actions": 25,
        "runbook_max_per_source": 1,
    }
    runbook = module.build_operational_runbook(report)
    assert "# SARA Engine Operational Runbook" in runbook
    assert "## Failure Focus" in runbook
    assert "Primary category: validation_error" in runbook
    assert "## Iterative Next Actions" in runbook
    assert "`python scripts/eval/release_gate.py --skip-accuracy`" in runbook
    assert "## Retry Queue" in runbook
    assert "reason=timeout" in runbook
    assert "## Failed Checks" in runbook
    assert "- phase3_accuracy" in runbook
    assert "## Execution Manifest" in runbook
    assert "Configured max actions: 25" in runbook
    assert "Configured max per source: 1" in runbook
    assert "Considered candidates: 4" in runbook
    assert "Skipped by duplicate: 1" in runbook
    assert "Skipped by empty command: 1" in runbook
    assert "Skipped by remeasure command history quota: 1" in runbook
    assert "Skipped by empty command (iterative_next_action): 1" in runbook
    assert "Skipped by duplicate (iterative_next_action): 1" in runbook
    assert "Skipped by source cap: 1" in runbook
    assert "Skipped by source cap (retry_queue): 1" in runbook
    assert "Skipped by max actions (fallback_action): 1" in runbook
    assert (
        "Skipped by remeasure command history quota "
        "(python scripts/eval/cognitive_runtime_benchmark.py): 1"
    ) in runbook
    assert "source=iterative_next_action" in runbook


def test_build_operational_runbook_includes_efficiency_shortcut_action_count():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "error_count": 1,
        "readiness_score": 0.2,
        "strict_production": True,
        "checks": {},
        "failure_focus": {},
        "iterative_repair_plan": {"next_step_hint": "", "next_actions": []},
        "repair_retry_queue_count": 0,
        "repair_retry_queue": [],
        "runbook_actions": [
            {"command": "python scripts/eval/energy_efficiency_benchmark.py", "source": "efficiency_incident_shortcut"},
            {"command": "python scripts/eval/phase3_accuracy_suite.py", "source": "efficiency_incident_shortcut"},
        ],
        "runbook_action_build_stats": {},
    }
    runbook = module.build_operational_runbook(report)
    assert "- Efficiency incident shortcut actions: 2" in runbook


def test_build_operational_runbook_includes_efficiency_shortcut_overuse_section():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "error_count": 1,
        "readiness_score": 0.2,
        "strict_production": True,
        "checks": {},
        "failure_focus": {},
        "iterative_repair_plan": {"next_step_hint": "", "next_actions": []},
        "repair_retry_queue_count": 0,
        "repair_retry_queue": [],
        "runbook_actions": [],
        "runbook_action_build_stats": {},
        "operational_checklist": {
            "efficiency_shortcut_action_ok": False,
            "efficiency_shortcut_action_count": 6,
            "efficiency_shortcut_action_threshold": 3,
        },
    }
    runbook = module.build_operational_runbook(report)
    assert "## Efficiency Shortcut Overuse Incident" in runbook
    assert "- Shortcut action count: 6" in runbook
    assert "- Threshold: 3" in runbook
    assert "python scripts/eval/operational_readiness.py --strict-production" in runbook


def test_build_operational_runbook_prefers_report_runbook_actions():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "error_count": 1,
        "readiness_score": 0.5,
        "strict_production": True,
        "checks": {"release_gate": {"passed": False, "errors": ["x"]}},
        "failure_focus": {},
        "iterative_repair_plan": {"next_step_hint": "", "next_actions": []},
        "repair_retry_queue_count": 0,
        "repair_retry_queue": [],
        "runbook_actions": [
            {
                "source": "fallback_action",
                "priority": "medium",
                "command": "python scripts/eval/custom_action.py",
            }
        ],
        "runbook_action_build_stats": {},
    }
    runbook = module.build_operational_runbook(report)
    assert "python scripts/eval/custom_action.py" in runbook


def test_build_operational_runbook_includes_efficiency_kpi_incident_section():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "error_count": 2,
        "readiness_score": 0.5,
        "strict_production": True,
        "checks": {"phase3_accuracy": {"passed": False, "errors": ["x"]}},
        "failure_focus": {"primary_category": "phase3_efficiency_kpi", "secondary_category": "", "confidence": 0.9},
        "iterative_repair_plan": {"next_step_hint": "", "next_actions": []},
        "repair_retry_queue_count": 0,
        "repair_retry_queue": [],
        "runbook_actions": [],
        "runbook_action_build_stats": {},
        "runbook_action_build_rates": {},
        "error_details": [
            {
                "index": 1,
                "type": "check_failure",
                "category": "phase3_efficiency_kpi",
                "error": "Phase 3 efficiency_readiness did not satisfy performance-per-energy ratio proxy.",
            },
            {
                "index": 2,
                "type": "check_failure",
                "category": "phase3_efficiency_kpi",
                "error": "Phase 3 efficiency_readiness did not satisfy ANN-reference cost advantage proxy.",
            },
        ],
    }
    runbook = module.build_operational_runbook(report)
    assert "## Efficiency KPI Incident" in runbook
    assert "Failure count: 2" in runbook
    assert "Categories: phase3_efficiency_kpi" in runbook
    assert "performance-per-energy ratio proxy" in runbook
    assert "Immediate commands:" in runbook
    assert "python scripts/eval/energy_efficiency_benchmark.py" in runbook
    assert "python scripts/eval/phase3_accuracy_suite.py" in runbook
    assert "python scripts/eval/release_gate.py" in runbook


def test_build_operational_runbook_includes_roadmap_patch_preview():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "readiness_score": 0.6,
        "error_count": 1,
        "failure_focus": {},
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "checks": {},
        "research_review": {
            "compact": {
                "passed": False,
                "review_score": 0.4,
                "requires_human_approval": True,
                "next_hypothesis_count": 1,
                "regression_watchlist_count": 1,
                "negative_result_count": 2,
                "bounded_experiment_graph_node_count": 3,
                "bounded_experiment_graph_edge_count": 2,
                "cause_boundary_documentation_count": 1,
                "targeted_fixture_repair_count": 1,
            },
            "report": {
                "schema": "sara-research-review-report-v1",
                "generated_at": 123.0,
                "experiment_planner": {
                    "next_hypotheses": [{"id": "linear_snn_fusion_metric_recovery"}],
                    "regression_watchlist": [{"id": "release_gate_safety_review"}],
                    "cause_boundary_documentation_tasks": [
                        {"id": "predictive_spike_entropy_reduction_observed"}
                    ],
                    "targeted_fixture_repair_tasks": [
                        {"id": "phase_binding_coincidence_integrity_observed"}
                    ],
                },
            },
        },
        "execution_log": [
            {
                "command": "python scripts/eval/research_automation_benchmark.py --append-journal",
                "status": "skipped",
                "source": "roadmap_patch_review",
                "covered_checks": ["roadmap_patch_suggestion"],
                "resolved_timestamp": 200.0,
                "roadmap_patch_review_decision": "rejected",
                "roadmap_patch_review_reason": "Needs real-data evidence.",
            }
        ],
        "runbook_actions": [
            {
                "command": "python scripts/eval/research_automation_benchmark.py --append-journal",
                "source": "roadmap_patch_review",
                "priority": "high",
            }
        ],
    }

    runbook = module.build_operational_runbook(report)

    assert "## Roadmap Patch Review" in runbook
    assert "- Review status: NEEDS_REVIEW" in runbook
    assert "- Review score: 0.400" in runbook
    assert "- Requires human approval: True" in runbook
    assert "- Apply automatically: False" in runbook
    assert "- Cause boundary documentation count: 1" in runbook
    assert "- Targeted fixture repair count: 1" in runbook
    assert "- Planner task pending count: 2" in runbook
    assert "- Planner task completed count: 0" in runbook
    assert "- Planner task completion ratio: 0.000" in runbook
    assert "- Planner task cleanup needed: True" in runbook
    assert "- Planner task cleanup pending count: 0" in runbook
    assert "- Planner task cleanup stalled: False" in runbook
    assert "- Planner task cleanup stalled reason: " in runbook
    assert "- Review decision recorded: True" in runbook
    assert "- Review decision: rejected" in runbook
    assert "- Review decision reason: Needs real-data evidence." in runbook
    assert "NEXT: validate `linear_snn_fusion_metric_recovery`" in runbook
    assert "REVIEW: inspect `release_gate_safety_review`" in runbook
    assert "DOC: document targeted-probe boundary for `predictive_spike_entropy_reduction_observed`" in runbook
    assert "FIXTURE: add or repair minimal targeted fixture for `phase_binding_coincidence_integrity_observed`" in runbook


def test_operational_summary_and_runbook_include_research_journal_summary():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "readiness_score": 0.5,
        "error_count": 1,
        "strict_production": False,
        "checks": {},
        "stage_b_promotion": {},
        "stage_d_readiness": {},
        "stage_e_readiness": {},
        "phase5_entry_readiness": {},
        "neuromorphic_profile_readiness": {},
        "iterative_repair_plan": {"next_actions": []},
        "repair_plan": {},
        "failure_focus": {},
        "repair_retry_queue": [],
        "research_journal_summary": {
            "entry_count": 2,
            "total_seen_count": 3,
            "stale_age_seconds": 50.0,
            "remeasure_result_count": 2,
            "alternative_probe_result_count": 1,
            "remeasure_quota_hold_count": 1,
            "completed_research_planner_task_count": 1,
            "remeasure_status_counts": {"failed": 1, "success": 1},
            "remeasure_trends": [
                {
                    "id": "predictive_spike_entropy_reduction_observed",
                    "trend": "recovered",
                    "latest_status": "success",
                }
            ],
            "remeasure_quota_holds": [
                {
                    "id": "predictive_spike_entropy_reduction_observed",
                    "command": "python scripts/eval/cognitive_runtime_benchmark.py",
                    "history_count": 2,
                    "quota": 2,
                }
            ],
            "alternative_probe_trends": [
                {
                    "id": "predictive_spike_entropy_reduction_observed",
                    "trend": "targeted_probe_passed",
                    "latest_status": "success",
                }
            ],
            "completed_research_planner_tasks": [
                {
                    "id": "predictive_spike_entropy_reduction_observed",
                    "task_type": "cause_boundary_documentation",
                    "status": "success",
                }
            ],
            "roadmap_patch_review_approved_count": 1,
            "roadmap_patch_review_rejected_count": 1,
            "roadmap_patch_rejected_item_count": 1,
            "roadmap_patch_refreshed_item_count": 1,
            "roadmap_patch_refresh_to_rejection_ratio": 1.0,
            "roadmap_patch_rejected_items": [
                {
                    "id": "linear_snn_fusion_metric_recovery",
                    "count": 1,
                    "latest_reason": "Needs real-data evidence.",
                }
            ],
            "roadmap_patch_refreshed_items": [
                {
                    "id": "linear_snn_fusion_metric_recovery",
                    "count": 1,
                    "latest_reason": "targeted_probe_passed",
                }
            ],
            "stage_e_observed_acceptance_candidate_repair_loop": {
                "id": "stage_e_observed_acceptance_candidate_repair",
                "needs_followup": True,
                "remeasure_recommended": True,
                "remeasure_suppressed": False,
                "alternative_probe_recommended": True,
                "latest_remeasure_trend": "still_failing",
                "latest_alternative_probe_trend": "targeted_probe_failed",
            },
            "top_negative_results": [
                {"id": "predictive_spike_entropy_reduction_observed", "count": 2}
            ],
            "top_next_hypotheses": [
                {"id": "linear_snn_fusion_metric_recovery", "count": 2}
            ],
            "recommended_benchmark_actions": [
                {
                    "id": "predictive_spike_entropy_reduction_observed",
                    "source": "negative_result",
                    "command": "python scripts/eval/cognitive_runtime_benchmark.py",
                    "priority": "high",
                    "count": 2,
                }
            ],
            "suppressed_benchmark_actions": [
                {
                    "id": "linear_snn_fusion_metric_recovery",
                    "source": "next_hypothesis",
                    "command": "python scripts/eval/cognitive_runtime_benchmark.py",
                    "priority": "low",
                    "count": 2,
                    "remeasure_trend": "recovered",
                    "seconds_until_next_remeasure": 3600.0,
                }
            ],
            "alternative_benchmark_actions": [
                {
                    "id": "predictive_spike_entropy_reduction_observed",
                    "source": "remeasure_quota_hold",
                    "command": "PYTHONPATH=src workspace/.venv310/bin/python -m pytest -q tests/test_phase3_accuracy_benchmarks.py::test_cognitive_runtime_benchmark_returns_expected_metrics",
                    "priority": "high",
                }
            ],
        },
    }

    summary = module.format_operational_summary(report)
    runbook = module.build_operational_runbook(report)
    actions = module.build_operational_runbook_actions(report)

    assert "- research_journal_entry_count: 2" in summary
    assert "- research_journal_total_seen_count: 3" in summary
    assert "- research_journal_remeasure_result_count: 2" in summary
    assert "- research_journal_remeasure_success_count: 1" in summary
    assert "- research_journal_remeasure_failed_count: 1" in summary
    assert "- research_journal_alternative_probe_result_count: 1" in summary
    assert "- research_journal_completed_research_planner_task_count: 1" in summary
    assert "- research_journal_remeasure_quota_hold_count: 1" in summary
    assert "- research_journal_roadmap_patch_approved_count: 1" in summary
    assert "- research_journal_roadmap_patch_rejected_count: 1" in summary
    assert "- research_journal_roadmap_patch_rejected_item_count: 1" in summary
    assert "- research_journal_roadmap_patch_refreshed_item_count: 1" in summary
    assert "- research_journal_roadmap_patch_refresh_to_rejection_ratio: 1.000" in summary
    assert "- research_journal_roadmap_patch_refresh_policy_status: insufficient_history" in summary
    assert "- research_journal_roadmap_patch_refresh_policy_needs_followup: False" in summary
    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_needs_followup: True"
    ) in summary
    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_remeasure_recommended: True"
    ) in summary
    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_alternative_probe_recommended: True"
    ) in summary
    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_latest_remeasure_trend: still_failing"
    ) in summary
    assert "## Research Journal Summary" in runbook
    assert "- Entry count: 2" in runbook
    assert "- Remeasure result count: 2" in runbook
    assert "- remeasure_status_count: success=1" in runbook
    assert (
        "- remeasure_trend: predictive_spike_entropy_reduction_observed "
        "trend=recovered latest_status=success"
    ) in runbook
    assert (
        "- alternative_probe_trend: predictive_spike_entropy_reduction_observed "
        "trend=targeted_probe_passed latest_status=success"
    ) in runbook
    assert (
        "- completed_research_planner_task: predictive_spike_entropy_reduction_observed "
        "type=cause_boundary_documentation status=success"
    ) in runbook
    assert (
        "- remeasure_quota_hold: predictive_spike_entropy_reduction_observed "
        "command=python scripts/eval/cognitive_runtime_benchmark.py history=2/2"
    ) in runbook
    assert "- top_negative_result: predictive_spike_entropy_reduction_observed count=2" in runbook
    assert "- top_next_hypothesis: linear_snn_fusion_metric_recovery count=2" in runbook
    assert "- roadmap_patch_rejected_item: linear_snn_fusion_metric_recovery count=1 reason=Needs real-data evidence." in runbook
    assert "- roadmap_patch_refreshed_item: linear_snn_fusion_metric_recovery count=1 reason=targeted_probe_passed" in runbook
    assert "- Roadmap patch refresh policy status: insufficient_history" in runbook
    assert "- Roadmap patch refresh policy needs followup: False" in runbook
    assert "- Stage E observed acceptance candidate repair needs followup: True" in runbook
    assert "- Stage E observed acceptance candidate repair remeasure recommended: True" in runbook
    assert (
        "- Stage E observed acceptance candidate repair alternative probe recommended: True"
    ) in runbook
    assert (
        "- Stage E observed acceptance candidate repair latest remeasure trend: still_failing"
    ) in runbook
    assert "recommended_benchmark_action: python scripts/eval/cognitive_runtime_benchmark.py" in runbook
    assert "suppressed_benchmark_action: python scripts/eval/cognitive_runtime_benchmark.py" in runbook
    assert "trend=recovered" in runbook
    assert "alternative_benchmark_action: PYTHONPATH=src workspace/.venv310/bin/python -m pytest" in runbook
    assert actions[0]["source"] == "research_journal_remeasure"
    assert actions[0]["command"] == "python scripts/eval/cognitive_runtime_benchmark.py"


def test_stage_e_observed_candidate_repair_recovery_returns_to_stability_review():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "readiness_score": 0.5,
        "error_count": 1,
        "strict_production": False,
        "checks": {},
        "stage_b_promotion": {},
        "stage_d_readiness": {},
        "stage_e_readiness": {},
        "phase5_entry_readiness": {},
        "neuromorphic_profile_readiness": {},
        "iterative_repair_plan": {"next_actions": []},
        "repair_plan": {},
        "failure_focus": {},
        "repair_retry_queue": [],
        "research_journal_summary": {
            "entry_count": 1,
            "stage_e_observed_acceptance_candidate_repair_loop": {
                "id": "stage_e_observed_acceptance_candidate_repair",
                "needs_followup": False,
                "recovery_confirmed": True,
                "recovery_source": "remeasure,alternative_probe",
                "promotion_review_recommended": True,
                "next_review_action": "stage_e_observed_acceptance_candidate_stability",
                "latest_remeasure_trend": "recovered",
                "latest_alternative_probe_trend": "targeted_probe_passed",
            },
        },
    }

    summary = module.format_operational_summary(report)
    runbook = module.build_operational_runbook(report)
    actions = module.build_operational_runbook_actions(report)

    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_recovery_confirmed: True"
    ) in summary
    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_recommended: True"
    ) in summary
    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_next_review_action: "
        "stage_e_observed_acceptance_candidate_stability"
    ) in summary
    assert "- Stage E observed acceptance candidate repair recovery confirmed: True" in runbook
    assert (
        "- Stage E observed acceptance candidate repair promotion review recommended: True"
    ) in runbook
    recovery_actions = [
        item
        for item in actions
        if item["source"] == "stage_e_observed_acceptance_candidate_recovery_review"
    ]
    assert len(recovery_actions) == 1
    assert "review_stage_e_observed_acceptance_candidate_recovery_for_stability" in (
        recovery_actions[0]["command"]
    )
    assert "stage_e_observed_acceptance_candidate_stability" in recovery_actions[0]["affected_checks"]


def test_stage_e_observed_candidate_recovery_review_completion_suppresses_repeat_action():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "readiness_score": 0.5,
        "error_count": 1,
        "strict_production": False,
        "checks": {},
        "stage_b_promotion": {},
        "stage_d_readiness": {},
        "stage_e_readiness": {},
        "phase5_entry_readiness": {},
        "neuromorphic_profile_readiness": {},
        "iterative_repair_plan": {"next_actions": []},
        "repair_plan": {},
        "failure_focus": {},
        "repair_retry_queue": [],
        "execution_log": [
            {
                "command": "review_stage_e_observed_acceptance_candidate_recovery_for_stability",
                "status": "success",
                "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review",
                "covered_checks": [
                    "stage_e_observed_acceptance_candidate_repair_recovery",
                    "stage_e_observed_acceptance_candidate_stability",
                    "research_journal_summary",
                ],
                "resolved_timestamp": 900.0,
            }
        ],
        "research_journal_summary": {
            "entry_count": 1,
            "stage_e_observed_acceptance_candidate_repair_loop": {
                "id": "stage_e_observed_acceptance_candidate_repair",
                "needs_followup": False,
                "recovery_confirmed": True,
                "recovery_source": "remeasure",
                "promotion_review_recommended": True,
                "next_review_action": "stage_e_observed_acceptance_candidate_stability",
                "latest_remeasure_trend": "recovered",
            },
        },
    }

    summary = module.format_operational_summary(report)
    runbook = module.build_operational_runbook(report)
    actions = module.build_operational_runbook_actions(report)

    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_completed: True"
    ) in summary
    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_recommended: False"
    ) in summary
    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_latest_status: success"
    ) in summary
    assert (
        "- Stage E observed acceptance candidate repair promotion review completed: True"
    ) in runbook
    assert not any(
        item["source"] == "stage_e_observed_acceptance_candidate_recovery_review"
        for item in actions
    )


def test_stage_e_observed_candidate_recovery_review_stale_pending_adds_followup_action():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "readiness_score": 0.5,
        "error_count": 1,
        "strict_production": False,
        "checks": {},
        "stage_b_promotion": {},
        "stage_d_readiness": {},
        "stage_e_readiness": {},
        "phase5_entry_readiness": {},
        "neuromorphic_profile_readiness": {},
        "iterative_repair_plan": {"next_actions": []},
        "repair_plan": {},
        "failure_focus": {},
        "repair_retry_queue": [],
        "execution_log": [
            {
                "command": "review_stage_e_observed_acceptance_candidate_recovery_for_stability",
                "status": "pending",
                "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review",
                "covered_checks": [
                    "stage_e_observed_acceptance_candidate_repair_recovery",
                    "stage_e_observed_acceptance_candidate_stability",
                    "research_journal_summary",
                ],
                "timestamp": 1.0,
            }
        ],
        "research_journal_summary": {
            "entry_count": 1,
            "stage_e_observed_acceptance_candidate_repair_loop": {
                "id": "stage_e_observed_acceptance_candidate_repair",
                "needs_followup": False,
                "recovery_confirmed": True,
                "recovery_source": "remeasure",
                "promotion_review_recommended": True,
                "next_review_action": "stage_e_observed_acceptance_candidate_stability",
                "latest_remeasure_trend": "recovered",
            },
        },
    }

    summary = module.format_operational_summary(report)
    actions = module.build_operational_runbook_actions(report)

    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_in_progress: True"
    ) in summary
    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_stale: True"
    ) in summary
    assert not any(
        item["source"] == "stage_e_observed_acceptance_candidate_recovery_review"
        for item in actions
    )
    followups = [
        item
        for item in actions
        if item["source"] == "stage_e_observed_acceptance_candidate_recovery_review_followup"
    ]
    assert len(followups) == 1
    assert "followup_stage_e_observed_acceptance_candidate_recovery_review" in followups[0]["command"]
    assert "stage_e_observed_acceptance_candidate_recovery_review_stale=True" in followups[0]["reason"]


def test_stage_e_observed_candidate_recovery_review_followup_pending_suppresses_duplicate_followup():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "readiness_score": 0.5,
        "error_count": 1,
        "strict_production": False,
        "checks": {},
        "stage_b_promotion": {},
        "stage_d_readiness": {},
        "stage_e_readiness": {},
        "phase5_entry_readiness": {},
        "neuromorphic_profile_readiness": {},
        "iterative_repair_plan": {"next_actions": []},
        "repair_plan": {},
        "failure_focus": {},
        "repair_retry_queue": [],
        "execution_log": [
            {
                "command": "review_stage_e_observed_acceptance_candidate_recovery_for_stability",
                "status": "pending",
                "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review",
                "covered_checks": [
                    "stage_e_observed_acceptance_candidate_repair_recovery",
                    "stage_e_observed_acceptance_candidate_stability",
                    "research_journal_summary",
                ],
                "timestamp": 1.0,
            },
            {
                "command": "followup_stage_e_observed_acceptance_candidate_recovery_review",
                "status": "pending",
                "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_followup",
                "covered_checks": [
                    "stage_e_observed_acceptance_candidate_repair_recovery",
                    "stage_e_observed_acceptance_candidate_stability",
                    "research_journal_summary",
                ],
                "timestamp": 2.0,
            },
        ],
        "research_journal_summary": {
            "entry_count": 1,
            "stage_e_observed_acceptance_candidate_repair_loop": {
                "id": "stage_e_observed_acceptance_candidate_repair",
                "needs_followup": False,
                "recovery_confirmed": True,
                "recovery_source": "remeasure",
                "promotion_review_recommended": True,
                "next_review_action": "stage_e_observed_acceptance_candidate_stability",
                "latest_remeasure_trend": "recovered",
            },
        },
    }

    summary = module.format_operational_summary(report)
    actions = module.build_operational_runbook_actions(report)

    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_followup_in_progress: True"
    ) in summary
    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_followup_latest_status: pending"
    ) in summary
    assert not any(
        item["source"] == "stage_e_observed_acceptance_candidate_recovery_review_followup"
        for item in actions
    )


def test_stage_e_observed_candidate_recovery_review_followup_failure_adds_retry_action():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "readiness_score": 0.5,
        "error_count": 1,
        "strict_production": False,
        "checks": {},
        "stage_b_promotion": {},
        "stage_d_readiness": {},
        "stage_e_readiness": {},
        "phase5_entry_readiness": {},
        "neuromorphic_profile_readiness": {},
        "iterative_repair_plan": {"next_actions": []},
        "repair_plan": {},
        "failure_focus": {},
        "repair_retry_queue": [],
        "execution_log": [
            {
                "command": "followup_stage_e_observed_acceptance_candidate_recovery_review",
                "status": "failed",
                "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_followup",
                "covered_checks": [
                    "stage_e_observed_acceptance_candidate_repair_recovery",
                    "stage_e_observed_acceptance_candidate_stability",
                    "research_journal_summary",
                ],
                "resolved_timestamp": 900.0,
            }
        ],
        "research_journal_summary": {
            "entry_count": 1,
            "stage_e_observed_acceptance_candidate_repair_loop": {
                "id": "stage_e_observed_acceptance_candidate_repair",
                "needs_followup": False,
                "recovery_confirmed": True,
                "recovery_source": "remeasure",
                "latest_remeasure_trend": "recovered",
            },
        },
    }

    summary = module.format_operational_summary(report)
    actions = module.build_operational_runbook_actions(report)

    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_followup_failed: True"
    ) in summary
    retry_actions = [
        item
        for item in actions
        if item["source"] == "stage_e_observed_acceptance_candidate_recovery_review_followup_retry"
    ]
    assert len(retry_actions) == 1
    assert "retry_stage_e_observed_acceptance_candidate_recovery_review_followup" in (
        retry_actions[0]["command"]
    )
    assert "latest_followup_status=failed" in retry_actions[0]["reason"]


def test_stage_e_observed_candidate_recovery_review_followup_retry_pending_suppresses_duplicate_retry():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "readiness_score": 0.5,
        "error_count": 1,
        "strict_production": False,
        "checks": {},
        "stage_b_promotion": {},
        "stage_d_readiness": {},
        "stage_e_readiness": {},
        "phase5_entry_readiness": {},
        "neuromorphic_profile_readiness": {},
        "iterative_repair_plan": {"next_actions": []},
        "repair_plan": {},
        "failure_focus": {},
        "repair_retry_queue": [],
        "execution_log": [
            {
                "command": "followup_stage_e_observed_acceptance_candidate_recovery_review",
                "status": "failed",
                "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_followup",
                "covered_checks": [
                    "stage_e_observed_acceptance_candidate_repair_recovery",
                    "stage_e_observed_acceptance_candidate_stability",
                    "research_journal_summary",
                ],
                "resolved_timestamp": 900.0,
            },
            {
                "command": "retry_stage_e_observed_acceptance_candidate_recovery_review_followup",
                "status": "pending",
                "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_followup_retry",
                "covered_checks": [
                    "stage_e_observed_acceptance_candidate_repair_recovery",
                    "stage_e_observed_acceptance_candidate_stability",
                    "research_journal_summary",
                ],
                "resolved_timestamp": 950.0,
            },
        ],
        "research_journal_summary": {
            "entry_count": 1,
            "stage_e_observed_acceptance_candidate_repair_loop": {
                "id": "stage_e_observed_acceptance_candidate_repair",
                "needs_followup": False,
                "recovery_confirmed": True,
                "recovery_source": "remeasure",
                "latest_remeasure_trend": "recovered",
            },
        },
    }

    summary = module.format_operational_summary(report)
    actions = module.build_operational_runbook_actions(report)

    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_followup_retry_in_progress: True"
    ) in summary
    assert not [
        item
        for item in actions
        if item["source"] == "stage_e_observed_acceptance_candidate_recovery_review_followup_retry"
    ]


def test_stage_e_observed_candidate_recovery_review_followup_retry_failure_adds_escalation():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "readiness_score": 0.5,
        "error_count": 1,
        "strict_production": False,
        "checks": {},
        "stage_b_promotion": {},
        "stage_d_readiness": {},
        "stage_e_readiness": {},
        "phase5_entry_readiness": {},
        "neuromorphic_profile_readiness": {},
        "iterative_repair_plan": {"next_actions": []},
        "repair_plan": {},
        "failure_focus": {},
        "repair_retry_queue": [],
        "execution_log": [
            {
                "command": "followup_stage_e_observed_acceptance_candidate_recovery_review",
                "status": "failed",
                "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_followup",
                "covered_checks": [
                    "stage_e_observed_acceptance_candidate_repair_recovery",
                    "stage_e_observed_acceptance_candidate_stability",
                    "research_journal_summary",
                ],
                "resolved_timestamp": 900.0,
            },
            {
                "command": "retry_stage_e_observed_acceptance_candidate_recovery_review_followup",
                "status": "failed",
                "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_followup_retry",
                "covered_checks": [
                    "stage_e_observed_acceptance_candidate_repair_recovery",
                    "stage_e_observed_acceptance_candidate_stability",
                    "research_journal_summary",
                ],
                "resolved_timestamp": 950.0,
            },
        ],
        "research_journal_summary": {
            "entry_count": 1,
            "stage_e_observed_acceptance_candidate_repair_loop": {
                "id": "stage_e_observed_acceptance_candidate_repair",
                "needs_followup": False,
                "recovery_confirmed": True,
                "recovery_source": "remeasure",
                "latest_remeasure_trend": "recovered",
            },
        },
    }

    summary = module.format_operational_summary(report)
    actions = module.build_operational_runbook_actions(report)

    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_followup_retry_failed: True"
    ) in summary
    retry_actions = [
        item
        for item in actions
        if item["source"] == "stage_e_observed_acceptance_candidate_recovery_review_followup_retry"
    ]
    escalation_actions = [
        item
        for item in actions
        if item["source"]
        == "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation"
    ]
    assert retry_actions == []
    assert len(escalation_actions) == 1
    assert "escalate_stage_e_observed_acceptance_candidate_recovery_review_followup_retry" in (
        escalation_actions[0]["command"]
    )
    assert "latest_retry_status=failed" in escalation_actions[0]["reason"]


def test_stage_e_observed_candidate_recovery_review_escalation_failure_adds_evidence_collection():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "readiness_score": 0.5,
        "error_count": 1,
        "strict_production": False,
        "checks": {},
        "stage_b_promotion": {},
        "stage_d_readiness": {},
        "stage_e_readiness": {},
        "phase5_entry_readiness": {},
        "neuromorphic_profile_readiness": {},
        "iterative_repair_plan": {"next_actions": []},
        "repair_plan": {},
        "failure_focus": {},
        "repair_retry_queue": [],
        "execution_log": [
            {
                "command": "retry_stage_e_observed_acceptance_candidate_recovery_review_followup",
                "status": "failed",
                "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_followup_retry",
                "covered_checks": [
                    "stage_e_observed_acceptance_candidate_repair_recovery",
                    "stage_e_observed_acceptance_candidate_stability",
                    "research_journal_summary",
                ],
                "resolved_timestamp": 900.0,
            },
            {
                "command": "escalate_stage_e_observed_acceptance_candidate_recovery_review_followup_retry",
                "status": "failed",
                "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation",
                "covered_checks": [
                    "stage_e_observed_acceptance_candidate_repair_recovery",
                    "stage_e_observed_acceptance_candidate_stability",
                    "research_journal_summary",
                ],
                "resolved_timestamp": 950.0,
            },
        ],
        "research_journal_summary": {
            "entry_count": 1,
            "stage_e_observed_acceptance_candidate_repair_loop": {
                "id": "stage_e_observed_acceptance_candidate_repair",
                "needs_followup": False,
                "recovery_confirmed": True,
                "recovery_source": "remeasure",
                "latest_remeasure_trend": "recovered",
            },
        },
    }

    summary = module.format_operational_summary(report)
    actions = module.build_operational_runbook_actions(report)

    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_evidence_collection_in_progress: False"
    ) in summary
    evidence_actions = [
        item
        for item in actions
        if item["source"]
        == "stage_e_observed_acceptance_candidate_recovery_review_evidence_collection"
    ]
    escalation_actions = [
        item
        for item in actions
        if item["source"]
        == "stage_e_observed_acceptance_candidate_recovery_review_followup_retry_escalation"
    ]
    assert escalation_actions == []
    assert len(evidence_actions) == 1
    assert "collect_stage_e_observed_acceptance_candidate_recovery_review_evidence" in (
        evidence_actions[0]["command"]
    )
    assert "latest_escalation_status=failed" in evidence_actions[0]["reason"]


def test_stage_e_observed_candidate_recovery_review_evidence_completion_adds_recheck():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "readiness_score": 0.5,
        "error_count": 1,
        "strict_production": False,
        "checks": {},
        "stage_b_promotion": {},
        "stage_d_readiness": {},
        "stage_e_readiness": {},
        "phase5_entry_readiness": {},
        "neuromorphic_profile_readiness": {},
        "iterative_repair_plan": {"next_actions": []},
        "repair_plan": {},
        "failure_focus": {},
        "repair_retry_queue": [],
        "execution_log": [
            {
                "command": "collect_stage_e_observed_acceptance_candidate_recovery_review_evidence",
                "status": "success",
                "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_evidence_collection",
                "covered_checks": [
                    "stage_e_observed_acceptance_candidate_repair_recovery",
                    "stage_e_observed_acceptance_candidate_stability",
                    "research_journal_summary",
                ],
                "resolved_timestamp": 950.0,
            },
        ],
        "research_journal_summary": {
            "entry_count": 1,
            "stage_e_observed_acceptance_candidate_repair_loop": {
                "id": "stage_e_observed_acceptance_candidate_repair",
                "needs_followup": False,
                "recovery_confirmed": True,
                "recovery_source": "remeasure",
                "latest_remeasure_trend": "recovered",
            },
        },
    }

    summary = module.format_operational_summary(report)
    actions = module.build_operational_runbook_actions(report)

    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_evidence_collection_completed: True"
    ) in summary
    recheck_actions = [
        item
        for item in actions
        if item["source"] == "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck"
    ]
    assert len(recheck_actions) == 1
    assert "recheck_stage_e_observed_acceptance_candidate_recovery_review_evidence" in (
        recheck_actions[0]["command"]
    )
    assert "latest_evidence_collection_status=success" in recheck_actions[0]["reason"]


def test_stage_e_observed_candidate_recovery_review_evidence_recheck_pending_suppresses_duplicate():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "readiness_score": 0.5,
        "error_count": 1,
        "strict_production": False,
        "checks": {},
        "stage_b_promotion": {},
        "stage_d_readiness": {},
        "stage_e_readiness": {},
        "phase5_entry_readiness": {},
        "neuromorphic_profile_readiness": {},
        "iterative_repair_plan": {"next_actions": []},
        "repair_plan": {},
        "failure_focus": {},
        "repair_retry_queue": [],
        "execution_log": [
            {
                "command": "collect_stage_e_observed_acceptance_candidate_recovery_review_evidence",
                "status": "success",
                "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_evidence_collection",
                "covered_checks": [
                    "stage_e_observed_acceptance_candidate_repair_recovery",
                    "stage_e_observed_acceptance_candidate_stability",
                    "research_journal_summary",
                ],
                "resolved_timestamp": 950.0,
            },
            {
                "command": "recheck_stage_e_observed_acceptance_candidate_recovery_review_evidence",
                "status": "pending",
                "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck",
                "covered_checks": [
                    "stage_e_observed_acceptance_candidate_repair_recovery",
                    "stage_e_observed_acceptance_candidate_stability",
                    "research_journal_summary",
                ],
                "resolved_timestamp": 975.0,
            },
        ],
        "research_journal_summary": {
            "entry_count": 1,
            "stage_e_observed_acceptance_candidate_repair_loop": {
                "id": "stage_e_observed_acceptance_candidate_repair",
                "needs_followup": False,
                "recovery_confirmed": True,
                "recovery_source": "remeasure",
                "latest_remeasure_trend": "recovered",
            },
        },
    }

    summary = module.format_operational_summary(report)
    actions = module.build_operational_runbook_actions(report)

    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_evidence_recheck_in_progress: True"
    ) in summary
    assert not [
        item
        for item in actions
        if item["source"] == "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck"
    ]


def test_stage_e_observed_candidate_recovery_review_evidence_recheck_failure_adds_targeted_probe():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "readiness_score": 0.5,
        "error_count": 1,
        "strict_production": False,
        "checks": {},
        "stage_b_promotion": {},
        "stage_d_readiness": {},
        "stage_e_readiness": {},
        "phase5_entry_readiness": {},
        "neuromorphic_profile_readiness": {},
        "iterative_repair_plan": {"next_actions": []},
        "repair_plan": {},
        "failure_focus": {},
        "repair_retry_queue": [],
        "execution_log": [
            {
                "command": "collect_stage_e_observed_acceptance_candidate_recovery_review_evidence",
                "status": "success",
                "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_evidence_collection",
                "covered_checks": [
                    "stage_e_observed_acceptance_candidate_repair_recovery",
                    "stage_e_observed_acceptance_candidate_stability",
                    "research_journal_summary",
                ],
                "resolved_timestamp": 950.0,
            },
            {
                "command": "recheck_stage_e_observed_acceptance_candidate_recovery_review_evidence",
                "status": "failed",
                "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck",
                "covered_checks": [
                    "stage_e_observed_acceptance_candidate_repair_recovery",
                    "stage_e_observed_acceptance_candidate_stability",
                    "research_journal_summary",
                ],
                "resolved_timestamp": 975.0,
            },
        ],
        "research_journal_summary": {
            "entry_count": 1,
            "stage_e_observed_acceptance_candidate_repair_loop": {
                "id": "stage_e_observed_acceptance_candidate_repair",
                "needs_followup": False,
                "recovery_confirmed": True,
                "recovery_source": "remeasure",
                "latest_remeasure_trend": "recovered",
            },
        },
    }

    summary = module.format_operational_summary(report)
    actions = module.build_operational_runbook_actions(report)

    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_evidence_recheck_failed: True"
    ) in summary
    recheck_actions = [
        item
        for item in actions
        if item["source"] == "stage_e_observed_acceptance_candidate_recovery_review_evidence_recheck"
    ]
    probe_actions = [
        item
        for item in actions
        if item["source"] == "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe"
    ]
    assert recheck_actions == []
    assert len(probe_actions) == 1
    assert "probe_stage_e_observed_acceptance_candidate_recovery_review_evidence" in (
        probe_actions[0]["command"]
    )
    assert "latest_evidence_recheck_status=failed" in probe_actions[0]["reason"]


def test_stage_e_observed_candidate_recovery_review_targeted_probe_completion_adds_recheck():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "readiness_score": 0.5,
        "error_count": 1,
        "strict_production": False,
        "checks": {},
        "stage_b_promotion": {},
        "stage_d_readiness": {},
        "stage_e_readiness": {},
        "phase5_entry_readiness": {},
        "neuromorphic_profile_readiness": {},
        "iterative_repair_plan": {"next_actions": []},
        "repair_plan": {},
        "failure_focus": {},
        "repair_retry_queue": [],
        "execution_log": [
            {
                "command": "probe_stage_e_observed_acceptance_candidate_recovery_review_evidence",
                "status": "success",
                "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_targeted_probe",
                "covered_checks": [
                    "stage_e_observed_acceptance_candidate_repair_recovery",
                    "stage_e_observed_acceptance_candidate_stability",
                    "research_journal_summary",
                ],
                "resolved_timestamp": 975.0,
            },
        ],
        "research_journal_summary": {
            "entry_count": 1,
            "stage_e_observed_acceptance_candidate_repair_loop": {
                "id": "stage_e_observed_acceptance_candidate_repair",
                "needs_followup": False,
                "recovery_confirmed": True,
                "recovery_source": "remeasure",
                "latest_remeasure_trend": "recovered",
            },
        },
    }

    summary = module.format_operational_summary(report)
    actions = module.build_operational_runbook_actions(report)

    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_targeted_probe_completed: True"
    ) in summary
    recheck_actions = [
        item
        for item in actions
        if item["source"]
        == "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck"
    ]
    assert len(recheck_actions) == 1
    assert "recheck_stage_e_observed_acceptance_candidate_recovery_review_targeted_probe" in (
        recheck_actions[0]["command"]
    )
    assert "latest_targeted_probe_status=success" in recheck_actions[0]["reason"]


def test_stage_e_observed_candidate_recovery_review_targeted_probe_recheck_pending_suppresses_duplicate():
    module = _load_script("operational_readiness.py")
    report = {
        "passed": False,
        "readiness_score": 0.5,
        "error_count": 1,
        "strict_production": False,
        "checks": {},
        "stage_b_promotion": {},
        "stage_d_readiness": {},
        "stage_e_readiness": {},
        "phase5_entry_readiness": {},
        "neuromorphic_profile_readiness": {},
        "iterative_repair_plan": {"next_actions": []},
        "repair_plan": {},
        "failure_focus": {},
        "repair_retry_queue": [],
        "execution_log": [
            {
                "command": "probe_stage_e_observed_acceptance_candidate_recovery_review_evidence",
                "status": "success",
                "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_targeted_probe",
                "covered_checks": [
                    "stage_e_observed_acceptance_candidate_repair_recovery",
                    "stage_e_observed_acceptance_candidate_stability",
                    "research_journal_summary",
                ],
                "resolved_timestamp": 975.0,
            },
            {
                "command": "recheck_stage_e_observed_acceptance_candidate_recovery_review_targeted_probe",
                "status": "pending",
                "source": "runbook_action:stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck",
                "covered_checks": [
                    "stage_e_observed_acceptance_candidate_repair_recovery",
                    "stage_e_observed_acceptance_candidate_stability",
                    "research_journal_summary",
                ],
                "resolved_timestamp": 990.0,
            },
        ],
        "research_journal_summary": {
            "entry_count": 1,
            "stage_e_observed_acceptance_candidate_repair_loop": {
                "id": "stage_e_observed_acceptance_candidate_repair",
                "needs_followup": False,
                "recovery_confirmed": True,
                "recovery_source": "remeasure",
                "latest_remeasure_trend": "recovered",
            },
        },
    }

    summary = module.format_operational_summary(report)
    actions = module.build_operational_runbook_actions(report)

    assert (
        "- research_journal_stage_e_observed_acceptance_candidate_repair_promotion_review_targeted_probe_recheck_in_progress: True"
    ) in summary
    assert not [
        item
        for item in actions
        if item["source"]
        == "stage_e_observed_acceptance_candidate_recovery_review_targeted_probe_recheck"
    ]


def test_build_operational_runbook_actions_deduplicates_and_prioritizes():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {
            "next_actions": [
                {
                    "title": "rerun release gate",
                    "command": "python scripts/eval/release_gate.py --skip-accuracy",
                    "priority": "high",
                    "affected_checks": ["release_gate"],
                }
            ]
        },
        "repair_retry_queue": [
            {
                "command": "python scripts/eval/release_gate.py --skip-accuracy",
                "reason": "failed",
                "priority_tier": "high",
                "priority_score": 5.0,
                "covered_checks": ["release_gate"],
            },
            {
                "command": "python scripts/eval/release_soak.py --profile extended",
                "reason": "timeout",
                "priority_tier": "medium",
                "priority_score": 4.0,
                "covered_checks": ["release_gate"],
            },
        ],
        "repair_plan": {
            "fallback_actions": [
                {
                    "title": "fallback rerun",
                    "command": "python scripts/eval/phase3_accuracy_suite.py",
                    "affected_checks": ["phase3_accuracy"],
                }
            ]
        },
    }
    actions = module.build_operational_runbook_actions(report)
    assert len(actions) == 3
    assert actions[0]["source"] == "iterative_next_action"
    assert actions[0]["command"] == "python scripts/eval/release_gate.py --skip-accuracy"
    assert actions[1]["source"] == "retry_queue"
    assert actions[1]["command"] == "python scripts/eval/release_soak.py --profile extended"
    assert actions[2]["source"] == "fallback_action"
    assert actions[2]["command"] == "python scripts/eval/phase3_accuracy_suite.py"


def test_build_operational_runbook_actions_adds_ann_efficiency_next_evidence():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "ann_efficiency_roadmap": {
            "next_evidence_actions": [
                {
                    "category": "pending_joule_pair",
                    "priority": "high",
                    "task": "real_data_external_validity",
                    "command": (
                        "python scripts/sara_cli.py record-energy-measurement "
                        "--run-id <run-id> --system sara --task real_data_external_validity "
                        "--success-count <count> --joules <J>"
                    ),
                }
            ]
        },
    }

    actions = module.build_operational_runbook_actions(report)

    action = next(item for item in actions if item["source"] == "ann_efficiency_next_evidence")
    assert action["priority"] == "high"
    assert "record-energy-measurement" in action["command"]
    assert "energy_measurement" in action["affected_checks"]


def test_build_operational_runbook_actions_adds_phase8_reference_evidence():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "ann_efficiency_roadmap": {
            "next_evidence_actions": [
                {
                    "source": "external_reference_readiness",
                    "category": "missing_reference_directory",
                    "priority": "high",
                    "task": "phase8_reference_strength",
                    "command": (
                        "Provide a valid local directory for Local Cross-Encoder Reference "
                        "and rerun python scripts/sara_cli.py eval-external-validity."
                    ),
                }
            ]
        },
    }

    actions = module.build_operational_runbook_actions(report)

    action = next(item for item in actions if item["source"] == "ann_efficiency_next_evidence")
    assert action["priority"] == "high"
    assert "eval-external-validity" in action["command"]
    assert "ann_efficiency_roadmap" in action["affected_checks"]
    assert "external_validity" in action["affected_checks"]


def test_build_operational_runbook_actions_adds_internal_maintenance_evidence():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "ann_efficiency_roadmap": {
            "next_evidence_actions": [
                {
                    "source": "internal_maintenance_reference",
                    "category": "missing_internal_maintenance_reference",
                    "priority": "medium",
                    "task": "phase6_maintenance_efficiency",
                    "command": "python scripts/sara_cli.py eval-internal-maintenance-efficiency",
                }
            ]
        },
    }

    actions = module.build_operational_runbook_actions(report)

    action = next(item for item in actions if item["source"] == "ann_efficiency_next_evidence")
    assert action["priority"] == "medium"
    assert "eval-internal-maintenance-efficiency" in action["command"]
    assert "ann_efficiency_roadmap" in action["affected_checks"]
    assert "internal_maintenance_efficiency" in action["affected_checks"]


def test_build_operational_runbook_actions_adds_sara_ann_comparison_evidence():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "sara_ann_comparison": {
            "next_actions": [
                {
                    "category": "missing_event_memory_maintenance_coupling_surface",
                    "priority": "medium",
                    "command": "python scripts/sara_cli.py eval-event-memory-maintenance-coupling",
                }
            ]
        },
    }

    actions = module.build_operational_runbook_actions(report)

    action = next(item for item in actions if item["source"] == "sara_ann_comparison_next_action")
    assert action["priority"] == "medium"
    assert "eval-event-memory-maintenance-coupling" in action["command"]
    assert "sara_ann_comparison" in action["affected_checks"]
    assert "event_memory_ingest_pipeline" in action["affected_checks"]
    assert "event_memory_maintenance_coupling" in action["affected_checks"]


def test_build_operational_runbook_actions_adds_adaptive_credit_repair_actions():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "checks": {
            "adaptive_credit_field": {"passed": False},
            "adaptive_credit_event_memory": {"passed": False},
        },
    }

    actions = module.build_operational_runbook_actions(report)

    field_action = next(
        item
        for item in actions
        if item["source"] == "adaptive_credit_repair"
        and "adaptive_credit_field_benchmark.py" in item["command"]
    )
    memory_action = next(
        item
        for item in actions
        if item["source"] == "adaptive_credit_repair"
        and "adaptive_credit_event_memory_benchmark.py" in item["command"]
    )
    assert field_action["priority"] == "high"
    assert field_action["affected_checks"] == ["adaptive_credit_field"]
    assert memory_action["priority"] == "high"
    assert "adaptive_credit_event_memory" in memory_action["affected_checks"]
    assert "event_memory_ingest_pipeline" in memory_action["affected_checks"]


def test_build_operational_runbook_actions_marks_orphan_pair_as_energy_measurement_work():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "ann_efficiency_roadmap": {
            "next_evidence_actions": [
                {
                    "category": "orphan_pair",
                    "priority": "medium",
                    "task": "extra_task",
                    "command": (
                        "Inspect data/raw/energy_measurements.jsonl for rows that do not belong "
                        "to the active physical session batch."
                    ),
                }
            ]
        },
    }

    actions = module.build_operational_runbook_actions(report)

    action = next(item for item in actions if item["source"] == "ann_efficiency_next_evidence")
    assert action["priority"] == "medium"
    assert "ann_efficiency_roadmap" in action["affected_checks"]
    assert "energy_measurement" in action["affected_checks"]


def test_build_operational_runbook_actions_marks_invalid_pair_fairness_mismatch_as_energy_measurement_work():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "ann_efficiency_roadmap": {
            "next_evidence_actions": [
                {
                    "category": "invalid_pair_fairness_mismatch",
                    "priority": "high",
                    "task": "real_data_external_validity",
                    "command": "Repair mismatched pair conditions before rerunning this physical pair.",
                }
            ]
        },
    }

    actions = module.build_operational_runbook_actions(report)

    action = next(item for item in actions if item["source"] == "ann_efficiency_next_evidence")
    assert action["priority"] == "high"
    assert "ann_efficiency_roadmap" in action["affected_checks"]
    assert "energy_measurement" in action["affected_checks"]


def test_build_operational_runbook_actions_escalates_critical_weak_joule_pair():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "ann_efficiency_roadmap": {
            "next_evidence_actions": [
                {
                    "category": "weak_joule_pair",
                    "priority": "medium",
                    "severity": "critical",
                    "task": "real_data_external_validity",
                    "relative_ratio": 0.4,
                    "ratio_gap": 1.2,
                    "command": "Repeat this paired measurement with more replicates.",
                }
            ]
        },
    }

    actions = module.build_operational_runbook_actions(report)

    action = next(item for item in actions if item["source"] == "ann_efficiency_next_evidence")
    assert action["priority"] == "high"
    assert "severity=critical" in action["reason"]
    assert "ratio_gap=1.200" in action["reason"]
    assert "relative_ratio=0.400" in action["reason"]


def test_build_operational_runbook_actions_merges_v1_actions_without_duplicates():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {
            "next_actions": [
                {
                    "title": "rerun release gate",
                    "command": "python scripts/eval/release_gate.py --skip-accuracy",
                    "priority": "high",
                    "affected_checks": ["release_gate"],
                }
            ]
        },
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
    }
    v1_actions = [
        {
            "category": "stage_b",
            "priority": "high",
            "command": "python scripts/eval/future_state_consistency_benchmark.py",
            "expected_effect": "Re-measure Stage B.",
            "affected_checks": ["stage_b_reward_policy_minimum"],
        },
        {
            "category": "stage_b",
            "priority": "high",
            "command": "python scripts/eval/future_state_consistency_benchmark.py",
            "expected_effect": "Duplicate command.",
            "affected_checks": ["stage_b_rlm_observation_minimum"],
        },
    ]

    actions = module.build_operational_runbook_actions(report, external_actions=v1_actions)

    assert len(actions) == 2
    assert actions[0]["source"] == "iterative_next_action"
    assert actions[1]["source"] == "v1_recovery_action"
    assert actions[1]["command"] == "python scripts/eval/future_state_consistency_benchmark.py"


def test_build_operational_runbook_actions_adds_v1_hygiene_action_when_snapshot_rejects_entries():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "v1_actions_snapshot": {
            "loaded_count": 4,
            "accepted_count": 1,
            "rejected_stale_count": 2,
            "rejected_missing_timestamp_count": 1,
        },
    }

    actions = module.build_operational_runbook_actions(report, external_actions=[])

    assert len(actions) == 1
    assert actions[0]["source"] == "v1_action_hygiene"
    assert actions[0]["priority"] == "high"
    assert actions[0]["command"] == "python scripts/eval/v1_release_gate.py"
    assert "stale=2" in actions[0]["reason"]
    assert "missing_timestamp=1" in actions[0]["reason"]


def test_build_operational_runbook_actions_adds_roadmap_patch_review_action():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "research_review": {
            "compact": {
                "passed": False,
                "next_hypothesis_count": 1,
                "regression_watchlist_count": 0,
                "negative_result_count": 2,
                "requires_human_approval": True,
                "release_gate_blocking": False,
            }
        },
    }

    actions = module.build_operational_runbook_actions(report)

    assert len(actions) == 1
    assert actions[0]["source"] == "roadmap_patch_review"
    assert actions[0]["priority"] == "high"
    assert actions[0]["command"] == "python scripts/eval/research_automation_benchmark.py --append-journal"
    assert actions[0]["affected_checks"] == ["research_review", "roadmap_patch_suggestion"]


def test_build_operational_runbook_actions_adds_experiment_priority_plan_actions():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "research_review": {
            "compact": {
                "passed": True,
                "experiment_priority_plan": {
                    "action_count": 2,
                    "top_priority_source": "experiment_regression_remeasure",
                    "top_priority_category": "regressing",
                    "actions": [
                        {
                            "category": "regressing",
                            "source": "experiment_regression_remeasure",
                            "priority": "high",
                            "count": 2,
                            "ids": [
                                "stage_e_architecture_integration_metric_recovery",
                                "sara_policy_alignment_recovery",
                            ],
                            "command_label": "experiment_regression_remeasure",
                            "policy": "remeasure before promotion or roadmap refresh",
                        },
                        {
                            "category": "adoption_candidate",
                            "source": "experiment_adoption_candidate_review",
                            "priority": "medium",
                            "count": 1,
                            "ids": ["predictive_spike_entropy_reduction_observed"],
                            "command_label": "experiment_adoption_candidate_review",
                            "policy": "review for bounded promotion after stable evidence",
                        },
                    ],
                },
            }
        },
    }

    actions = module.build_operational_runbook_actions(report)

    assert [action["source"] for action in actions] == [
        "experiment_regression_remeasure",
        "experiment_adoption_candidate_review",
    ]
    assert actions[0]["priority"] == "high"
    assert "experiment_regression_remeasure:stage_e_architecture_integration_metric_recovery" in actions[0]["command"]
    assert "experiment_priority_plan" in actions[0]["affected_checks"]
    assert "stage_e_architecture_integration_metric_recovery" in actions[0]["affected_checks"]
    assert "category=regressing" in actions[0]["reason"]


def test_build_operational_runbook_actions_adds_experiment_promotion_target_review_actions():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "research_review": {
            "compact": {
                "passed": True,
                "experiment_promotion_target_plan": {
                    "candidate_count": 1,
                    "review_action_count": 1,
                    "review_actions": [
                        {
                            "id": "stage_e_architecture_integration_observed_metrics",
                            "source": "experiment_promotion_target_review",
                            "priority": "medium",
                            "target_stage": "stage_e",
                            "target_surface": "observed_acceptance_candidate",
                            "promotion_path": "stage_e_architecture_acceptance_review",
                            "policy": "review micro-turn and phase-block traces before minimum-gate expansion",
                        }
                    ],
                },
            }
        },
    }

    actions = module.build_operational_runbook_actions(report)

    assert actions[0]["source"] == "experiment_promotion_target_review"
    assert actions[0]["priority"] == "medium"
    assert "stage_e_architecture_integration_observed_metrics:stage_e" in actions[0]["command"]
    assert "target_stage=stage_e" in actions[0]["reason"]
    assert "observed_acceptance_candidate" in actions[0]["affected_checks"]
    assert "stage_e_architecture_acceptance_review" in actions[0]["affected_checks"]


def test_build_operational_runbook_actions_prioritizes_research_planner_task_cleanup():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "research_planner_task_cleanup_threshold": 2,
        "research_journal_summary": {"completed_research_planner_task_count": 1},
        "research_review": {
            "compact": {
                "passed": False,
                "next_hypothesis_count": 1,
                "regression_watchlist_count": 0,
                "negative_result_count": 2,
                "cause_boundary_documentation_count": 1,
                "targeted_fixture_repair_count": 1,
                "cause_boundary_documentation_ids": [
                    "predictive_spike_entropy_reduction_observed"
                ],
                "targeted_fixture_repair_ids": [
                    "phase_binding_coincidence_integrity_observed"
                ],
                "requires_human_approval": True,
                "release_gate_blocking": False,
            },
            "report": {"generated_at": 100.0},
        },
    }

    actions = module.build_operational_runbook_actions(report)

    assert len(actions) == 1
    assert actions[0]["source"] == "research_planner_task_cleanup"
    assert actions[0]["priority"] == "high"
    assert "research_planner_task_pending_count=2" in actions[0]["reason"]
    assert "roadmap_patch_suggestion" not in actions[0]["affected_checks"]
    assert "predictive_spike_entropy_reduction_observed" in actions[0]["affected_checks"]
    assert "phase_binding_coincidence_integrity_observed" in actions[0]["affected_checks"]


def test_build_operational_runbook_actions_detects_stalled_research_planner_cleanup():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "research_planner_task_cleanup_threshold": 2,
        "research_journal_summary": {
            "completed_research_planner_task_count": 0,
            "research_planner_task_cleanup_pending_count": 1,
        },
        "research_review": {
            "compact": {
                "passed": False,
                "next_hypothesis_count": 1,
                "regression_watchlist_count": 0,
                "negative_result_count": 2,
                "cause_boundary_documentation_count": 1,
                "targeted_fixture_repair_count": 1,
            },
            "report": {"generated_at": 100.0},
        },
    }

    actions = module.build_operational_runbook_actions(report)

    assert len(actions) == 1
    assert actions[0]["source"] == "research_planner_fixture_repair_followup"
    assert "research_planner_task_cleanup_pending_count=1" in actions[0]["reason"]
    assert "stalled_reason=fixture_implementation_wait" in actions[0]["reason"]


def test_summarize_research_planner_task_status_computes_cleanup_signal():
    module = _load_script("operational_readiness.py")
    status = module.summarize_research_planner_task_status(
        {
            "cause_boundary_documentation_count": 1,
            "targeted_fixture_repair_count": 2,
        },
        {"completed_research_planner_task_count": 1},
        cleanup_threshold=3,
    )

    assert status["pending_count"] == 3
    assert status["completed_count"] == 1
    assert status["completion_ratio"] == 0.25
    assert status["cleanup_needed"] is True

    stalled = module.summarize_research_planner_task_status(
        {
            "cause_boundary_documentation_count": 1,
            "targeted_fixture_repair_count": 1,
        },
        {
            "completed_research_planner_task_count": 0,
            "research_planner_task_cleanup_pending_count": 1,
        },
        cleanup_threshold=2,
    )
    assert stalled["cleanup_needed"] is False
    assert stalled["cleanup_stalled"] is True
    assert stalled["cleanup_stalled_reason"] == "fixture_implementation_wait"

    manual = module.summarize_research_planner_task_status(
        {
            "cause_boundary_documentation_count": 2,
            "targeted_fixture_repair_count": 0,
        },
        {
            "completed_research_planner_task_count": 0,
            "research_planner_task_cleanup_pending_count": 1,
            "research_planner_task_cleanup_entries": [
                {
                    "status": "pending",
                    "source": "manual:research_planner_task_cleanup",
                    "command": "manual review",
                }
            ],
        },
        cleanup_threshold=2,
    )
    assert manual["cleanup_stalled_reason"] == "manual_review_wait"
    assert manual["cleanup_stalled_action_source"] == "research_planner_manual_review_followup"

    documentation = module.summarize_research_planner_task_status(
        {
            "cause_boundary_documentation_count": 2,
            "targeted_fixture_repair_count": 0,
        },
        {
            "completed_research_planner_task_count": 0,
            "research_planner_task_cleanup_pending_count": 1,
            "research_planner_task_cleanup_entries": [
                {
                    "status": "pending",
                    "source": "research_planner_task_cleanup",
                    "command": "write docs",
                }
            ],
        },
        cleanup_threshold=2,
    )
    assert documentation["cleanup_stalled_reason"] == "documentation_not_reflected"
    assert documentation["cleanup_stalled_action_source"] == "research_planner_documentation_followup"


def test_summarize_roadmap_patch_refresh_policy_classifies_ratio_health():
    module = _load_script("operational_readiness.py")

    insufficient = module.summarize_roadmap_patch_refresh_policy(
        {"roadmap_patch_rejected_item_count": 1, "roadmap_patch_refreshed_item_count": 0}
    )
    balanced = module.summarize_roadmap_patch_refresh_policy(
        {
            "roadmap_patch_rejected_item_count": 4,
            "roadmap_patch_refreshed_item_count": 2,
            "roadmap_patch_refresh_to_rejection_ratio": 0.5,
        }
    )
    over_suppression = module.summarize_roadmap_patch_refresh_policy(
        {
            "roadmap_patch_rejected_item_count": 4,
            "roadmap_patch_refreshed_item_count": 0,
            "roadmap_patch_refresh_to_rejection_ratio": 0.0,
        }
    )
    over_resurfacing = module.summarize_roadmap_patch_refresh_policy(
        {
            "roadmap_patch_rejected_item_count": 4,
            "roadmap_patch_refreshed_item_count": 4,
            "roadmap_patch_refresh_to_rejection_ratio": 1.0,
        }
    )

    assert insufficient["status"] == "insufficient_history"
    assert balanced["status"] == "balanced"
    assert over_suppression["status"] == "over_suppression"
    assert over_suppression["action_source"] == "roadmap_patch_refresh_over_suppression_followup"
    assert over_resurfacing["status"] == "over_resurfacing"
    assert over_resurfacing["action_source"] == "roadmap_patch_refresh_over_resurfacing_followup"


def test_build_operational_runbook_actions_adds_roadmap_patch_refresh_policy_followup():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "research_journal_summary": {
            "roadmap_patch_rejected_item_count": 4,
            "roadmap_patch_refreshed_item_count": 0,
            "roadmap_patch_refresh_to_rejection_ratio": 0.0,
        },
    }

    actions = module.build_operational_runbook_actions(report)

    assert actions[0]["source"] == "roadmap_patch_refresh_over_suppression_followup"
    assert actions[0]["priority"] == "medium"
    assert "roadmap_patch_refresh_policy_status=over_suppression" in actions[0]["reason"]
    assert "ratio=0.000" in actions[0]["reason"]
    assert actions[0]["affected_checks"] == [
        "research_journal_summary",
        "roadmap_patch_refresh_policy",
    ]


def test_roadmap_patch_refresh_policy_followup_completion_suppresses_repeat_action():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "execution_log": [
            {
                "command": "roadmap_patch_refresh_policy_review",
                "status": "success",
                "source": "runbook_action:roadmap_patch_refresh_over_suppression_followup",
                "covered_checks": [
                    "roadmap_patch_refresh_policy",
                    "research_journal_summary",
                ],
                "resolved_timestamp": 1000.0,
            }
        ],
        "research_journal_summary": {
            "roadmap_patch_rejected_item_count": 4,
            "roadmap_patch_refreshed_item_count": 0,
            "roadmap_patch_refresh_to_rejection_ratio": 0.0,
        },
    }

    attached = module.attach_roadmap_patch_refresh_policy_followups_to_research_journal_summary(
        report["research_journal_summary"],
        report["execution_log"],
    )
    policy = module.summarize_roadmap_patch_refresh_policy(attached)
    actions = module.build_operational_runbook_actions(report)

    assert attached["roadmap_patch_refresh_policy_followup_success_count"] == 1
    assert attached["roadmap_patch_refresh_policy_followup_latest_status"] == "success"
    assert policy["status"] == "followup_completed"
    assert policy["needs_followup"] is False
    assert all(
        action["source"] != "roadmap_patch_refresh_over_suppression_followup"
        for action in actions
    )


def test_roadmap_patch_refresh_policy_failed_followups_switch_to_evidence_collection():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "execution_log": [
            {
                "command": "roadmap_patch_refresh_policy_review",
                "status": "failed",
                "source": "runbook_action:roadmap_patch_refresh_over_suppression_followup",
                "covered_checks": ["roadmap_patch_refresh_policy", "research_journal_summary"],
                "resolved_timestamp": 1000.0,
            },
            {
                "command": "roadmap_patch_refresh_policy_review",
                "status": "timeout",
                "source": "runbook_action:roadmap_patch_refresh_over_suppression_followup",
                "covered_checks": ["roadmap_patch_refresh_policy", "research_journal_summary"],
                "resolved_timestamp": 1100.0,
            },
        ],
        "research_journal_summary": {
            "roadmap_patch_rejected_item_count": 4,
            "roadmap_patch_refreshed_item_count": 0,
            "roadmap_patch_refresh_to_rejection_ratio": 0.0,
        },
    }

    attached = module.attach_roadmap_patch_refresh_policy_followups_to_research_journal_summary(
        report["research_journal_summary"],
        report["execution_log"],
    )
    policy = module.summarize_roadmap_patch_refresh_policy(attached)
    actions = module.build_operational_runbook_actions(report)

    assert attached["roadmap_patch_refresh_policy_followup_failed_count"] == 2
    assert policy["status"] == "followup_failed_evidence_collection_needed"
    assert policy["action_source"] == "roadmap_patch_refresh_evidence_collection_fallback"
    assert actions[0]["source"] == "roadmap_patch_refresh_evidence_collection_fallback"
    assert "roadmap_patch_refresh_evidence_collection" in actions[0]["command"]
    assert actions[0]["affected_checks"] == [
        "evidence_collection",
        "research_journal_summary",
        "roadmap_patch_refresh_policy",
    ]


def test_roadmap_patch_evidence_collection_success_is_tracked_separately():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "execution_log": [
            {
                "command": "roadmap_patch_refresh_policy_review",
                "status": "failed",
                "source": "runbook_action:roadmap_patch_refresh_over_suppression_followup",
                "covered_checks": ["roadmap_patch_refresh_policy", "research_journal_summary"],
                "resolved_timestamp": 1000.0,
            },
            {
                "command": "roadmap_patch_refresh_policy_review",
                "status": "timeout",
                "source": "runbook_action:roadmap_patch_refresh_over_suppression_followup",
                "covered_checks": ["roadmap_patch_refresh_policy", "research_journal_summary"],
                "resolved_timestamp": 1100.0,
            },
            {
                "command": "roadmap_patch_refresh_evidence_collection",
                "status": "success",
                "source": "runbook_action:roadmap_patch_refresh_evidence_collection_fallback",
                "covered_checks": [
                    "roadmap_patch_refresh_policy",
                    "evidence_collection",
                    "targeted_probe",
                    "research_journal_summary",
                ],
                "resolved_timestamp": 1200.0,
            },
        ],
        "research_journal_summary": {
            "roadmap_patch_rejected_item_count": 4,
            "roadmap_patch_refreshed_item_count": 0,
            "roadmap_patch_refresh_to_rejection_ratio": 0.0,
            "roadmap_patch_refreshed_items": [],
        },
    }

    attached = module.attach_roadmap_patch_refresh_policy_followups_to_research_journal_summary(
        report["research_journal_summary"],
        report["execution_log"],
    )
    policy = module.summarize_roadmap_patch_refresh_policy(attached)
    actions = module.build_operational_runbook_actions(report)

    assert attached["roadmap_patch_evidence_collection_success_count"] == 1
    assert attached["roadmap_patch_evidence_collection_latest_status"] == "success"
    assert attached["roadmap_patch_evidence_collection_latest_kind"] == "targeted_probe"
    assert attached["roadmap_patch_evidence_collection_next_required_kind"] == "real_data_fixture"
    assert attached["roadmap_patch_evidence_collection_kind_counts"] == {"targeted_probe": 1}
    assert attached["roadmap_patch_evidence_collection_entries"][0]["evidence_kind"] == "targeted_probe"
    assert attached["roadmap_patch_refreshed_items"] == []
    assert policy["status"] == "evidence_collection_completed"
    assert policy["needs_followup"] is False
    assert all(action["source"] != "roadmap_patch_refresh_evidence_collection_fallback" for action in actions)


def test_build_operational_runbook_actions_adds_drop_rate_recovery_action():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "operational_checklist": {"runbook_drop_rate_ok": False},
        "runbook_drop_rate_threshold": 0.8,
    }
    actions = module.build_operational_runbook_actions(report)
    assert len(actions) == 1
    assert actions[0]["source"] == "runbook_drop_rate_recovery"
    assert actions[0]["priority"] == "medium"
    assert "--runbook-max-actions 50" in actions[0]["command"]
    assert "--runbook-max-per-source 0" in actions[0]["command"]
    assert "--runbook-drop-rate-threshold 0.800" in actions[0]["command"]


def test_build_operational_runbook_actions_adds_efficiency_shortcut_recovery_action():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "operational_checklist": {
            "efficiency_shortcut_action_ok": False,
            "efficiency_shortcut_action_count": 7,
            "efficiency_shortcut_action_threshold": 3,
        },
    }
    actions = module.build_operational_runbook_actions(report)
    assert len(actions) == 1
    assert actions[0]["source"] == "efficiency_shortcut_recovery"
    assert actions[0]["priority"] == "medium"
    assert "efficiency_shortcut_action_count_exceeded:7>3" in actions[0]["reason"]
    assert actions[0]["affected_checks"] == ["efficiency_shortcut_action_ok"]


def test_build_operational_runbook_actions_adds_efficiency_shortcut_chronic_recovery_action():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "operational_checklist": {
            "efficiency_shortcut_overuse_rate_ok": False,
            "efficiency_shortcut_overuse_rate": 0.8,
            "efficiency_shortcut_overuse_rate_threshold": 0.5,
        },
    }
    actions = module.build_operational_runbook_actions(report)
    assert len(actions) == 1
    assert actions[0]["source"] == "efficiency_shortcut_chronic_recovery"
    assert actions[0]["priority"] == "high"
    assert "efficiency_shortcut_overuse_rate_exceeded:0.800>0.500" in actions[0]["reason"]
    assert actions[0]["affected_checks"] == ["efficiency_shortcut_overuse_rate_ok"]


def test_build_operational_runbook_actions_adds_efficiency_incident_shortcut_actions():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "error_details": [
            {
                "check": "release_gate",
                "category": "release_gate_efficiency_kpi",
                "error": "ANN-cost advantage proxy below threshold.",
            }
        ],
    }
    actions = module.build_operational_runbook_actions(report)
    commands = [str(item.get("command", "")) for item in actions if isinstance(item, dict)]
    assert "python scripts/eval/energy_efficiency_benchmark.py" in commands
    assert "python scripts/eval/phase3_accuracy_suite.py" in commands
    assert "python scripts/eval/release_gate.py" in commands
    assert all(
        str(item.get("source", "")) == "efficiency_incident_shortcut"
        for item in actions
        if isinstance(item, dict)
    )


def test_build_operational_runbook_actions_applies_source_cap():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {
            "next_actions": [
                {"title": "i1", "command": "python scripts/eval/release_gate.py --skip-accuracy", "priority": "high"},
                {"title": "i2", "command": "python scripts/eval/release_gate.py", "priority": "high"},
            ]
        },
        "repair_retry_queue": [
            {"command": "python scripts/eval/release_soak.py --profile extended", "reason": "failed", "priority_tier": "high", "priority_score": 9.0},
            {"command": "python scripts/eval/phase3_accuracy_suite.py", "reason": "failed", "priority_tier": "medium", "priority_score": 5.0},
        ],
        "repair_plan": {
            "fallback_actions": [
                {"title": "f1", "command": "python scripts/eval/phase4_scale_continual_benchmark.py"},
                {"title": "f2", "command": "python scripts/eval/phase5_entry_gate.py"},
            ]
        },
    }

    actions = module.build_operational_runbook_actions(report, max_per_source=1)

    assert len(actions) == 3
    assert [item["source"] for item in actions] == [
        "iterative_next_action",
        "retry_queue",
        "fallback_action",
    ]


def test_build_operational_runbook_actions_returns_build_stats():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {
            "next_actions": [
                {"title": "a0", "command": "", "priority": "high"},
                {"title": "a1", "command": "python scripts/eval/release_gate.py", "priority": "high"},
                {"title": "a2", "command": "python scripts/eval/release_gate.py", "priority": "high"},
                {"title": "a3", "command": "python scripts/eval/release_soak.py --profile extended", "priority": "high"},
            ]
        },
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
    }
    actions, stats = module.build_operational_runbook_actions(
        report,
        max_actions=2,
        max_per_source=1,
        return_metadata=True,
    )
    assert len(actions) == 1
    assert stats["considered_count"] == 4
    assert stats["appended_count"] == 1
    assert stats["skipped_duplicate_count"] == 1
    assert stats["skipped_duplicate_by_source"]["iterative_next_action"] == 1
    assert stats["skipped_empty_command_count"] == 1
    assert stats["skipped_empty_command_by_source"]["iterative_next_action"] == 1
    assert stats["skipped_source_cap_count"] == 1
    assert stats["skipped_source_cap_by_source"]["iterative_next_action"] == 1
    assert stats["skipped_max_actions_count"] == 0


def test_build_operational_runbook_actions_applies_remeasure_command_history_quota():
    module = _load_script("operational_readiness.py")
    command = "python scripts/eval/cognitive_runtime_benchmark.py"
    report = {
        "iterative_repair_plan": {"next_actions": []},
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
        "execution_log": [
            {
                "command": command,
                "status": "success",
                "source": "runbook_action:research_journal_remeasure",
                "covered_checks": ["research_journal_summary", "predictive_spike_entropy_reduction_observed"],
            },
            {
                "command": command,
                "status": "failed",
                "source": "runbook_action:research_journal_remeasure",
                "covered_checks": ["research_journal_summary", "phase_binding_coincidence_integrity_observed"],
            },
        ],
        "research_journal_summary": {
            "recommended_benchmark_actions": [
                {
                    "id": "predictive_spike_entropy_reduction_observed",
                    "source": "negative_result",
                    "command": command,
                    "priority": "high",
                    "count": 4,
                },
                {
                    "id": "release_gate_safety_review",
                    "source": "regression_watchlist",
                    "command": "python scripts/eval/release_gate.py",
                    "priority": "high",
                    "count": 1,
                },
            ]
        },
    }

    actions, stats = module.build_operational_runbook_actions(report, return_metadata=True)

    assert [item["source"] for item in actions] == [
        "research_journal_alternative_probe",
        "research_journal_remeasure",
    ]
    assert actions[0]["command"].endswith(
        "tests/test_phase3_accuracy_benchmarks.py::test_cognitive_runtime_benchmark_returns_expected_metrics"
    )
    assert actions[1]["command"] == "python scripts/eval/release_gate.py"
    assert stats["skipped_remeasure_command_history_quota_count"] == 1
    assert stats["skipped_remeasure_command_history_quota_by_command"][command] == 1
    assert stats["remeasure_history_command_counts"][command] == 2
    assert stats["skipped_remeasure_command_history_quota_items"][0]["id"] == (
        "predictive_spike_entropy_reduction_observed"
    )
    assert stats["skipped_remeasure_command_history_quota_items"][0]["history_count"] == 2
    assert stats["skipped_remeasure_command_history_quota_items"][0][
        "alternative_command"
    ].endswith("tests/test_phase3_accuracy_benchmarks.py::test_cognitive_runtime_benchmark_returns_expected_metrics")


def test_attach_remeasure_quota_holds_to_research_journal_summary():
    module = _load_script("operational_readiness.py")
    command = "python scripts/eval/cognitive_runtime_benchmark.py"
    summary = module.attach_remeasure_quota_holds_to_research_journal_summary(
        {"entry_count": 2},
        {
            "skipped_remeasure_command_history_quota_items": [
                {
                    "id": "predictive_spike_entropy_reduction_observed",
                    "source": "negative_result",
                    "command": command,
                    "priority": "high",
                    "history_count": 3,
                    "quota": 2,
                    "hold_reason": "remeasure_command_history_quota",
                    "alternative_command": "PYTHONPATH=src workspace/.venv310/bin/python -m pytest -q tests/test_phase3_accuracy_benchmarks.py::test_cognitive_runtime_benchmark_returns_expected_metrics",
                    "alternative_reason": "target predictive fixture.",
                },
                {
                    "id": "stage_e_observed_acceptance_candidate_repair",
                    "source": "negative_result",
                    "command": command,
                    "priority": "high",
                    "history_count": 3,
                    "quota": 2,
                    "hold_reason": "remeasure_command_history_quota",
                    "alternative_command": "PYTHONPATH=src workspace/.venv310/bin/python -m pytest -q tests/test_phase3_accuracy_benchmarks.py::test_stage_e_observed_acceptance_candidate_failures_are_structured",
                    "alternative_reason": "target Stage E observed candidate fixture.",
                }
            ]
        },
    )

    assert summary["entry_count"] == 2
    assert summary["remeasure_quota_hold_count"] == 2
    assert summary["remeasure_quota_holds"][0]["id"] == "predictive_spike_entropy_reduction_observed"
    assert summary["remeasure_quota_holds"][0]["command"] == command
    assert summary["remeasure_quota_holds"][0]["history_count"] == 3
    assert summary["alternative_benchmark_actions"][0]["id"] == "predictive_spike_entropy_reduction_observed"
    assert summary["alternative_benchmark_actions"][0]["command"].endswith(
        "tests/test_phase3_accuracy_benchmarks.py::test_cognitive_runtime_benchmark_returns_expected_metrics"
    )
    repair_loop = summary["stage_e_observed_acceptance_candidate_repair_loop"]
    assert repair_loop["alternative_probe_recommended"] is True
    assert repair_loop["needs_followup"] is True


def test_attach_research_planner_task_completions_to_research_journal_summary():
    module = _load_script("operational_readiness.py")
    summary = module.attach_research_planner_task_completions_to_research_journal_summary(
        {"entry_count": 1},
        [
            {
                "command": "doc-boundary",
                "status": "success",
                "source": "manual:cause_boundary_documentation",
                "covered_checks": [
                    "cause_boundary_documentation",
                    "predictive_spike_entropy_reduction_observed",
                ],
                "resolved_timestamp": 500.0,
            },
            {
                "command": "fix-fixture",
                "status": "success",
                "source": "manual:targeted_fixture_repair",
                "covered_checks": [
                    "targeted_fixture_repair",
                    "phase_binding_coincidence_integrity_observed",
                ],
                "resolved_timestamp": 510.0,
            },
            {
                "command": "research_planner_task_cleanup",
                "status": "pending",
                "source": "research_planner_task_cleanup",
                "covered_checks": ["research_planner_task_cleanup"],
                "timestamp": 520.0,
            },
        ],
    )

    assert summary["entry_count"] == 1
    assert summary["completed_research_planner_task_count"] == 2
    assert summary["completed_cause_boundary_documentation_ids"] == [
        "predictive_spike_entropy_reduction_observed"
    ]
    assert summary["completed_targeted_fixture_repair_ids"] == [
        "phase_binding_coincidence_integrity_observed"
    ]
    assert summary["research_planner_task_cleanup_pending_count"] == 1
    assert summary["research_planner_task_cleanup_success_count"] == 0
    assert summary["research_planner_task_cleanup_entries"][0]["status"] == "pending"
    assert summary["completed_research_planner_tasks"][0]["task_type"] == (
        "cause_boundary_documentation"
    )


def test_build_operational_runbook_actions_honors_max_actions():
    module = _load_script("operational_readiness.py")
    report = {
        "iterative_repair_plan": {
            "next_actions": [
                {"title": "a1", "command": "python scripts/eval/release_gate.py", "priority": "high"},
                {"title": "a2", "command": "python scripts/eval/phase3_accuracy_suite.py", "priority": "high"},
                {"title": "a3", "command": "python scripts/eval/phase4_scale_continual_benchmark.py", "priority": "high"},
            ]
        },
        "repair_retry_queue": [],
        "repair_plan": {"fallback_actions": []},
    }
    actions, stats = module.build_operational_runbook_actions(
        report,
        max_actions=2,
        return_metadata=True,
    )
    assert len(actions) == 2
    assert stats["appended_count"] == 2
    assert stats["skipped_max_actions_count"] == 1
    assert stats["skipped_max_actions_by_source"]["iterative_next_action"] == 1


def test_summarize_runbook_actions_counts_source_and_priority():
    module = _load_script("operational_readiness.py")
    actions = [
        {"source": "retry_queue", "priority": "high", "command": "a"},
        {"source": "retry_queue", "priority": "medium", "command": "b"},
        {"source": "fallback_action", "priority": "medium", "command": "c"},
    ]
    summary = module.summarize_runbook_actions(actions)
    assert summary["total_actions"] == 3
    assert summary["source_counts"]["retry_queue"] == 2
    assert summary["source_counts"]["fallback_action"] == 1
    assert summary["priority_counts"]["medium"] == 2
    assert summary["priority_counts"]["high"] == 1


def test_format_operational_summary_includes_runbook_action_distribution():
    module = _load_script("operational_readiness.py")
    summary = module.format_operational_summary(
        {
            "passed": True,
            "error_count": 0,
            "readiness_score": 1.0,
            "strict_production": True,
            "checks": {
                "phase3_accuracy": {"passed": True},
                "phase3_completion": {"passed": True},
                "phase4_completion": {"passed": True},
                "phase5_entry_gate": {"passed": True},
                "release_gate": {"passed": True},
                "production_profile": {"passed": True},
            },
            "stage_b_promotion": {},
            "stage_d_readiness": {},
            "stage_e_readiness": {},
            "phase5_entry_readiness": {},
            "iterative_repair_plan": {"completed": True, "stalled": False, "stop_reason": "", "next_step_hint": "", "next_actions": []},
            "repair_plan": {"covered_checks": [], "uncovered_checks": []},
            "failure_focus": {},
            "repair_retry_queue_count": 0,
            "repair_retry_cooldown_seconds": 0.0,
            "repair_retry_cooldown_blocked_count": 0,
            "repair_pending_count": 0,
            "repair_timeout_count": 0,
            "runbook_max_actions": 25,
            "runbook_max_per_source": 1,
            "runbook_action_summary": {
                "total_actions": 2,
                "source_counts": {"retry_queue": 1, "fallback_action": 1},
                "priority_counts": {"high": 1, "medium": 1},
            },
            "runbook_action_build_stats": {
                "considered_count": 5,
                "skipped_duplicate_count": 1,
                "skipped_duplicate_by_source": {"iterative_next_action": 1},
                "skipped_empty_command_count": 1,
                "skipped_empty_command_by_source": {"retry_queue": 1},
                "skipped_source_cap_count": 2,
                "skipped_source_cap_by_source": {"retry_queue": 2},
                "skipped_max_actions_count": 1,
                "skipped_max_actions_by_source": {"fallback_action": 1},
            },
            "runbook_action_build_rates": {
                "drop_rate": 1.0,
                "duplicate_drop_rate": 0.2,
                "empty_drop_rate": 0.2,
                "source_cap_drop_rate": 0.4,
                "max_actions_drop_rate": 0.2,
            },
            "repair_auto_dispatch": {},
            "error_details": [],
            "error_details_summary": {},
            "operational_checklist": {"passed": True, "managed_output_paths_ok": True, "report_summary_review_ready": True},
            "recovery_actions": [],
        }
    )
    assert "- runbook_max_per_source: 1" in summary
    assert "- runbook_max_actions: 25" in summary
    assert "- runbook_action_total: 2" in summary
    assert "- efficiency_incident_shortcut_action_count: 0" in summary
    assert "- runbook_action_source_count: retry_queue=1" in summary
    assert "- runbook_action_considered_count: 5" in summary
    assert "- runbook_action_skipped_empty_command_count: 1" in summary
    assert "- runbook_action_skipped_empty_command_by_source: retry_queue=1" in summary
    assert "- runbook_action_skipped_duplicate_by_source: iterative_next_action=1" in summary
    assert "- runbook_action_skipped_source_cap_count: 2" in summary
    assert "- runbook_action_skipped_source_cap_by_source: retry_queue=2" in summary
    assert "- runbook_action_skipped_max_actions_by_source: fallback_action=1" in summary
    assert "- runbook_action_drop_rate: 1.000" in summary
    assert "- efficiency_shortcut_overuse_event_count: 0" in summary


def test_summarize_runbook_action_build_stats_computes_drop_rates():
    module = _load_script("operational_readiness.py")
    rates = module.summarize_runbook_action_build_stats(
        {
            "considered_count": 10,
            "skipped_duplicate_count": 2,
            "skipped_empty_command_count": 1,
            "skipped_source_cap_count": 1,
            "skipped_max_actions_count": 1,
            "skipped_remeasure_command_history_quota_count": 1,
        }
    )
    assert rates["drop_rate"] == 0.6
    assert rates["duplicate_drop_rate"] == 0.2
    assert rates["empty_drop_rate"] == 0.1
    assert rates["remeasure_command_history_quota_drop_rate"] == 0.1


def test_append_efficiency_shortcut_overuse_timeline_accumulates_history():
    module = _load_script("operational_readiness.py")
    output = {
        "generated_at": 200.0,
        "operational_checklist": {
            "efficiency_shortcut_action_ok": False,
            "efficiency_shortcut_action_count": 6,
            "efficiency_shortcut_action_threshold": 3,
        },
    }
    previous = {
        "efficiency_shortcut_overuse_timeline": [
            {
                "timestamp": 100.0,
                "overuse_active": True,
                "shortcut_action_count": 5,
                "shortcut_action_threshold": 3,
            }
        ]
    }
    module._append_efficiency_shortcut_overuse_timeline(output, previous_report=previous, max_entries=8)
    timeline = output["efficiency_shortcut_overuse_timeline"]
    assert len(timeline) == 2
    assert timeline[-1]["timestamp"] == 200.0
    assert timeline[-1]["overuse_active"] is True
    assert output["efficiency_shortcut_overuse_event_count"] == 2


def test_load_recent_v1_actions_filters_stale_and_missing_timestamp(tmp_path):
    module = _load_script("operational_readiness.py")
    path = tmp_path / "v1_release_gate_actions.json"
    path.write_text(
        """
[
  {"command":"python scripts/eval/release_gate.py","priority":"high","generated_at":90.0},
  {"command":"python scripts/eval/phase3_accuracy_suite.py","priority":"medium","generated_at":10.0},
  {"command":"python scripts/eval/phase4_scale_continual_benchmark.py","priority":"low"}
]
""".strip(),
        encoding="utf-8",
    )

    actions, snapshot = module._load_recent_v1_actions(
        str(path),
        max_age_seconds=30.0,
        now_timestamp=100.0,
    )

    assert len(actions) == 1
    assert actions[0]["command"] == "python scripts/eval/release_gate.py"
    assert snapshot["loaded_count"] == 3
    assert snapshot["accepted_count"] == 1
    assert snapshot["rejected_stale_count"] == 1
    assert snapshot["rejected_missing_timestamp_count"] == 1


def test_build_refresh_commands_updates_phase5_entry_artifacts():
    module = _load_script("operational_readiness.py")

    commands = module._build_refresh_commands(
        "extended",
        include_accuracy=True,
        phase3_regression_tolerance=0.05,
    )
    command_texts = [" ".join(command) for command in commands]
    phase3_suite_index = next(index for index, command in enumerate(command_texts) if "phase3_accuracy_suite.py" in command)
    phase5_benchmark_index = next(
        index for index, command in enumerate(command_texts) if "phase5_predictive_coding_benchmark.py" in command
    )
    phase5_gate_index = next(index for index, command in enumerate(command_texts) if "phase5_entry_gate.py" in command)
    sparse_diffusion_index = next(
        index for index, command in enumerate(command_texts) if "sparse_diffusion_block_readiness.py" in command
    )
    adaptive_credit_field_index = next(
        index for index, command in enumerate(command_texts) if "adaptive_credit_field_benchmark.py" in command
    )
    adaptive_credit_event_memory_index = next(
        index
        for index, command in enumerate(command_texts)
        if "adaptive_credit_event_memory_benchmark.py" in command
    )
    phase5_completion_index = next(
        index for index, command in enumerate(command_texts) if "phase5_completion_gate.py" in command
    )
    external_validity_index = next(
        index for index, command in enumerate(command_texts) if "real_data_external_validity.py" in command
    )
    external_validity_ladder_index = next(
        index for index, command in enumerate(command_texts) if "real_data_external_validity_ladder.py" in command
    )
    release_soak_index = next(index for index, command in enumerate(command_texts) if "release_soak.py" in command)

    assert any("scripts/eval/phase5_predictive_coding_benchmark.py" in command for command in command_texts)
    assert any("scripts/eval/phase5_entry_gate.py" in command for command in command_texts)
    assert any("scripts/eval/sparse_diffusion_block_readiness.py" in command for command in command_texts)
    assert any("scripts/eval/adaptive_credit_field_benchmark.py" in command for command in command_texts)
    assert any("scripts/eval/adaptive_credit_event_memory_benchmark.py" in command for command in command_texts)
    assert any("scripts/eval/phase5_completion_gate.py" in command for command in command_texts)
    assert any("scripts/eval/real_data_external_validity.py" in command for command in command_texts)
    assert any("scripts/eval/real_data_external_validity_ladder.py" in command for command in command_texts)
    assert "--regression-tolerance 0.050000" in command_texts[phase3_suite_index]
    assert phase5_benchmark_index < phase5_gate_index
    assert phase5_gate_index < sparse_diffusion_index
    assert sparse_diffusion_index < adaptive_credit_field_index
    assert adaptive_credit_field_index < adaptive_credit_event_memory_index
    assert adaptive_credit_event_memory_index < phase5_completion_index
    assert phase5_gate_index < phase5_completion_index
    assert phase5_completion_index < external_validity_index
    assert external_validity_index < external_validity_ladder_index
    assert external_validity_ladder_index < release_soak_index
