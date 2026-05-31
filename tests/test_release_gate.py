import importlib.util
import json
import os
import tempfile


def _load_release_gate_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "release_gate.py")
    )
    spec = importlib.util.spec_from_file_location("release_gate_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase5_completion_gate_report(passed=True):
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
    checks = {
        "phase5_entry_gate_passed": {"passed": bool(passed)},
        "multi_step_trace_complete": {"passed": bool(passed)},
        "counterfactual_branch_separable": {"passed": bool(passed)},
        "macro_step_reduction": {"passed": bool(passed)},
        "macro_cost_reduction": {"passed": bool(passed)},
        "subgoal_coverage_ratio": {"passed": bool(passed)},
        "micro_es_low_rank_trace_complete": {"passed": bool(passed)},
        "micro_es_fitness_improvement": {"passed": bool(passed)},
        "micro_es_event_cost_reduction": {"passed": bool(passed)},
        "micro_es_population_event_budget": {"passed": bool(passed)},
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
    for name in entry_metrics:
        checks[f"metric.{name}"] = {"passed": bool(passed)}
        checks[f"threshold.{name}"] = {"passed": bool(passed)}
    return {
        "suite_name": "Phase5CompletionGate",
        "passed": bool(passed),
        "phase5_overall_score": 1.0 if passed else 0.0,
        "failed_checks": [] if passed else ["micro_es_fitness_improvement"],
        "errors": [] if passed else ["Phase 5 micro-ES fitness improvement is below completion threshold."],
        "checks": checks,
    }


def _healthy_phase3_accuracy_report(module):
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
    focus_summary = {
        "few_shot": {"passed": True, "score": 1.0},
        "continual": {"passed": True, "score": 1.0},
        "retrieval_hygiene": {"passed": True, "score": 1.0},
        "adaptive_readiness": {"passed": True, "score": 1.0},
        "predictive_readiness": {"passed": True, "score": 1.0},
        "efficiency_readiness": {"passed": True, "score": 1.0},
        "consolidation_readiness": {"passed": True, "score": 1.0},
        "cognitive_runtime_readiness": {"passed": True, "score": 1.0},
        "phase5_entry_readiness": {
            "passed": True,
            "score": 1.0,
            "metrics": {
                f"phase5_predictive_coding.{metric_name}": 1.0
                for metric_name in module.PHASE5_ENTRY_METRIC_NAMES
            },
        },
    }
    return {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 0.95,
        "passed": True,
        "component_reports": {
            component_name: {"passed": True, "overall_score": 1.0}
            for component_name in required_components
        },
        "focus_summary": focus_summary,
        "trend": {
            "has_previous": True,
            "regression_count": 0,
            "gate_regression_count": 0,
        },
        "stage_a_acceptance": {"passed": True},
        "stage_b_readiness": {
            "passed": True,
            "minimum_requirements_passed": True,
            "minimum_checks": {
                check_name: True for check_name in module.STAGE_B_REQUIRED_MINIMUM_CHECKS
            },
        },
        "stage_c_readiness": {
            "passed": True,
            "minimum_requirements_passed": True,
            "minimum_checks": {
                check_name: True for check_name in module.STAGE_C_REQUIRED_MINIMUM_CHECKS
            },
        },
        "stage_d_readiness": {
            "passed": True,
            "minimum_requirements_passed": True,
            "minimum_checks": {
                check_name: True for check_name in module.STAGE_D_REQUIRED_MINIMUM_CHECKS
            },
        },
        "stage_e_readiness": {
            "passed": True,
            "minimum_requirements_passed": True,
            "minimum_checks": {
                check_name: True for check_name in module.STAGE_E_REQUIRED_MINIMUM_CHECKS
            },
        },
    }


def test_release_gate_accepts_healthy_report():
    module = _load_release_gate_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "min_duration_seconds": 5.0,
            "min_agent_turns": 8,
            "min_inference_iterations": 12,
            "min_pattern_count": 1,
        },
        "agent": {"history_bounded": True, "issue_count": 0, "turns": 8},
        "inference": {
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 10,
            "iterations": 12,
        },
    }

    assert module.validate_release_report(report) == []

    accuracy_report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 0.95,
        "passed": True,
        "component_reports": {
            "agent_dialogue": {"passed": True, "overall_score": 0.9},
            "sara_inference": {"passed": True, "overall_score": 1.0},
            "spiking_llm": {"passed": True, "overall_score": 0.95},
            "task_switch_adaptation": {"passed": True, "overall_score": 1.0},
            "future_state_consistency": {"passed": True, "overall_score": 1.0},
            "energy_efficiency": {"passed": True, "overall_score": 0.95},
            "continual_consolidation": {"passed": True, "overall_score": 1.0},
            "cognitive_runtime": {"passed": True, "overall_score": 1.0},
            "phase5_predictive_coding": {"passed": True, "overall_score": 1.0},
        },
        "focus_summary": {
            "few_shot": {"passed": True, "score": 1.0},
            "continual": {"passed": True, "score": 1.0},
            "retrieval_hygiene": {"passed": True, "score": 0.8},
            "adaptive_readiness": {"passed": True, "score": 1.0},
            "predictive_readiness": {"passed": True, "score": 1.0},
            "efficiency_readiness": {"passed": True, "score": 0.95},
            "consolidation_readiness": {"passed": True, "score": 1.0},
            "cognitive_runtime_readiness": {"passed": True, "score": 1.0},
            "phase5_entry_readiness": {
                "passed": True,
                "score": 1.0,
                "metrics": {
                    "phase5_predictive_coding.latent_transition_alignment": 1.0,
                    "phase5_predictive_coding.prediction_error_observability": 1.0,
                    "phase5_predictive_coding.correction_event_coverage": 1.0,
                    "phase5_predictive_coding.anti_collapse_event_diversity": 1.0,
                    "phase5_predictive_coding.counterfactual_transition_separation": 1.0,
                    "phase5_predictive_coding.multi_step_latent_chain_integrity": 1.0,
                    "phase5_predictive_coding.long_horizon_error_correction_convergence": 1.0,
                    "phase5_predictive_coding.horizon_bucket_stability": 1.0,
                    "phase5_predictive_coding.macro_action_effectiveness": 1.0,
                    "phase5_predictive_coding.subgoal_decomposition_integrity": 1.0,
                    "phase5_predictive_coding.depth_selective_routing_integrity": 1.0,
                    "phase5_predictive_coding.micro_es_policy_refinement_integrity": 1.0,
                },
            },
        },
        "trend": {
            "has_previous": True,
            "regression_count": 0,
        },
        "stage_a_acceptance": {
            "passed": True,
        },
        "stage_b_readiness": {
            "passed": True,
            "minimum_requirements_passed": True,
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
        },
        "stage_c_readiness": {
            "passed": True,
            "minimum_requirements_passed": True,
            "minimum_checks": {
                "metric.meta_adaptation_loop": True,
                "metric.meta_adaptation_parameter_integrity": True,
                "metric.temporal_self_distillation_stability": True,
            },
        },
        "stage_d_readiness": {
            "passed": True,
            "minimum_requirements_passed": True,
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
    }

    assert module.validate_phase3_accuracy_report(accuracy_report) == []


def test_release_gate_allows_observed_only_phase3_trend_regressions():
    module = _load_release_gate_module()
    accuracy_report = _healthy_phase3_accuracy_report(module)
    accuracy_report["trend"] = {
        "has_previous": True,
        "regression_count": 2,
        "gate_regression_count": 0,
        "regressions": [
            {"metric": "parameter_efficiency.average_quality_per_kparam"},
            {"metric": "parameter_efficiency.average_quality_per_mb"},
        ],
        "gate_regressions": [],
    }

    assert module.validate_phase3_accuracy_report(accuracy_report) == []


def test_release_gate_rejects_unhealthy_report():
    module = _load_release_gate_module()
    report = {
        "duration_seconds": 1.0,
        "criteria": {
            "min_duration_seconds": 5.0,
            "min_agent_turns": 8,
            "min_inference_iterations": 12,
            "min_pattern_count": 2,
        },
        "agent": {"history_bounded": False, "issue_count": 2, "turns": 4},
        "inference": {
            "roundtrip_ok": False,
            "tuple_keys_only": False,
            "pattern_count": 0,
            "iterations": 3,
        },
    }

    errors = module.validate_release_report(report)

    assert errors
    assert any("history" in item.lower() for item in errors)
    assert any("round-trip" in item.lower() for item in errors)
    assert any("duration" in item.lower() for item in errors)
    assert any("turn count" in item.lower() for item in errors)
    assert any("iteration count" in item.lower() for item in errors)


def test_release_gate_rejects_unhealthy_phase3_accuracy_report():
    module = _load_release_gate_module()
    accuracy_report = {
        "suite_name": "WrongSuite",
        "overall_score": 0.0,
        "passed": False,
        "component_reports": {
            "agent_dialogue": {"passed": False},
            "sara_inference": {"passed": True},
        },
        "focus_summary": {
            "few_shot": {"passed": False, "score": 0.0},
        },
        "trend": {
            "has_previous": True,
            "regression_count": 2,
        },
        "stage_b_readiness": {
            "passed": False,
            "minimum_requirements_passed": False,
            "minimum_checks": {
                "metric.future_state_transition_integrity": False,
                "metric.future_state_transition_operator_coverage": False,
            },
        },
    }

    errors = module.validate_phase3_accuracy_report(accuracy_report)

    assert errors
    assert any("unexpected suite name" in item.lower() for item in errors)
    assert any("did not pass" in item.lower() for item in errors)
    assert any("overall score" in item.lower() for item in errors)
    assert any("missing required components" in item.lower() for item in errors)
    assert any("focus summaries" in item.lower() for item in errors)
    assert any("regression" in item.lower() for item in errors)
    assert any("stage b" in item.lower() for item in errors)


def test_release_gate_reports_efficiency_kpi_threshold_failures():
    module = _load_release_gate_module()
    accuracy_report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 0.95,
        "passed": True,
        "component_reports": {
            "agent_dialogue": {"passed": True},
            "sara_inference": {"passed": True},
            "spiking_llm": {"passed": True},
            "task_switch_adaptation": {"passed": True},
            "future_state_consistency": {"passed": True},
            "energy_efficiency": {"passed": True},
            "continual_consolidation": {"passed": True},
            "cognitive_runtime": {"passed": True},
            "phase5_predictive_coding": {"passed": True},
        },
        "focus_summary": {
            "few_shot": {"passed": True},
            "continual": {"passed": True},
            "retrieval_hygiene": {"passed": True},
            "adaptive_readiness": {"passed": True},
            "predictive_readiness": {"passed": True},
            "efficiency_readiness": {
                "passed": False,
                "score": 0.6,
                "metrics": {
                    "energy_efficiency.performance_energy_ratio_proxy": 0.15,
                    "energy_efficiency.ann_cost_advantage_proxy": 6.5,
                    "energy_efficiency.brain_efficiency_alignment_proxy": 0.70,
                },
            },
            "consolidation_readiness": {"passed": True},
            "cognitive_runtime_readiness": {"passed": True},
            "phase5_entry_readiness": {
                "passed": True,
                "metrics": {
                    "phase5_predictive_coding.latent_transition_alignment": 1.0,
                    "phase5_predictive_coding.prediction_error_observability": 1.0,
                    "phase5_predictive_coding.correction_event_coverage": 1.0,
                    "phase5_predictive_coding.anti_collapse_event_diversity": 1.0,
                    "phase5_predictive_coding.counterfactual_transition_separation": 1.0,
                    "phase5_predictive_coding.multi_step_latent_chain_integrity": 1.0,
                    "phase5_predictive_coding.long_horizon_error_correction_convergence": 1.0,
                    "phase5_predictive_coding.horizon_bucket_stability": 1.0,
                    "phase5_predictive_coding.macro_action_effectiveness": 1.0,
                    "phase5_predictive_coding.subgoal_decomposition_integrity": 1.0,
                    "phase5_predictive_coding.depth_selective_routing_integrity": 1.0,
                    "phase5_predictive_coding.micro_es_policy_refinement_integrity": 1.0,
                },
            },
        },
        "stage_a_acceptance": {"passed": True},
        "stage_b_readiness": {"passed": True, "minimum_requirements_passed": True, "minimum_checks": {}},
        "stage_c_readiness": {"passed": True, "minimum_requirements_passed": True, "minimum_checks": {}},
        "stage_d_readiness": {"passed": True, "minimum_requirements_passed": True, "minimum_checks": {}},
        "stage_e_readiness": {"passed": True, "minimum_requirements_passed": True, "minimum_checks": {}},
    }

    errors = module.validate_phase3_accuracy_report(accuracy_report)

    assert any("efficiency_readiness did not satisfy performance-per-energy ratio proxy" in item for item in errors)
    assert any("efficiency_readiness did not satisfy ANN-reference cost advantage proxy" in item for item in errors)
    assert any("efficiency_readiness did not satisfy brain-efficiency alignment proxy" in item for item in errors)


def test_release_gate_accepts_embedded_accuracy_results_in_release_report():
    module = _load_release_gate_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "min_duration_seconds": 5.0,
            "min_agent_turns": 8,
            "min_inference_iterations": 12,
            "min_pattern_count": 1,
            "require_phase3_accuracy": True,
        },
        "agent": {"history_bounded": True, "issue_count": 0, "turns": 8},
        "inference": {
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 10,
            "iterations": 12,
        },
        "accuracy": {
            "suite_name": "Phase3AccuracySuite",
            "overall_score": 0.95,
            "passed": True,
            "component_reports": {
                "agent_dialogue": {"passed": True, "overall_score": 0.9},
                "sara_inference": {"passed": True, "overall_score": 1.0},
                "spiking_llm": {"passed": True, "overall_score": 1.0},
                "task_switch_adaptation": {"passed": True, "overall_score": 1.0},
                "future_state_consistency": {"passed": True, "overall_score": 1.0},
                "energy_efficiency": {"passed": True, "overall_score": 0.95},
                "continual_consolidation": {"passed": True, "overall_score": 1.0},
                "cognitive_runtime": {"passed": True, "overall_score": 1.0},
                "phase5_predictive_coding": {"passed": True, "overall_score": 1.0},
            },
            "focus_summary": {
                "few_shot": {"passed": True, "score": 1.0},
                "continual": {"passed": True, "score": 1.0},
                "retrieval_hygiene": {"passed": True, "score": 0.8},
                "adaptive_readiness": {"passed": True, "score": 1.0},
                "predictive_readiness": {"passed": True, "score": 1.0},
                "efficiency_readiness": {"passed": True, "score": 0.95},
                "consolidation_readiness": {"passed": True, "score": 1.0},
                "cognitive_runtime_readiness": {"passed": True, "score": 1.0},
                "phase5_entry_readiness": {
                    "passed": True,
                    "score": 1.0,
                    "metrics": {
                        "phase5_predictive_coding.latent_transition_alignment": 1.0,
                        "phase5_predictive_coding.prediction_error_observability": 1.0,
                        "phase5_predictive_coding.correction_event_coverage": 1.0,
                        "phase5_predictive_coding.anti_collapse_event_diversity": 1.0,
                        "phase5_predictive_coding.counterfactual_transition_separation": 1.0,
                        "phase5_predictive_coding.multi_step_latent_chain_integrity": 1.0,
                        "phase5_predictive_coding.long_horizon_error_correction_convergence": 1.0,
                        "phase5_predictive_coding.horizon_bucket_stability": 1.0,
                        "phase5_predictive_coding.macro_action_effectiveness": 1.0,
                        "phase5_predictive_coding.subgoal_decomposition_integrity": 1.0,
                        "phase5_predictive_coding.depth_selective_routing_integrity": 1.0,
                        "phase5_predictive_coding.micro_es_policy_refinement_integrity": 1.0,
                    },
                },
            },
            "trend": {
                "has_previous": True,
                "regression_count": 0,
            },
            "stage_a_acceptance": {
                "passed": True,
            },
            "stage_b_readiness": {
                "passed": True,
                "minimum_requirements_passed": True,
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
            },
            "stage_c_readiness": {
                "passed": True,
                "minimum_requirements_passed": True,
                "minimum_checks": {
                    "metric.meta_adaptation_loop": True,
                    "metric.meta_adaptation_parameter_integrity": True,
                    "metric.temporal_self_distillation_stability": True,
                },
            },
            "stage_d_readiness": {
                "passed": True,
                "minimum_requirements_passed": True,
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
        },
        "release_metadata": {
            "versions_match": True,
            "has_expected_console_scripts": True,
            "release_notes_heading": "Current Pre-Release",
        },
        "release_checklist": {
            "passed": True,
            "managed_output_paths_ok": True,
            "release_notes_reviewed": True,
            "report_summary_review_ready": True,
            "extended_profile_ready": False,
        },
    }

    assert module.validate_release_report(report) == []


def test_release_gate_rejects_phase3_report_below_stage_a_acc_target():
    module = _load_release_gate_module()
    accuracy_report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 0.94,
        "passed": True,
        "component_reports": {
            "agent_dialogue": {"passed": True},
            "sara_inference": {"passed": True},
            "spiking_llm": {"passed": True},
            "task_switch_adaptation": {"passed": True},
            "future_state_consistency": {"passed": True},
            "energy_efficiency": {"passed": True},
            "continual_consolidation": {"passed": True},
        },
        "focus_summary": {
            "few_shot": {"passed": True},
            "continual": {"passed": True},
            "retrieval_hygiene": {"passed": True},
            "adaptive_readiness": {"passed": True},
            "predictive_readiness": {"passed": True},
            "efficiency_readiness": {"passed": True},
            "consolidation_readiness": {"passed": True},
        },
        "trend": {
            "has_previous": True,
            "regression_count": 0,
        },
        "stage_a_acceptance": {
            "passed": False,
        },
        "stage_b_readiness": {
            "passed": True,
            "minimum_requirements_passed": True,
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
        },
        "stage_d_readiness": {
            "passed": True,
            "minimum_requirements_passed": True,
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
    }

    errors = module.validate_phase3_accuracy_report(accuracy_report)

    assert any("acc target" in item.lower() for item in errors)
    assert any("stage a acceptance" in item.lower() for item in errors)


def test_release_gate_rejects_stage_b_world_model_minimum_failures():
    module = _load_release_gate_module()
    accuracy_report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 0.95,
        "passed": True,
        "component_reports": {
            "agent_dialogue": {"passed": True},
            "sara_inference": {"passed": True},
            "spiking_llm": {"passed": True},
            "task_switch_adaptation": {"passed": True},
            "future_state_consistency": {"passed": True},
            "energy_efficiency": {"passed": True},
            "continual_consolidation": {"passed": True},
        },
        "focus_summary": {
            "few_shot": {"passed": True},
            "continual": {"passed": True},
            "retrieval_hygiene": {"passed": True},
            "adaptive_readiness": {"passed": True},
            "predictive_readiness": {"passed": True},
            "efficiency_readiness": {"passed": True},
            "consolidation_readiness": {"passed": True},
        },
        "trend": {
            "has_previous": True,
            "regression_count": 0,
        },
        "stage_a_acceptance": {
            "passed": True,
        },
        "stage_b_readiness": {
            "passed": False,
            "minimum_requirements_passed": False,
            "minimum_checks": {
                "metric.future_state_transition_integrity": True,
                "metric.future_state_command_integrity": False,
                "metric.future_state_predictor_snapshot_integrity": True,
                "metric.future_state_runtime_tracking_integrity": True,
                "metric.future_state_shift_tracking_integrity": False,
                "metric.future_state_transition_operator_coverage": True,
                "metric.future_state_transition_operator_consistency": False,
                "metric.future_state_counterfactual_branch_viability": False,
                "metric.future_state_fluid_trace_integrity": False,
                "metric.future_state_fluid_support_integrity": True,
                "metric.future_state_refinement_loop_integrity": False,
                "metric.future_state_adaptive_refinement": False,
                "metric.future_state_rewarded_action_selection_integrity": False,
                "metric.future_state_policy_update_stability": False,
                "metric.future_state_energy_aware_action_preference": False,
                "metric.future_state_focused_retrieval_hit_ratio": False,
                "metric.future_state_branch_level_decision_consistency": False,
            },
        },
        "stage_d_readiness": {
            "passed": True,
            "minimum_requirements_passed": True,
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
    }

    errors = module.validate_phase3_accuracy_report(accuracy_report)

    assert any("stage b readiness criteria" in item.lower() for item in errors)
    assert any("predicted next-step commands" in item.lower() for item in errors)
    assert any("shift tracking" in item.lower() for item in errors)
    assert any("fluid trace integrity" in item.lower() for item in errors)
    assert any("adaptive refinement integrity" in item.lower() for item in errors)
    assert any("value=" in item.lower() and "required>=" in item.lower() for item in errors)
    assert any("minimum requirements" in item.lower() for item in errors)


def test_release_gate_reports_stage_d_metric_name_in_minimum_failure_error():
    module = _load_release_gate_module()
    accuracy_report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 1.0,
        "passed": True,
        "component_reports": {
            "agent_dialogue": {"passed": True},
            "sara_inference": {"passed": True},
            "spiking_llm": {"passed": True},
            "task_switch_adaptation": {"passed": True},
            "future_state_consistency": {"passed": True},
            "energy_efficiency": {"passed": True},
            "continual_consolidation": {"passed": True},
        },
        "focus_summary": {
            "few_shot": {"passed": True},
            "continual": {"passed": True},
            "retrieval_hygiene": {"passed": True},
            "adaptive_readiness": {"passed": True},
            "predictive_readiness": {"passed": True},
            "efficiency_readiness": {"passed": True},
            "consolidation_readiness": {"passed": True},
        },
        "trend": {"has_previous": True, "regression_count": 0},
        "stage_a_acceptance": {"passed": True},
        "stage_b_readiness": {
            "passed": True,
            "minimum_requirements_passed": True,
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
            },
        },
        "stage_c_readiness": {
            "passed": True,
            "minimum_requirements_passed": True,
            "minimum_checks": {
                "metric.meta_adaptation_loop": True,
                "metric.meta_adaptation_parameter_integrity": True,
                "metric.temporal_self_distillation_stability": True,
            },
        },
        "stage_d_readiness": {
            "passed": False,
            "minimum_requirements_passed": False,
            "metrics": {
                "replay_recovery_integrity": 1.0,
                "long_horizon_consolidation_retention": 1.0,
                "counterfactual_replay_selection_integrity": 1.0,
                "replay_upgrade_reindex_integrity": 1.0,
                "memory_health_index_integrity": 1.0,
                "astro_modulation_stability": 0.0,
            },
            "minimum_checks": {
                "metric.replay_recovery_integrity": True,
                "metric.long_horizon_consolidation_retention": True,
                "metric.counterfactual_replay_selection_integrity": True,
                "metric.replay_upgrade_reindex_integrity": True,
                "metric.memory_health_index_integrity": True,
                "metric.replay_noise_resilience_integrity": False,
                "metric.astro_modulation_stability": False,
            },
        },
    }

    errors = module.validate_phase3_accuracy_report(accuracy_report)

    assert any(
        "metric=astro_modulation_stability" in item
        and "required>=1.000" in item
        for item in errors
    )


def test_release_gate_rejects_stage_c_meta_adaptation_minimum_failures():
    module = _load_release_gate_module()
    accuracy_report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 0.95,
        "passed": True,
        "component_reports": {
            "agent_dialogue": {"passed": True},
            "sara_inference": {"passed": True},
            "spiking_llm": {"passed": True},
            "task_switch_adaptation": {"passed": True},
            "future_state_consistency": {"passed": True},
            "energy_efficiency": {"passed": True},
            "continual_consolidation": {"passed": True},
        },
        "focus_summary": {
            "few_shot": {"passed": True},
            "continual": {"passed": True},
            "retrieval_hygiene": {"passed": True},
            "adaptive_readiness": {"passed": True},
            "predictive_readiness": {"passed": True},
            "efficiency_readiness": {"passed": True},
            "consolidation_readiness": {"passed": True},
        },
        "trend": {
            "has_previous": True,
            "regression_count": 0,
        },
        "stage_a_acceptance": {
            "passed": True,
        },
        "stage_b_readiness": {
            "passed": True,
            "minimum_requirements_passed": True,
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
            },
        },
        "stage_c_readiness": {
            "passed": False,
            "minimum_requirements_passed": False,
            "minimum_checks": {
                "metric.meta_adaptation_loop": True,
                "metric.meta_adaptation_parameter_integrity": False,
                "metric.temporal_self_distillation_stability": False,
            },
            "metrics": {
                "meta_adaptation_loop": 1.0,
                "meta_adaptation_parameter_integrity": 0.5,
                "temporal_self_distillation_stability": 0.5,
            },
        },
        "stage_d_readiness": {
            "passed": True,
            "minimum_requirements_passed": True,
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
    }

    errors = module.validate_phase3_accuracy_report(accuracy_report)

    assert any("stage c readiness criteria" in item.lower() for item in errors)
    assert any("meta-adaptation parameter integrity" in item.lower() for item in errors)
    assert any("value=" in item.lower() and "required>=" in item.lower() for item in errors)
    assert any("stage c readiness reports unmet meta-adaptation minimum requirements" in item.lower() for item in errors)


def test_release_gate_rejects_missing_embedded_accuracy_when_required():
    module = _load_release_gate_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "min_duration_seconds": 5.0,
            "min_agent_turns": 8,
            "min_inference_iterations": 12,
            "min_pattern_count": 1,
            "require_phase3_accuracy": True,
        },
        "agent": {"history_bounded": True, "issue_count": 0, "turns": 8},
        "inference": {
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 10,
            "iterations": 12,
        },
    }

    errors = module.validate_release_report(report)

    assert errors
    assert any("embedded phase 3 accuracy" in item.lower() for item in errors)


def test_release_gate_skip_embedded_accuracy_bypasses_required_accuracy_check():
    module = _load_release_gate_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "min_duration_seconds": 5.0,
            "min_agent_turns": 8,
            "min_inference_iterations": 12,
            "min_pattern_count": 1,
            "require_phase3_accuracy": True,
        },
        "agent": {"history_bounded": True, "issue_count": 0, "turns": 8},
        "inference": {
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 10,
            "iterations": 12,
        },
    }

    errors = module.validate_release_report(report, skip_embedded_accuracy=True)

    assert not any("embedded phase 3 accuracy" in item.lower() for item in errors)


def test_release_gate_rejects_invalid_embedded_release_metadata():
    module = _load_release_gate_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "min_duration_seconds": 5.0,
            "min_agent_turns": 8,
            "min_inference_iterations": 12,
            "min_pattern_count": 1,
            "require_phase3_accuracy": False,
        },
        "agent": {"history_bounded": True, "issue_count": 0, "turns": 8},
        "inference": {
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 10,
            "iterations": 12,
        },
        "release_metadata": {
            "versions_match": False,
            "has_expected_console_scripts": False,
            "release_notes_heading": "",
        },
    }

    errors = module.validate_release_report(report)

    assert errors
    assert any("mismatched package versions" in item.lower() for item in errors)
    assert any("missing console scripts" in item.lower() for item in errors)
    assert any("release notes heading" in item.lower() for item in errors)


def test_release_gate_rejects_failed_embedded_release_checklist():
    module = _load_release_gate_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "min_duration_seconds": 5.0,
            "min_agent_turns": 8,
            "min_inference_iterations": 12,
            "min_pattern_count": 1,
            "require_phase3_accuracy": False,
        },
        "agent": {"history_bounded": True, "issue_count": 0, "turns": 8},
        "inference": {
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 10,
            "iterations": 12,
        },
        "release_metadata": {
            "versions_match": True,
            "has_expected_console_scripts": True,
            "release_notes_heading": "Current Pre-Release",
        },
        "release_checklist": {
            "passed": False,
            "managed_output_paths_ok": False,
            "release_notes_reviewed": False,
            "report_summary_review_ready": False,
            "extended_profile_ready": False,
        },
    }

    errors = module.validate_release_report(report)

    assert any("release checklist" in item.lower() for item in errors)
    assert any("unmanaged output paths" in item.lower() for item in errors)


def test_release_gate_suggests_recovery_actions_for_gate_failures():
    module = _load_release_gate_module()
    errors = [
        "Soak duration is below the minimum required window (5.0 seconds).",
        "Phase 3 Stage B readiness did not satisfy the world-model prototype minimum for adaptive refinement integrity (metric.future_state_adaptive_refinement, value=0.500, required>=1.000).",
    ]

    actions = module.suggest_release_gate_recovery_actions(errors)

    assert actions
    assert all(isinstance(action.get("priority", ""), str) and action.get("priority", "") for action in actions)
    assert all(
        isinstance(action.get("expected_effect", ""), str) and action.get("expected_effect", "")
        for action in actions
    )
    assert all(isinstance(action.get("affected_checks", []), list) for action in actions)
    assert any("stage_b.minimum_checks" in action.get("affected_checks", []) for action in actions)
    assert any("soak.duration_seconds" in action.get("affected_checks", []) for action in actions)
    assert actions[0]["priority"] == "high"
    assert any("phase3_accuracy_suite.py" in action.get("command", "") for action in actions)
    assert any("--profile extended --include-accuracy" in action.get("command", "") for action in actions)


def test_release_gate_recovery_actions_are_empty_when_no_errors():
    module = _load_release_gate_module()

    actions = module.suggest_release_gate_recovery_actions([])

    assert actions == []


def test_release_gate_suggests_stage_c_adaptation_recovery_action():
    module = _load_release_gate_module()
    errors = [
        "Phase 3 focus summary 'adaptive_readiness' did not pass.",
        "task_switch_adaptation.meta_adaptation_parameter_integrity dropped below threshold.",
    ]

    actions = module.suggest_release_gate_recovery_actions(errors)

    assert any(
        "task_switch_adaptation_benchmark.py" in action.get("command", "")
        for action in actions
        if isinstance(action, dict)
    )
    assert any(
        "stage_c.meta_adaptation_parameter_integrity" in action.get("affected_checks", [])
        for action in actions
        if isinstance(action, dict)
    )
    assert any(
        "stage_c.adaptive_readiness" in action.get("affected_checks", [])
        for action in actions
        if isinstance(action, dict)
    )
    assert any(
        "stage_c.temporal_self_distillation_stability" in action.get("affected_checks", [])
        for action in actions
        if isinstance(action, dict)
    )


def test_release_gate_infers_stage_c_failed_checks_from_errors():
    module = _load_release_gate_module()
    errors = [
        "Phase 3 focus summary 'adaptive_readiness' did not pass.",
        "task_switch_adaptation.meta_adaptation_parameter_integrity dropped below threshold.",
        "task_switch_adaptation.temporal_self_distillation_stability dropped below threshold.",
    ]

    inferred = module._infer_failed_checks_from_errors(errors)

    assert "stage_c.adaptive_readiness" in inferred
    assert "stage_c.meta_adaptation_parameter_integrity" in inferred
    assert "stage_c.temporal_self_distillation_stability" in inferred


def test_release_gate_suggests_efficiency_recovery_action():
    module = _load_release_gate_module()
    errors = [
        "Phase 3 efficiency_readiness did not satisfy performance-per-energy ratio proxy "
        "(energy_efficiency.performance_energy_ratio_proxy, value=0.150, required>=0.200).",
    ]

    actions = module.suggest_release_gate_recovery_actions(errors)

    assert any(
        "energy_efficiency_benchmark.py" in action.get("command", "")
        for action in actions
        if isinstance(action, dict)
    )
    assert any(
        "focus.efficiency_readiness.passed" in action.get("affected_checks", [])
        for action in actions
        if isinstance(action, dict)
    )


def test_release_gate_infers_efficiency_failed_checks_from_errors():
    module = _load_release_gate_module()
    errors = [
        "Phase 3 efficiency_readiness did not satisfy ANN-reference cost advantage proxy "
        "(energy_efficiency.ann_cost_advantage_proxy, value=6.500, required>=8.000).",
        "Phase 3 efficiency_readiness did not satisfy brain-efficiency alignment proxy "
        "(energy_efficiency.brain_efficiency_alignment_proxy, value=0.700, required>=0.850).",
    ]

    inferred = module._infer_failed_checks_from_errors(errors)

    assert "focus.efficiency_readiness.passed" in inferred


def test_release_gate_suggests_stage_d_consolidation_recovery_action():
    module = _load_release_gate_module()
    errors = [
        "Phase 3 focus summary 'consolidation_readiness' did not pass.",
        "continual_consolidation.replay_recovery_integrity dropped below threshold.",
    ]

    actions = module.suggest_release_gate_recovery_actions(errors)

    assert any(
        "continual_consolidation_benchmark.py" in action.get("command", "")
        for action in actions
        if isinstance(action, dict)
    )
    assert any(
        "stage_d.replay_recovery_integrity" in action.get("affected_checks", [])
        for action in actions
        if isinstance(action, dict)
    )
    assert any(
        "stage_d.consolidation_readiness" in action.get("affected_checks", [])
        for action in actions
        if isinstance(action, dict)
    )
    assert any(
        "stage_d.astro_modulation_stability" in action.get("affected_checks", [])
        for action in actions
        if isinstance(action, dict)
    )
    assert any(
        "stage_d.replay_noise_resilience_integrity" in action.get("affected_checks", [])
        for action in actions
        if isinstance(action, dict)
    )


def test_release_gate_infers_stage_d_failed_checks_from_errors():
    module = _load_release_gate_module()
    errors = [
        "Phase 3 focus summary 'consolidation_readiness' did not pass.",
        "continual_consolidation.replay_recovery_integrity dropped below threshold.",
        "continual_consolidation.long_horizon_consolidation_retention dropped below threshold.",
        "continual_consolidation.replay_noise_resilience_integrity dropped below threshold.",
        "continual_consolidation.astro_modulation_stability dropped below threshold.",
    ]

    inferred = module._infer_failed_checks_from_errors(errors)

    assert "stage_d.consolidation_readiness" in inferred
    assert "stage_d.replay_recovery_integrity" in inferred
    assert "stage_d.long_horizon_consolidation_retention" in inferred
    assert "stage_d.replay_noise_resilience_integrity" in inferred
    assert "stage_d.astro_modulation_stability" in inferred


def test_release_gate_builds_minimal_repair_plan():
    module = _load_release_gate_module()
    errors = [
        "Soak duration is below the minimum required window (5.0 seconds).",
        "Release soak report requires embedded Phase 3 accuracy results.",
        "Phase 3 Stage B readiness did not satisfy the world-model prototype minimum for adaptive refinement integrity (metric.future_state_adaptive_refinement, value=0.500, required>=1.000).",
    ]

    plan = module.build_release_gate_repair_plan(errors)

    assert plan["estimated_steps"] >= 1
    assert isinstance(plan["selected_actions"], list)
    assert plan["selected_actions"]
    assert 0.0 <= float(plan["coverage_ratio"]) <= 1.0
    assert isinstance(plan["covered_checks"], list)
    assert isinstance(plan["uncovered_checks"], list)
    assert any(
        "stage_b.minimum_checks" in action.get("affected_checks", [])
        for action in plan["selected_actions"]
        if isinstance(action, dict)
    )
    assert any(
        "release_gate.embedded_accuracy_present" in action.get("affected_checks", [])
        for action in plan["selected_actions"]
        if isinstance(action, dict)
    )
    assert isinstance(plan.get("fallback_actions", []), list)


def test_release_gate_builds_fallback_plan_for_uncovered_checks():
    module = _load_release_gate_module()
    errors = [
        "Unknown subsystem failure for shadow pipeline.",
    ]

    plan = module.build_release_gate_repair_plan(errors)

    assert isinstance(plan.get("fallback_actions", []), list)
    assert plan["fallback_actions"]
    assert any(
        "release_gate.unknown_error" in action.get("affected_checks", [])
        for action in plan["fallback_actions"]
        if isinstance(action, dict)
    )


def test_release_gate_iterative_repair_plan_updates_remaining_checks():
    module = _load_release_gate_module()
    errors = [
        "Soak duration is below the minimum required window (5.0 seconds).",
        "Phase 3 Stage B readiness did not satisfy the world-model prototype minimum for adaptive refinement integrity (metric.future_state_adaptive_refinement, value=0.500, required>=1.000).",
    ]
    execution_log = [
        {
            "command": "python scripts/eval/release_soak.py --profile extended --include-accuracy",
            "status": "success",
            "covered_checks": ["soak.duration_seconds"],
        }
    ]

    iterative = module.build_iterative_release_gate_repair_plan(errors, execution_log=execution_log)

    assert iterative["executed_steps"] == 1
    assert iterative["successful_steps"] == 1
    assert isinstance(iterative["remaining_checks"], list)
    assert "soak.duration_seconds" not in iterative["remaining_checks"]
    assert isinstance(iterative["next_actions"], list)
    assert iterative["next_actions"]
    assert iterative["completed"] is False
    assert iterative["auto_stopped"] is False
    assert iterative["stop_reason"] == "pending_actions"
    assert isinstance(iterative["next_step_hint"], str)


def test_release_gate_iterative_repair_plan_auto_stops_when_completed():
    module = _load_release_gate_module()
    errors = [
        "Soak duration is below the minimum required window (5.0 seconds).",
    ]
    execution_log = [
        {
            "command": "python scripts/eval/release_soak.py --profile extended --include-accuracy",
            "status": "success",
            "covered_checks": ["soak.duration_seconds"],
        }
    ]

    iterative = module.build_iterative_release_gate_repair_plan(errors, execution_log=execution_log)

    assert iterative["completed"] is True
    assert iterative["auto_stopped"] is True
    assert iterative["stop_reason"] == "auto_stopped_completed"
    assert iterative["next_actions"] == []
    assert "No further action required" in iterative["next_step_hint"]


def test_release_gate_loads_repair_execution_log_from_json_array():
    module = _load_release_gate_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "repair_log.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                [
                    {"command": "python scripts/eval/release_gate.py", "status": "success"},
                    {"command": "python scripts/eval/release_soak.py", "status": "failed"},
                ],
                handle,
            )

        rows = module.load_repair_execution_log(path)

    assert isinstance(rows, list)
    assert len(rows) == 2
    assert rows[0]["status"] == "success"


def test_release_gate_loads_repair_execution_log_from_jsonl():
    module = _load_release_gate_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "repair_log.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('{"command":"python scripts/eval/release_gate.py","status":"success"}\n')
            handle.write('{"command":"python scripts/eval/release_soak.py","status":"failed"}\n')

        rows = module.load_repair_execution_log(path)

    assert isinstance(rows, list)
    assert len(rows) == 2
    assert rows[1]["status"] == "failed"


def test_release_gate_collects_artifacts_bundle():
    module = _load_release_gate_module()
    errors = [
        "Soak duration is below the minimum required window (5.0 seconds).",
    ]
    artifacts = module.collect_release_gate_artifacts(errors, execution_log=[])

    assert isinstance(artifacts, dict)
    assert "recovery_actions" in artifacts
    assert "repair_plan" in artifacts
    assert "iterative_repair_plan" in artifacts
    assert "error_details" in artifacts
    assert "error_details_summary" in artifacts
    assert "failure_focus" in artifacts


def test_release_gate_builds_structured_error_details():
    module = _load_release_gate_module()
    errors = [
        "Phase 3 Stage D readiness did not satisfy the continual-consolidation minimum for astro modulation stability (metric.astro_modulation_stability, metric=astro_modulation_stability, value=0.000, required>=1.000).",
        "continual_consolidation.replay_noise_resilience_integrity dropped below threshold.",
        "Release soak report requires embedded Phase 3 accuracy results.",
    ]

    details = module.build_release_gate_error_details(errors)

    assert isinstance(details, list)
    assert len(details) == 3
    minimum_detail = details[0]
    assert minimum_detail["type"] == "minimum_threshold_failure"
    assert minimum_detail["stage"] == "stage_d"
    assert minimum_detail["check_name"] == "metric.astro_modulation_stability"
    assert minimum_detail["metric_name"] == "astro_modulation_stability"
    assert minimum_detail["actual_value"] == 0.0
    assert minimum_detail["required_value"] == 1.0
    threshold_detail = details[1]
    assert threshold_detail["type"] == "metric_threshold_drop"
    assert threshold_detail["metric_name"] == "continual_consolidation.replay_noise_resilience_integrity"
    general_detail = details[2]
    assert general_detail["type"] == "general_error"


def test_release_gate_builds_stage_e_structured_error_details():
    module = _load_release_gate_module()
    errors = [
        "Phase 3 Stage E readiness did not satisfy the modular-cognitive-runtime minimum for common spike space integrity (metric.common_spike_space_integrity, metric=common_spike_space_integrity, value=0.000, required>=1.000).",
    ]

    details = module.build_release_gate_error_details(errors)

    assert len(details) == 1
    detail = details[0]
    assert detail["type"] == "minimum_threshold_failure"
    assert detail["stage"] == "stage_e"
    assert detail["category"] == "stage_e.common_spike_space_integrity"
    assert "stage_e.minimum_checks" in detail["inferred_checks"]
    assert detail["check_name"] == "metric.common_spike_space_integrity"
    assert detail["metric_name"] == "common_spike_space_integrity"
    assert detail["actual_value"] == 0.0
    assert detail["required_value"] == 1.0


def test_release_gate_rejects_missing_phase5_entry_readiness_with_recovery_action():
    module = _load_release_gate_module()
    accuracy_report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 0.95,
        "passed": True,
        "component_reports": {
            "agent_dialogue": {"passed": True},
            "sara_inference": {"passed": True},
            "spiking_llm": {"passed": True},
            "task_switch_adaptation": {"passed": True},
            "future_state_consistency": {"passed": True},
            "energy_efficiency": {"passed": True},
            "continual_consolidation": {"passed": True},
            "cognitive_runtime": {"passed": True},
        },
        "focus_summary": {
            "few_shot": {"passed": True},
            "continual": {"passed": True},
            "retrieval_hygiene": {"passed": True},
            "adaptive_readiness": {"passed": True},
            "predictive_readiness": {"passed": True},
            "efficiency_readiness": {"passed": True},
            "consolidation_readiness": {"passed": True},
            "cognitive_runtime_readiness": {"passed": True},
        },
        "stage_a_acceptance": {"passed": True},
        "stage_b_readiness": {"passed": True, "minimum_requirements_passed": True, "minimum_checks": {}},
        "stage_c_readiness": {"passed": True, "minimum_requirements_passed": True, "minimum_checks": {}},
        "stage_d_readiness": {"passed": True, "minimum_requirements_passed": True, "minimum_checks": {}},
        "stage_e_readiness": {"passed": True, "minimum_requirements_passed": True, "minimum_checks": {}},
    }

    errors = module.validate_phase3_accuracy_report(accuracy_report)
    actions = module.suggest_release_gate_recovery_actions(errors)
    inferred_checks = set()
    for detail in module.build_release_gate_error_details(errors):
        inferred_checks.update(detail.get("inferred_checks", []))

    assert any("phase5_predictive_coding" in error for error in errors)
    assert any("Phase 5 entry readiness" in error for error in errors)
    assert any(action["command"] == "python scripts/eval/phase5_predictive_coding_benchmark.py" for action in actions)
    assert "phase5.entry_readiness" in inferred_checks


def test_release_gate_builds_error_details_summary():
    module = _load_release_gate_module()
    details = [
        {
            "type": "minimum_threshold_failure",
            "category": "stage_d.minimum_checks",
            "metric_name": "astro_modulation_stability",
        },
        {
            "type": "metric_threshold_drop",
            "category": "stage_d.replay_noise_resilience_integrity",
            "metric_name": "continual_consolidation.replay_noise_resilience_integrity",
        },
        {
            "type": "general_error",
            "category": "release_gate.embedded_accuracy_present",
        },
    ]

    summary = module.build_release_gate_error_details_summary(details)

    assert summary["total"] == 3
    assert summary["by_type"]["minimum_threshold_failure"] == 1
    assert summary["by_type"]["metric_threshold_drop"] == 1
    assert summary["by_type"]["general_error"] == 1
    assert summary["by_category"]["stage_d.minimum_checks"] == 1
    assert summary["by_metric"]["astro_modulation_stability"] == 1
    assert summary["top_types"][0]["name"] in {
        "general_error",
        "metric_threshold_drop",
        "minimum_threshold_failure",
    }
    assert summary["top_categories"][0]["count"] == 1
    assert any(
        item.get("name") == "astro_modulation_stability"
        for item in summary["top_metrics"]
        if isinstance(item, dict)
    )


def test_release_gate_phase5_macro_and_subgoal_errors_have_explicit_details_and_actions():
    module = _load_release_gate_module()
    errors = [
        "Phase 5 entry readiness did not satisfy predictive-coding metric phase5_predictive_coding.macro_action_effectiveness (value=0.750, required>=1.000).",
        "Phase 5 entry readiness did not satisfy predictive-coding metric phase5_predictive_coding.subgoal_decomposition_integrity (value=0.800, required>=1.000).",
        "Phase 5 entry readiness did not satisfy predictive-coding metric phase5_predictive_coding.micro_es_policy_refinement_integrity (value=0.700, required>=1.000).",
    ]

    details = module.build_release_gate_error_details(errors)
    inferred = module._infer_failed_checks_from_errors(errors)
    actions = module.suggest_release_gate_recovery_actions(errors)

    assert any(
        item.get("metric_name") == "phase5_predictive_coding.macro_action_effectiveness"
        and item.get("type") == "minimum_threshold_failure"
        and "phase5.macro_action_effectiveness" in item.get("inferred_checks", [])
        for item in details
        if isinstance(item, dict)
    )
    assert any(
        item.get("metric_name") == "phase5_predictive_coding.subgoal_decomposition_integrity"
        and item.get("type") == "minimum_threshold_failure"
        and "phase5.subgoal_decomposition_integrity" in item.get("inferred_checks", [])
        for item in details
        if isinstance(item, dict)
    )
    assert "phase5.macro_action_effectiveness" in inferred
    assert "phase5.subgoal_decomposition_integrity" in inferred
    assert "phase5.micro_es_policy_refinement_integrity" in inferred
    assert any(
        action.get("title") == "Re-run Phase 5 Predictive Coding Benchmark"
        and "phase5.macro_action_effectiveness" in action.get("affected_checks", [])
        and "phase5.subgoal_decomposition_integrity" in action.get("affected_checks", [])
        and "phase5.micro_es_policy_refinement_integrity" in action.get("affected_checks", [])
        for action in actions
        if isinstance(action, dict)
    )


def test_release_gate_phase5_completion_missing_required_checks_are_structured():
    module = _load_release_gate_module()
    errors = [
        "Phase 5 completion gate check map is missing required checks: metric.macro_action_effectiveness, threshold.macro_action_effectiveness, subgoal_coverage_ratio, micro_es_low_rank_trace_complete, micro_es_fitness_improvement, micro_es_event_cost_reduction, micro_es_population_event_budget",
    ]

    inferred = module._infer_failed_checks_from_errors(errors)
    details = module.build_release_gate_error_details(errors)
    actions = module.suggest_release_gate_recovery_actions(errors)

    assert "phase5.completion_gate" in inferred
    assert "phase5.completion_required_checks" in inferred
    assert "phase5.micro_es_low_rank_trace_complete" in inferred
    assert "phase5.micro_es_fitness_improvement" in inferred
    assert "phase5.micro_es_event_cost_reduction" in inferred
    assert "phase5.micro_es_population_event_budget" in inferred
    assert len(details) == 1
    detail = details[0]
    assert detail["type"] == "missing_required_checks"
    assert detail["stage"] == "phase5_completion"
    assert detail["check_name"] == "phase5_completion.required_checks"
    assert "metric.macro_action_effectiveness" in detail["missing_checks"]
    assert "subgoal_coverage_ratio" in detail["missing_checks"]
    assert "micro_es_low_rank_trace_complete" in detail["missing_checks"]
    assert "micro_es_fitness_improvement" in detail["missing_checks"]
    assert "micro_es_event_cost_reduction" in detail["missing_checks"]
    assert "micro_es_population_event_budget" in detail["missing_checks"]
    assert any(
        action.get("title") == "Re-run Phase 5 Completion Gate"
        and "phase5.completion_required_checks" in action.get("affected_checks", [])
        and "phase5.micro_es_low_rank_trace_complete" in action.get("affected_checks", [])
        and "phase5.micro_es_population_event_budget" in action.get("affected_checks", [])
        for action in actions
        if isinstance(action, dict)
    )


def test_release_gate_validates_phase5_completion_gate_report_required_checks():
    module = _load_release_gate_module()

    assert module.validate_phase5_completion_gate_report(_phase5_completion_gate_report(True)) == []

    missing = _phase5_completion_gate_report(True)
    del missing["checks"]["micro_es_fitness_improvement"]
    errors = module.validate_phase5_completion_gate_report(missing)
    assert any("missing required checks" in error.lower() for error in errors)
    assert any("micro_es_fitness_improvement" in error for error in errors)

    failed = _phase5_completion_gate_report(True)
    failed["checks"]["micro_es_population_event_budget"] = {"passed": False}
    errors = module.validate_phase5_completion_gate_report(failed)
    assert any("contains failed checks" in error.lower() for error in errors)
    assert any("micro_es_population_event_budget" in error for error in errors)


def test_release_gate_rejects_failed_phase5_completion_gate_report():
    module = _load_release_gate_module()

    errors = module.validate_phase5_completion_gate_report(_phase5_completion_gate_report(False))

    assert any("did not pass" in error.lower() for error in errors)
    assert any("overall score" in error.lower() for error in errors)
    assert any("failed checks" in error.lower() for error in errors)
    assert any("reported errors" in error.lower() for error in errors)


def test_release_gate_external_validity_failures_are_structured():
    module = _load_release_gate_module()
    errors = [
        "Real-data external validity metric did not satisfy release threshold (performance_energy_ratio_proxy, value=1.250, required>=2.000).",
        "Real-data external validity check failed: ann_cost_advantage_proxy (value=1.500, required>=2.000).",
        "Real-data external validity check failed: trend.no_regressions (value=1.000, required<=0.000).",
    ]

    inferred = module._infer_failed_checks_from_errors(errors)
    details = module.build_release_gate_error_details(errors)
    actions = module.suggest_release_gate_recovery_actions(errors)

    assert "external_validity.report" in inferred
    assert "external_validity.performance_energy_ratio_proxy" in inferred
    assert "external_validity.ann_cost_advantage_proxy" in inferred
    assert any(
        detail.get("stage") == "external_validity"
        and detail.get("metric_name") == "performance_energy_ratio_proxy"
        and detail.get("actual_value") == 1.25
        for detail in details
        if isinstance(detail, dict)
    )
    assert any(
        detail.get("type") == "required_check_failure"
        and detail.get("check_name") == "ann_cost_advantage_proxy"
        and detail.get("actual_value") == 1.5
        and detail.get("required_value") == 2.0
        and detail.get("threshold_operator") == ">="
        for detail in details
        if isinstance(detail, dict)
    )
    assert any(
        detail.get("type") == "required_check_failure"
        and detail.get("check_name") == "trend.no_regressions"
        and detail.get("actual_value") == 1.0
        and detail.get("required_value") == 0.0
        and detail.get("threshold_operator") == "<="
        for detail in details
        if isinstance(detail, dict)
    )
    assert any(
        action.get("title") == "Re-run Real-Data External Validity Benchmark"
        and action.get("command") == "python scripts/eval/real_data_external_validity.py"
        and "external_validity.performance_energy_ratio_proxy" in action.get("affected_checks", [])
        for action in actions
        if isinstance(action, dict)
    )


def test_release_gate_validates_external_validity_report_thresholds():
    module = _load_release_gate_module()
    report = {
        "suite_name": "RealDataExternalValidity",
        "passed": False,
        "checks": {
            "real_data_task_count": True,
            "sparse_accuracy_floor": True,
            "sparse_matches_dense_accuracy": True,
            "summary_keyword_coverage_floor": True,
            "continual_memory_hit_rate_floor": True,
            "ann_cost_advantage_proxy": False,
            "performance_energy_ratio_proxy": False,
            "trend.no_regressions": True,
        },
        "metrics": {
            "real_data_qa_accuracy": 0.90,
            "ann_proxy_qa_accuracy": 0.95,
            "real_data_summary_keyword_coverage": 0.70,
            "continual_memory_hit_rate": 0.90,
            "performance_energy_ratio_proxy": 1.25,
            "ann_cost_advantage_proxy": 1.50,
        },
    }

    errors = module.validate_external_validity_report(report)

    assert any("did not pass" in error for error in errors)
    assert any("ann_cost_advantage_proxy" in error for error in errors)
    assert any("performance_energy_ratio_proxy" in error for error in errors)


def test_release_gate_prefers_external_validity_check_details():
    module = _load_release_gate_module()
    report = {
        "suite_name": "RealDataExternalValidity",
        "passed": False,
        "checks": {
            "real_data_task_count": True,
            "sparse_accuracy_floor": True,
            "sparse_matches_dense_accuracy": True,
            "summary_keyword_coverage_floor": True,
            "continual_memory_hit_rate_floor": True,
            "ann_cost_advantage_proxy": False,
            "performance_energy_ratio_proxy": False,
            "trend.no_regressions": True,
        },
        "check_details": {
            "ann_cost_advantage_proxy": {"passed": False, "value": 1.5, "required_min": 2.0},
            "performance_energy_ratio_proxy": {"passed": False, "value": 1.25, "required_min": 2.0},
        },
        "metrics": {
            "real_data_qa_accuracy": 1.0,
            "ann_proxy_qa_accuracy": 1.0,
            "real_data_summary_keyword_coverage": 1.0,
            "continual_memory_hit_rate": 1.0,
            "performance_energy_ratio_proxy": 99.0,
            "ann_cost_advantage_proxy": 99.0,
        },
    }

    errors = module.validate_external_validity_report(report)

    assert any("ann_cost_advantage_proxy (value=1.500, required>=2.000)" in error for error in errors)
    assert any("performance_energy_ratio_proxy (value=1.250, required>=2.000)" in error for error in errors)
    assert not any("metric did not satisfy release threshold" in error for error in errors)


def test_release_gate_builds_failure_focus():
    module = _load_release_gate_module()
    summary = {
        "top_categories": [
            {"name": "stage_d.minimum_checks", "count": 2},
            {"name": "stage_c.minimum_checks", "count": 1},
        ],
        "top_metrics": [
            {"name": "astro_modulation_stability", "count": 2},
        ],
    }
    repair_plan = {
        "selected_actions": [
            {
                "title": "Re-run Continual Consolidation Benchmark",
                "command": "python scripts/eval/continual_consolidation_benchmark.py",
                "priority": "high",
            }
        ]
    }

    focus = module.build_release_gate_failure_focus(summary, repair_plan)

    assert focus["primary_category"] == "stage_d.minimum_checks"
    assert focus["secondary_category"] == "stage_c.minimum_checks"
    assert focus["primary_metric"] == "astro_modulation_stability"
    assert focus["primary_action"]["title"] == "Re-run Continual Consolidation Benchmark"
    assert 0.0 <= float(focus["confidence"]) <= 1.0
