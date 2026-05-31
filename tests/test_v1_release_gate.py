import importlib.util
import os


def _load_script(script_name: str):
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..",
                     "scripts", "eval", script_name)
    )
    spec = importlib.util.spec_from_file_location(
        f"{script_name}_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase5_focus_summary(passed: bool = True):
    value = 1.0 if passed else 0.0
    return {
        "phase5_entry_readiness": {
            "passed": passed,
            "score": value,
            "metrics": {
                "phase5_predictive_coding.latent_transition_alignment": value,
                "phase5_predictive_coding.prediction_error_observability": value,
                "phase5_predictive_coding.correction_event_coverage": value,
                "phase5_predictive_coding.anti_collapse_event_diversity": value,
                "phase5_predictive_coding.counterfactual_transition_separation": value,
                "phase5_predictive_coding.multi_step_latent_chain_integrity": value,
                "phase5_predictive_coding.long_horizon_error_correction_convergence": value,
                "phase5_predictive_coding.horizon_bucket_stability": value,
                "phase5_predictive_coding.macro_action_effectiveness": value,
                "phase5_predictive_coding.subgoal_decomposition_integrity": value,
                "phase5_predictive_coding.depth_selective_routing_integrity": value,
                "phase5_predictive_coding.micro_es_policy_refinement_integrity": value,
            },
        }
    }


def _operational_phase5_snapshot(passed: bool = True):
    value = 1.0 if passed else 0.0
    return {
        "phase5_entry_readiness": {
            "passed": passed,
            "readiness_score": value,
            "latent_transition_alignment": value,
            "prediction_error_observability": value,
            "correction_event_coverage": value,
            "anti_collapse_event_diversity": value,
            "counterfactual_transition_separation": value,
            "multi_step_latent_chain_integrity": value,
            "long_horizon_error_correction_convergence": value,
            "horizon_bucket_stability": value,
            "macro_action_effectiveness": value,
            "subgoal_decomposition_integrity": value,
            "depth_selective_routing_integrity": value,
            "micro_es_policy_refinement_integrity": value,
        }
    }


def _phase5_completion_gate_report(passed: bool = True):
    value = 1.0 if passed else 0.0
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
        "phase5_entry_gate_passed": {"passed": passed},
        "multi_step_trace_complete": {"passed": passed},
        "counterfactual_branch_separable": {"passed": passed},
        "macro_step_reduction": {"passed": passed, "details": {"value": value * 3.0, "required_min": 2.0}},
        "macro_cost_reduction": {"passed": passed, "details": {"value": value * 0.42, "required_min": 0.30}},
        "subgoal_coverage_ratio": {"passed": passed, "details": {"value": value, "required_min": 1.0}},
        "micro_es_low_rank_trace_complete": {"passed": passed},
        "micro_es_fitness_improvement": {"passed": passed, "details": {"value": value * 0.249, "required_gt": 0.05}},
        "micro_es_event_cost_reduction": {"passed": passed, "details": {"value": value * 0.090, "required_min": 0.04}},
        "micro_es_population_event_budget": {
            "passed": passed,
            "details": {"value": 0.160 if passed else 0.400, "event_budget": 0.250},
        },
        "sparse_diffusion_block_readiness_passed": {"passed": passed},
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
        checks[f"sparse_diffusion.{name}"] = {"passed": passed}
    for name in entry_metrics:
        checks[f"metric.{name}"] = {"passed": passed}
        checks[f"threshold.{name}"] = {"passed": passed}
    return {
        "suite_name": "Phase5CompletionGate",
        "passed": passed,
        "phase5_overall_score": value,
        "failed_checks": [] if passed else ["metric.counterfactual_transition_separation"],
        "checks": checks,
    }


def _external_validity_report(passed: bool = True):
    value = 1.0 if passed else 0.0
    ratio = 4.0 if passed else 1.0
    checks = {
        "real_data_task_count": passed,
        "sparse_accuracy_floor": passed,
        "sparse_matches_dense_accuracy": passed,
        "summary_keyword_coverage_floor": passed,
        "continual_memory_hit_rate_floor": passed,
        "ann_cost_advantage_proxy": passed,
        "performance_energy_ratio_proxy": passed,
        "trend.no_regressions": passed,
    }
    return {
        "suite_name": "RealDataExternalValidity",
        "passed": passed,
        "checks": checks,
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
                "passed": passed,
                "value": ratio,
                "required_min": 2.0,
            },
            "ann_cost_advantage_proxy": {
                "passed": passed,
                "value": ratio,
                "required_min": 2.0,
            },
            "trend.no_regressions": {
                "passed": passed,
                "value": 0 if passed else 1,
                "required_max": 0,
            },
        },
    }


def _operational_stage_d_snapshot(passed: bool = True):
    value = 1.0 if passed else 0.0
    return {
        "stage_d_readiness": {
            "passed": passed,
            "minimum_requirements_passed": passed,
            "readiness_score": value,
            "replay_recovery_integrity": value,
            "long_horizon_consolidation_retention": value,
            "counterfactual_replay_selection_integrity": value,
            "replay_upgrade_reindex_integrity": value,
            "memory_health_index_integrity": value,
            "replay_noise_resilience_integrity": value,
            "astro_modulation_stability": value,
        }
    }


def _operational_stage_e_snapshot(passed: bool = True):
    value = 1.0 if passed else 0.0
    return {
        "stage_e_readiness": {
            "passed": passed,
            "minimum_requirements_passed": passed,
            "readiness_score": value,
            "common_spike_space_integrity": value,
            "temporal_compression_efficiency": value,
            "modality_temporal_budget_integrity": value,
            "dendritic_context_gate_stability": value,
            "spiking_hjepa_latent_transition": value,
            "reverse_reasoning_trace_integrity": value,
            "causal_candidate_trace_integrity": value,
            "module_orchestration_integrity": value,
            "counterfactual_lane_integrity": value,
            "action_trace_observability": value,
            "runtime_trace_replay_consistency": value,
        }
    }


def _stage_b_reward_policy_minimum(passed: bool = True):
    value = 1.0 if passed else 0.0
    return {
        "stage_b_readiness": {
            "minimum_requirements_passed": passed,
            "promotion_candidate_promoted": passed,
            "rlm_observation_candidate_promoted": passed,
            "minimum_checks": {
                "metric.future_state_rewarded_action_selection_integrity": passed,
                "metric.future_state_policy_update_stability": passed,
                "metric.future_state_energy_aware_action_preference": passed,
                "metric.future_state_focused_retrieval_hit_ratio": passed,
                "metric.future_state_branch_level_decision_consistency": passed,
            },
            "metrics": {
                "future_state_rewarded_action_selection_integrity": value,
                "future_state_policy_update_stability": value,
                "future_state_energy_aware_action_preference": value,
                "future_state_focused_retrieval_hit_ratio": value,
                "future_state_branch_level_decision_consistency": value,
            },
        }
    }


def _stage_d_minimum(passed: bool = True):
    value = 1.0 if passed else 0.0
    return {
        "stage_d_readiness": {
            "minimum_requirements_passed": passed,
            "minimum_checks": {
                "metric.replay_recovery_integrity": passed,
                "metric.long_horizon_consolidation_retention": passed,
                "metric.counterfactual_replay_selection_integrity": passed,
                "metric.replay_upgrade_reindex_integrity": passed,
                "metric.memory_health_index_integrity": passed,
                "metric.replay_noise_resilience_integrity": passed,
                "metric.astro_modulation_stability": passed,
            },
            "metrics": {
                "replay_recovery_integrity": value,
                "long_horizon_consolidation_retention": value,
                "counterfactual_replay_selection_integrity": value,
                "replay_upgrade_reindex_integrity": value,
                "memory_health_index_integrity": value,
                "replay_noise_resilience_integrity": value,
                "astro_modulation_stability": value,
            },
        }
    }


def _stage_e_minimum(passed: bool = True):
    value = 1.0 if passed else 0.0
    return {
        "stage_e_readiness": {
            "minimum_requirements_passed": passed,
            "minimum_checks": {
                "metric.common_spike_space_integrity": passed,
                "metric.temporal_compression_efficiency": passed,
                "metric.modality_temporal_budget_integrity": passed,
                "metric.dendritic_context_gate_stability": passed,
                "metric.spiking_hjepa_latent_transition": passed,
                "metric.reverse_reasoning_trace_integrity": passed,
                "metric.causal_candidate_trace_integrity": passed,
                "metric.module_orchestration_integrity": passed,
                "metric.counterfactual_lane_integrity": passed,
                "metric.action_trace_observability": passed,
                "metric.runtime_trace_replay_consistency": passed,
            },
            "metrics": {
                "common_spike_space_integrity": value,
                "temporal_compression_efficiency": value,
                "modality_temporal_budget_integrity": value,
                "dendritic_context_gate_stability": value,
                "spiking_hjepa_latent_transition": value,
                "reverse_reasoning_trace_integrity": value,
                "causal_candidate_trace_integrity": value,
                "module_orchestration_integrity": value,
                "counterfactual_lane_integrity": value,
                "action_trace_observability": value,
                "runtime_trace_replay_consistency": value,
            },
        }
    }


def _research_product_completion_report(passed: bool = True):
    check_count = 12
    return {
        "schema": "sara-research-product-completion-gate-v1",
        "passed": bool(passed),
        "completion_score": 1.0 if passed else 0.75,
        "check_count": check_count,
        "pass_count": check_count if passed else check_count - 1,
        "failed_checks": [] if passed else ["energy_measurement_session_plan"],
        "checks": {
            "energy_measurement_session_plan": {
                "passed": bool(passed),
                "errors": [] if passed else ["bad session plan"],
                "details": {"planned_run_count": 4 if passed else 0},
            }
        },
    }


def test_v1_release_gate_passes_when_all_requirements_are_met():
    module = _load_script("v1_release_gate.py")
    report = module.evaluate_v1_release_gate(
        operational_report={
            "passed": True,
            "strict_production": True,
            **_operational_phase5_snapshot(),
            **_operational_stage_d_snapshot(),
            **_operational_stage_e_snapshot(),
        },
        phase3_report={
            "overall_score": 0.98,
            "phase3_completion": {"passed": True},
            "focus_summary": _phase5_focus_summary(),
            **_stage_b_reward_policy_minimum(),
            **_stage_d_minimum(),
            **_stage_e_minimum(),
        },
        phase4_report={"passed": True, "overall_score": 1.0},
        phase5_completion_gate_report=_phase5_completion_gate_report(),
        research_product_completion_report=_research_product_completion_report(),
        pyproject_text='version = "1.1.0"\n',
        cargo_text='version = "1.1.0"\n',
    )
    assert report["passed"] is True
    assert report["failed_checks"] == []
    assert report["readiness_score"] == 1.0
    assert report["category_summary"]["stage_b"]["passed"] is True
    assert report["category_summary"]["stage_e"]["score"] == 1.0
    assert report["category_summary"]["research_product"]["passed"] is True
    assert report["failure_focus"]["primary_category"] == ""
    assert report["recovery_actions"] == []


def test_v1_release_gate_rejects_research_product_completion_failure():
    module = _load_script("v1_release_gate.py")
    report = module.evaluate_v1_release_gate(
        operational_report={
            "passed": True,
            "strict_production": True,
            **_operational_phase5_snapshot(),
            **_operational_stage_d_snapshot(),
            **_operational_stage_e_snapshot(),
        },
        phase3_report={
            "overall_score": 0.98,
            "phase3_completion": {"passed": True},
            "focus_summary": _phase5_focus_summary(),
            **_stage_b_reward_policy_minimum(),
            **_stage_d_minimum(),
            **_stage_e_minimum(),
        },
        phase4_report={"passed": True, "overall_score": 1.0},
        phase5_completion_gate_report=_phase5_completion_gate_report(),
        research_product_completion_report=_research_product_completion_report(False),
        pyproject_text='version = "1.1.0"\n',
        cargo_text='version = "1.1.0"\n',
    )

    assert report["passed"] is False
    assert "research_product_completion" in report["failed_checks"]
    assert report["category_summary"]["research_product"]["passed"] is False
    assert report["failure_focus"]["primary_category"] == "research_product"
    assert report["recovery_actions"][0]["command"] == "python scripts/eval/research_product_completion_gate.py"


def test_v1_release_gate_detects_version_alignment_failure():
    module = _load_script("v1_release_gate.py")
    report = module.evaluate_v1_release_gate(
        operational_report={
            "passed": True,
            "strict_production": True,
            **_operational_phase5_snapshot(),
            **_operational_stage_d_snapshot(),
            **_operational_stage_e_snapshot(),
        },
        phase3_report={
            "overall_score": 0.98,
            "phase3_completion": {"passed": True},
            "focus_summary": _phase5_focus_summary(),
            **_stage_b_reward_policy_minimum(),
            **_stage_d_minimum(),
            **_stage_e_minimum(),
        },
        phase4_report={"passed": True, "overall_score": 1.0},
        phase5_completion_gate_report=_phase5_completion_gate_report(),
        pyproject_text='version = "0.9.9"\n',
        cargo_text='version = "0.9.8"\n',
    )
    assert report["passed"] is False
    assert "version_alignment" in report["failed_checks"]
    assert report["category_summary"]["version"]["passed"] is False
    assert report["failure_focus"]["primary_category"] == "version"
    assert report["failure_focus"]["confidence"] == 1.0
    assert report["recovery_actions"][0]["category"] == "version"
    assert report["recovery_actions"][0]["priority"] == "medium"


def test_v1_release_gate_rejects_versions_below_target_release():
    module = _load_script("v1_release_gate.py")
    report = module.evaluate_v1_release_gate(
        operational_report={
            "passed": True,
            "strict_production": True,
            **_operational_phase5_snapshot(),
            **_operational_stage_d_snapshot(),
            **_operational_stage_e_snapshot(),
        },
        phase3_report={
            "overall_score": 0.98,
            "phase3_completion": {"passed": True},
            "focus_summary": _phase5_focus_summary(),
            **_stage_b_reward_policy_minimum(),
            **_stage_d_minimum(),
            **_stage_e_minimum(),
        },
        phase4_report={"passed": True, "overall_score": 1.0},
        phase5_completion_gate_report=_phase5_completion_gate_report(),
        pyproject_text='version = "1.0.0"\n',
        cargo_text='version = "1.0.0"\n',
    )

    assert report["passed"] is False
    assert "version_alignment" in report["failed_checks"]
    assert report["target_version"] == "1.1.0"
    details = report["checks"]["version_alignment"]["details"]
    assert details["versions_match"] is True
    assert details["target_version_met"] is False


def test_v1_release_gate_rejects_phase5_entry_regression():
    module = _load_script("v1_release_gate.py")
    report = module.evaluate_v1_release_gate(
        operational_report={
            "passed": True,
            "strict_production": True,
            **_operational_phase5_snapshot(),
            **_operational_stage_d_snapshot(),
            **_operational_stage_e_snapshot(),
        },
        phase3_report={
            "overall_score": 0.98,
            "phase3_completion": {"passed": True},
            "focus_summary": _phase5_focus_summary(passed=False),
            **_stage_b_reward_policy_minimum(),
            **_stage_d_minimum(),
            **_stage_e_minimum(),
        },
        phase4_report={"passed": True, "overall_score": 1.0},
        phase5_completion_gate_report=_phase5_completion_gate_report(),
        pyproject_text='version = "1.1.0"\n',
        cargo_text='version = "1.1.0"\n',
    )
    assert report["passed"] is False
    assert "phase5_entry_quality" in report["failed_checks"]


def test_v1_release_gate_rejects_missing_operational_phase5_snapshot():
    module = _load_script("v1_release_gate.py")
    report = module.evaluate_v1_release_gate(
        operational_report={
            "passed": True,
            "strict_production": True,
            **_operational_stage_d_snapshot(),
            **_operational_stage_e_snapshot(),
        },
        phase3_report={
            "overall_score": 0.98,
            "phase3_completion": {"passed": True},
            "focus_summary": _phase5_focus_summary(),
            **_stage_b_reward_policy_minimum(),
            **_stage_d_minimum(),
            **_stage_e_minimum(),
        },
        phase4_report={"passed": True, "overall_score": 1.0},
        phase5_completion_gate_report=_phase5_completion_gate_report(),
        pyproject_text='version = "1.1.0"\n',
        cargo_text='version = "1.1.0"\n',
    )
    assert report["passed"] is False
    assert "operational_phase5_snapshot" in report["failed_checks"]


def test_v1_release_gate_rejects_phase5_completion_gate_regression():
    module = _load_script("v1_release_gate.py")
    report = module.evaluate_v1_release_gate(
        operational_report={
            "passed": True,
            "strict_production": True,
            **_operational_phase5_snapshot(),
            **_operational_stage_d_snapshot(),
            **_operational_stage_e_snapshot(),
        },
        phase3_report={
            "overall_score": 0.98,
            "phase3_completion": {"passed": True},
            "focus_summary": _phase5_focus_summary(),
            **_stage_b_reward_policy_minimum(),
            **_stage_d_minimum(),
            **_stage_e_minimum(),
        },
        phase4_report={"passed": True, "overall_score": 1.0},
        phase5_completion_gate_report=_phase5_completion_gate_report(passed=False),
        pyproject_text='version = "1.1.0"\n',
        cargo_text='version = "1.1.0"\n',
    )
    assert report["passed"] is False
    assert "phase5_completion_quality" in report["failed_checks"]


def test_v1_release_gate_rejects_external_validity_regression():
    module = _load_script("v1_release_gate.py")
    report = module.evaluate_v1_release_gate(
        operational_report={
            "passed": True,
            "strict_production": True,
            **_operational_phase5_snapshot(),
            **_operational_stage_d_snapshot(),
            **_operational_stage_e_snapshot(),
        },
        phase3_report={
            "overall_score": 0.98,
            "phase3_completion": {"passed": True},
            "focus_summary": _phase5_focus_summary(),
            **_stage_b_reward_policy_minimum(),
            **_stage_d_minimum(),
            **_stage_e_minimum(),
        },
        phase4_report={"passed": True, "overall_score": 1.0},
        phase5_completion_gate_report=_phase5_completion_gate_report(),
        external_validity_report=_external_validity_report(passed=False),
        pyproject_text='version = "1.1.0"\n',
        cargo_text='version = "1.1.0"\n',
    )

    assert report["passed"] is False
    assert "external_validity_quality" in report["failed_checks"]
    assert report["category_summary"]["external_validity"]["passed"] is False
    details = report["checks"]["external_validity_quality"]["details"]
    assert details["failed_check_details"]["performance_energy_ratio_proxy"]["value"] == 1.0
    assert any(action["category"] == "external_validity" for action in report["recovery_actions"])


def test_v1_release_gate_rejects_phase5_completion_when_required_checks_missing():
    module = _load_script("v1_release_gate.py")
    completion_report = _phase5_completion_gate_report(True)
    completion_report["checks"] = {"phase5_entry_gate_passed": {"passed": True}}

    report = module.evaluate_v1_release_gate(
        operational_report={
            "passed": True,
            "strict_production": True,
            **_operational_phase5_snapshot(),
            **_operational_stage_d_snapshot(),
            **_operational_stage_e_snapshot(),
        },
        phase3_report={
            "overall_score": 0.98,
            "phase3_completion": {"passed": True},
            "focus_summary": _phase5_focus_summary(),
            **_stage_b_reward_policy_minimum(),
            **_stage_d_minimum(),
            **_stage_e_minimum(),
        },
        phase4_report={"passed": True, "overall_score": 1.0},
        phase5_completion_gate_report=completion_report,
        pyproject_text='version = "1.1.0"\n',
        cargo_text='version = "1.1.0"\n',
    )

    assert report["passed"] is False
    assert "phase5_completion_quality" in report["failed_checks"]
    details = report["checks"]["phase5_completion_quality"]["details"]
    assert "metric.macro_action_effectiveness" in details["missing_required_checks"]
    assert details["failed_required_checks"] == []


def test_v1_release_gate_rejects_phase5_completion_when_required_checks_fail():
    module = _load_script("v1_release_gate.py")
    completion_report = _phase5_completion_gate_report(True)
    completion_report["checks"]["metric.macro_action_effectiveness"] = {"passed": False}
    completion_report["checks"]["subgoal_coverage_ratio"] = {"passed": False}

    report = module.evaluate_v1_release_gate(
        operational_report={
            "passed": True,
            "strict_production": True,
            **_operational_phase5_snapshot(),
            **_operational_stage_d_snapshot(),
            **_operational_stage_e_snapshot(),
        },
        phase3_report={
            "overall_score": 0.98,
            "phase3_completion": {"passed": True},
            "focus_summary": _phase5_focus_summary(),
            **_stage_b_reward_policy_minimum(),
            **_stage_d_minimum(),
            **_stage_e_minimum(),
        },
        phase4_report={"passed": True, "overall_score": 1.0},
        phase5_completion_gate_report=completion_report,
        pyproject_text='version = "1.1.0"\n',
        cargo_text='version = "1.1.0"\n',
    )

    assert report["passed"] is False
    assert "phase5_completion_quality" in report["failed_checks"]
    details = report["checks"]["phase5_completion_quality"]["details"]
    assert "metric.macro_action_effectiveness" in details["failed_required_checks"]
    assert "subgoal_coverage_ratio" in details["failed_required_checks"]


def test_v1_release_gate_rejects_missing_stage_b_reward_policy_minimum():
    module = _load_script("v1_release_gate.py")
    report = module.evaluate_v1_release_gate(
        operational_report={
            "passed": True,
            "strict_production": True,
            **_operational_phase5_snapshot(),
            **_operational_stage_d_snapshot(),
            **_operational_stage_e_snapshot(),
        },
        phase3_report={
            "overall_score": 0.98,
            "phase3_completion": {"passed": True},
            "focus_summary": _phase5_focus_summary(),
            **_stage_b_reward_policy_minimum(passed=False),
            **_stage_d_minimum(),
            **_stage_e_minimum(),
        },
        phase4_report={"passed": True, "overall_score": 1.0},
        phase5_completion_gate_report=_phase5_completion_gate_report(),
        pyproject_text='version = "1.1.0"\n',
        cargo_text='version = "1.1.0"\n',
    )
    assert report["passed"] is False
    assert "stage_b_reward_policy_minimum" in report["failed_checks"]


def test_v1_release_gate_rejects_missing_stage_b_rlm_observation_minimum():
    module = _load_script("v1_release_gate.py")
    stage_b = _stage_b_reward_policy_minimum()
    stage_b["stage_b_readiness"]["rlm_observation_candidate_promoted"] = False
    stage_b["stage_b_readiness"]["minimum_checks"]["metric.future_state_focused_retrieval_hit_ratio"] = False
    stage_b["stage_b_readiness"]["metrics"]["future_state_focused_retrieval_hit_ratio"] = 0.0
    report = module.evaluate_v1_release_gate(
        operational_report={
            "passed": True,
            "strict_production": True,
            **_operational_phase5_snapshot(),
            **_operational_stage_d_snapshot(),
            **_operational_stage_e_snapshot(),
        },
        phase3_report={
            "overall_score": 0.98,
            "phase3_completion": {"passed": True},
            "focus_summary": _phase5_focus_summary(),
            **stage_b,
            **_stage_d_minimum(),
            **_stage_e_minimum(),
        },
        phase4_report={"passed": True, "overall_score": 1.0},
        phase5_completion_gate_report=_phase5_completion_gate_report(),
        pyproject_text='version = "1.1.0"\n',
        cargo_text='version = "1.1.0"\n',
    )
    assert report["passed"] is False
    assert "stage_b_rlm_observation_minimum" in report["failed_checks"]


def test_v1_release_gate_rejects_missing_stage_d_consolidation_minimum():
    module = _load_script("v1_release_gate.py")
    report = module.evaluate_v1_release_gate(
        operational_report={
            "passed": True,
            "strict_production": True,
            **_operational_phase5_snapshot(),
            **_operational_stage_d_snapshot(),
            **_operational_stage_e_snapshot(),
        },
        phase3_report={
            "overall_score": 0.98,
            "phase3_completion": {"passed": True},
            "focus_summary": _phase5_focus_summary(),
            **_stage_b_reward_policy_minimum(),
            **_stage_d_minimum(passed=False),
            **_stage_e_minimum(),
        },
        phase4_report={"passed": True, "overall_score": 1.0},
        phase5_completion_gate_report=_phase5_completion_gate_report(),
        pyproject_text='version = "1.1.0"\n',
        cargo_text='version = "1.1.0"\n',
    )
    assert report["passed"] is False
    assert "stage_d_consolidation_minimum" in report["failed_checks"]


def test_v1_release_gate_rejects_missing_operational_stage_d_snapshot():
    module = _load_script("v1_release_gate.py")
    report = module.evaluate_v1_release_gate(
        operational_report={
            "passed": True,
            "strict_production": True,
            **_operational_phase5_snapshot(),
            **_operational_stage_e_snapshot(),
        },
        phase3_report={
            "overall_score": 0.98,
            "phase3_completion": {"passed": True},
            "focus_summary": _phase5_focus_summary(),
            **_stage_b_reward_policy_minimum(),
            **_stage_d_minimum(),
            **_stage_e_minimum(),
        },
        phase4_report={"passed": True, "overall_score": 1.0},
        phase5_completion_gate_report=_phase5_completion_gate_report(),
        pyproject_text='version = "1.1.0"\n',
        cargo_text='version = "1.1.0"\n',
    )
    assert report["passed"] is False
    assert "operational_stage_d_snapshot" in report["failed_checks"]


def test_v1_release_gate_rejects_missing_stage_e_runtime_minimum():
    module = _load_script("v1_release_gate.py")
    report = module.evaluate_v1_release_gate(
        operational_report={
            "passed": True,
            "strict_production": True,
            **_operational_phase5_snapshot(),
            **_operational_stage_d_snapshot(),
            **_operational_stage_e_snapshot(),
        },
        phase3_report={
            "overall_score": 0.98,
            "phase3_completion": {"passed": True},
            "focus_summary": _phase5_focus_summary(),
            **_stage_b_reward_policy_minimum(),
            **_stage_d_minimum(),
            **_stage_e_minimum(passed=False),
        },
        phase4_report={"passed": True, "overall_score": 1.0},
        phase5_completion_gate_report=_phase5_completion_gate_report(),
        pyproject_text='version = "1.1.0"\n',
        cargo_text='version = "1.1.0"\n',
    )
    assert report["passed"] is False
    assert "stage_e_runtime_minimum" in report["failed_checks"]


def test_v1_release_gate_rejects_missing_operational_stage_e_snapshot():
    module = _load_script("v1_release_gate.py")
    report = module.evaluate_v1_release_gate(
        operational_report={
            "passed": True,
            "strict_production": True,
            **_operational_phase5_snapshot(),
            **_operational_stage_d_snapshot(),
        },
        phase3_report={
            "overall_score": 0.98,
            "phase3_completion": {"passed": True},
            "focus_summary": _phase5_focus_summary(),
            **_stage_b_reward_policy_minimum(),
            **_stage_d_minimum(),
            **_stage_e_minimum(),
        },
        phase4_report={"passed": True, "overall_score": 1.0},
        phase5_completion_gate_report=_phase5_completion_gate_report(),
        pyproject_text='version = "1.1.0"\n',
        cargo_text='version = "1.1.0"\n',
    )
    assert report["passed"] is False
    assert "operational_stage_e_snapshot" in report["failed_checks"]


def test_v1_release_gate_summary_includes_category_scores_and_failure_focus():
    module = _load_script("v1_release_gate.py")
    report = module.evaluate_v1_release_gate(
        operational_report={
            "passed": True,
            "strict_production": True,
            **_operational_phase5_snapshot(),
            **_operational_stage_d_snapshot(),
            **_operational_stage_e_snapshot(),
        },
        phase3_report={
            "overall_score": 0.98,
            "phase3_completion": {"passed": True},
            "focus_summary": _phase5_focus_summary(),
            **_stage_b_reward_policy_minimum(),
            **_stage_d_minimum(),
            **_stage_e_minimum(),
        },
        phase4_report={"passed": True, "overall_score": 1.0},
        phase5_completion_gate_report=_phase5_completion_gate_report(),
        pyproject_text='version = "0.9.9"\n',
        cargo_text='version = "0.9.8"\n',
    )

    summary = module.format_v1_summary(report)

    assert "- readiness_score: " in summary
    assert "- failure_focus_primary_category: version" in summary
    assert "- failure_focus_confidence: 1.000" in summary
    assert "- phase5_completion_missing_required_count: 0" in summary
    assert "- phase5_completion_failed_required_count: 0" in summary
    assert "- phase5_completion_micro_es_fitness_improvement_value: 0.249 required_gt=0.050" in summary
    assert "- phase5_completion_micro_es_population_event_budget_value: 0.160 event_budget=0.250" in summary


def test_v1_release_gate_summary_includes_phase5_completion_required_check_diagnostics():
    module = _load_script("v1_release_gate.py")
    report = module.evaluate_v1_release_gate(
        operational_report={
            "passed": True,
            "strict_production": True,
            **_operational_phase5_snapshot(),
            **_operational_stage_d_snapshot(),
            **_operational_stage_e_snapshot(),
        },
        phase3_report={
            "overall_score": 0.98,
            "phase3_completion": {"passed": True},
            "focus_summary": _phase5_focus_summary(),
            **_stage_b_reward_policy_minimum(),
            **_stage_d_minimum(),
            **_stage_e_minimum(),
        },
        phase4_report={"passed": True, "overall_score": 1.0},
        phase5_completion_gate_report={
            **_phase5_completion_gate_report(True),
            "checks": {
                "phase5_entry_gate_passed": {"passed": True},
                "metric.latent_transition_alignment": {"passed": True},
            },
        },
        pyproject_text='version = "1.1.0"\n',
        cargo_text='version = "1.1.0"\n',
    )

    summary = module.format_v1_summary(report)
    assert "- phase5_completion_missing_required_count:" in summary
    assert "phase5_completion_missing_required:" in summary
    assert "- recovery_action_count: 1" in summary
    assert "- category.phase5: FAIL" in summary
    assert "- recovery_action: category=phase5 priority=high command=python scripts/eval/phase5_predictive_coding_benchmark.py && python scripts/eval/phase5_entry_gate.py && python scripts/eval/phase5_completion_gate.py" in summary


def test_v1_release_gate_recovery_actions_prioritize_failed_runtime_categories():
    module = _load_script("v1_release_gate.py")
    report = module.evaluate_v1_release_gate(
        operational_report={
            "passed": True,
            "strict_production": True,
            **_operational_phase5_snapshot(),
            **_operational_stage_d_snapshot(),
            **_operational_stage_e_snapshot(),
        },
        phase3_report={
            "overall_score": 0.98,
            "phase3_completion": {"passed": True},
            "focus_summary": _phase5_focus_summary(passed=False),
            **_stage_b_reward_policy_minimum(passed=False),
            **_stage_d_minimum(),
            **_stage_e_minimum(),
        },
        phase4_report={"passed": True, "overall_score": 1.0},
        phase5_completion_gate_report=_phase5_completion_gate_report(),
        pyproject_text='version = "1.1.0"\n',
        cargo_text='version = "1.1.0"\n',
    )

    actions = report["recovery_actions"]
    commands = [str(action.get("command", "")) for action in actions]

    assert [action["priority"] for action in actions] == ["high", "high"]
    assert actions[0]["category"] == "phase5"
    assert actions[1]["category"] == "stage_b"
    assert any("phase5_predictive_coding_benchmark.py" in command for command in commands)
    assert any("future_state_consistency_benchmark.py" in command for command in commands)


def test_v1_release_gate_builds_runbook_actions_with_priority_and_dedup():
    module = _load_script("v1_release_gate.py")
    report = {
        "recovery_actions": [
            {
                "category": "version",
                "priority": "medium",
                "command": "review pyproject.toml Cargo.toml version fields",
                "expected_effect": "Align versions.",
                "affected_checks": ["version_alignment"],
            },
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
                "expected_effect": "Duplicate command should be removed.",
                "affected_checks": ["stage_b_rlm_observation_minimum"],
            },
        ]
    }

    actions = module.build_v1_runbook_actions(report)

    assert len(actions) == 2
    assert actions[0]["step"] == 1
    assert actions[0]["priority"] == "high"
    assert "future_state_consistency_benchmark.py" in actions[0]["command"]
    assert actions[1]["step"] == 2
    assert actions[1]["priority"] == "medium"
    assert actions[1]["command"] == "review pyproject.toml Cargo.toml version fields"
    assert isinstance(actions[0].get("generated_at"), float)
    assert isinstance(actions[1].get("generated_at"), float)
