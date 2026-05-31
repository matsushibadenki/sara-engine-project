import importlib.util
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_gate_module():
    path = os.path.join(ROOT, "scripts", "eval", "ann_efficiency_roadmap_gate.py")
    spec = importlib.util.spec_from_file_location("ann_efficiency_roadmap_gate", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _energy_report():
    metrics = {
        "performance_energy_ratio_proxy": 0.22,
        "ann_cost_advantage_proxy": 12.0,
        "sparse_event_cost_score": 1.0,
        "brain_efficiency_alignment_proxy": 0.9,
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
    }
    return {"passed": True, "metrics": metrics}


def _external_validity_report(ratio=6.0):
    return {
        "passed": True,
        "metrics": {
            "real_data_qa_accuracy": 1.0,
            "real_data_summary_keyword_coverage": 0.8,
            "continual_memory_hit_rate": 1.0,
            "ann_cost_advantage_proxy": ratio,
            "performance_energy_ratio_proxy": ratio,
            "negative_control_abstention_integrity": 1.0,
            "negative_control_cost_advantage_proxy": 12.0,
            "partial_evidence_abstention_integrity": 1.0,
            "partial_evidence_cost_advantage_proxy": 8.0,
            "contrastive_control_accuracy": 1.0,
            "contrastive_control_cost_advantage_proxy": 7.0,
            "dense_embedding_ann_cost_advantage_proxy": 9.0,
            "sparse_diffusion_real_data_denoise_accuracy": 1.0,
            "sparse_diffusion_real_data_event_cost_advantage": 2.25,
            "sparse_diffusion_real_data_partition_integrity": 1.0,
            "sparse_diffusion_real_data_single_pass_integrity": 1.0,
        },
        "checks": {
            "trend.no_regressions": True,
        },
    }


def _external_ladder_report():
    return {
        "passed": True,
        "metrics": {
            "profile_count": 3,
            "min_real_data_qa_accuracy": 1.0,
            "min_ann_cost_advantage_proxy": 6.0,
            "min_performance_energy_ratio_proxy": 6.0,
            "min_negative_control_abstention_integrity": 1.0,
            "min_negative_control_cost_advantage_proxy": 12.0,
            "min_partial_evidence_abstention_integrity": 1.0,
            "min_partial_evidence_cost_advantage_proxy": 8.0,
            "min_contrastive_control_accuracy": 1.0,
            "min_contrastive_control_cost_advantage_proxy": 7.0,
            "min_dense_embedding_ann_cost_advantage_proxy": 9.0,
            "min_sparse_diffusion_real_data_denoise_accuracy": 1.0,
            "min_sparse_diffusion_real_data_event_cost_advantage": 2.25,
            "min_sparse_diffusion_real_data_partition_integrity": 1.0,
            "min_sparse_diffusion_real_data_single_pass_integrity": 1.0,
        },
        "checks": {
            "all_profiles_passed": True,
            "large_profile_present": True,
            "scale_doc_counts_monotonic": True,
            "negative_control_abstention_all_profiles": True,
            "negative_control_cost_advantage_all_profiles": True,
            "partial_evidence_abstention_all_profiles": True,
            "partial_evidence_cost_advantage_all_profiles": True,
            "contrastive_control_accuracy_all_profiles": True,
            "contrastive_control_cost_advantage_all_profiles": True,
            "dense_embedding_cost_advantage_all_profiles": True,
            "sparse_diffusion_real_data_denoise_all_profiles": True,
            "sparse_diffusion_real_data_cost_advantage_all_profiles": True,
            "sparse_diffusion_real_data_partition_all_profiles": True,
            "sparse_diffusion_real_data_single_pass_all_profiles": True,
            "no_trend_regressions_all_profiles": True,
        },
    }


def _energy_measurement_report(real_measurements=False):
    report = {
        "passed": True,
        "real_joule_measurements_present": bool(real_measurements),
        "checks": {
            "schema_ready": True,
            "rows_valid": True,
            "joule_efficiency_ratio_passed": bool(real_measurements),
            "paired_task_measurements_present": bool(real_measurements),
            "paired_task_rows_balanced": bool(real_measurements),
            "paired_task_efficiency_ratio_passed": bool(real_measurements),
        },
        "metrics": {
            "sara_joule_per_success": 0.2 if real_measurements else 0.0,
            "ann_joule_per_success": 0.8 if real_measurements else 0.0,
            "ann_to_sara_joule_efficiency_ratio": 4.0 if real_measurements else 0.0,
            "paired_task_count": 1.0 if real_measurements else 0.0,
            "min_paired_task_ann_to_sara_ratio": 4.0 if real_measurements else 0.0,
        },
    }
    if not real_measurements:
        report["measurement_plan"] = {
            "pending_pair_count": 1,
            "weak_pair_count": 0,
            "pending_pairs": [
                {
                    "task": "real_data_external_validity",
                    "missing_system": "sara",
                    "priority": "high",
                    "command_template": (
                        "python scripts/sara_cli.py record-energy-measurement "
                        "--run-id <run-id> --system sara --task real_data_external_validity "
                        "--success-count <count> --joules <J>"
                    ),
                }
            ],
            "weak_pairs": [],
        }
        report["measurement_session_plan"] = {
            "planned_run_count": 1,
            "planned_runs": [
                {
                    "category": "collect_missing_pair",
                    "priority": "high",
                    "task": "real_data_external_validity",
                    "system": "sara",
                    "run_id_template": "ann-efficiency-real-joule-real_data_external_validity-sara-<replicate>",
                    "command_template": (
                        "python scripts/sara_cli.py record-energy-measurement "
                        "--run-id ann-efficiency-real-joule-real_data_external_validity-sara-<replicate> "
                        "--system sara --task real_data_external_validity --success-count <count> "
                        "--joules <J> --source real_energy_session"
                    ),
                }
            ],
        }
    else:
        report["measurement_plan"] = {
            "pending_pair_count": 0,
            "weak_pair_count": 0,
            "pending_pairs": [],
            "weak_pairs": [],
        }
        report["measurement_session_plan"] = {"planned_run_count": 0, "planned_runs": []}
    return report


def _operational_report():
    return {
        "passed": True,
        "strict_production": True,
        "readiness_score": 1.0,
        "checks": {
            "external_validity": {"passed": True},
            "external_validity_ladder": {"passed": True},
        },
    }


def test_ann_efficiency_roadmap_gate_passes_all_stages():
    module = _load_gate_module()

    report = module.build_ann_efficiency_roadmap_report(
        energy_report=_energy_report(),
        external_validity_report=_external_validity_report(),
        external_ladder_report=_external_ladder_report(),
        energy_measurement_report=_energy_measurement_report(),
        operational_report=_operational_report(),
    )

    assert report["schema"] == "sara-ann-efficiency-roadmap-gate-v1"
    assert report["passed"] is True
    assert report["status"] == "ready_for_next_evidence_loop"
    assert report["completion_score"] == 1.0
    assert report["stage_count"] == 6
    assert report["failed_stages"] == []
    assert report["next_evidence_action_count"] == 1
    assert report["next_evidence_actions"][0]["source"] == "energy_measurement_session_plan"
    assert report["next_evidence_actions"][0]["category"] == "collect_missing_pair"
    assert report["next_evidence_actions"][0]["run_id_template"].startswith("ann-efficiency-real-joule")
    assert all(stage["passed"] for stage in report["stages"])

    summary = module.format_ann_efficiency_roadmap_summary(report)
    assert "SARA ANN Efficiency Roadmap Gate" in summary
    assert "stage_3_scale_ladder_advantage: PASS" in summary
    assert "Next Evidence Actions: 1" in summary
    assert "real_energy_session" in summary
    assert "record-energy-measurement" in summary


def test_ann_efficiency_roadmap_gate_blocks_weak_external_ratio():
    module = _load_gate_module()
    external_report = _external_validity_report(ratio=1.25)
    external_report["passed"] = False

    report = module.build_ann_efficiency_roadmap_report(
        energy_report=_energy_report(),
        external_validity_report=external_report,
        external_ladder_report=_external_ladder_report(),
        energy_measurement_report=_energy_measurement_report(),
        operational_report=_operational_report(),
    )

    assert report["passed"] is False
    assert report["status"] == "needs_targeted_repair"
    assert "stage_2_limited_real_data_advantage" in report["failed_stages"]
    stage_2 = report["stages"][1]
    assert stage_2["checks"]["ann_cost_advantage_proxy"] is False
    assert stage_2["checks"]["performance_energy_ratio_proxy"] is False
    assert stage_2["next_actions"]


def test_ann_efficiency_roadmap_gate_blocks_missing_negative_control():
    module = _load_gate_module()
    external_report = _external_validity_report()
    external_report["metrics"]["negative_control_abstention_integrity"] = 0.0
    external_report["checks"]["negative_control_abstention"] = False

    report = module.build_ann_efficiency_roadmap_report(
        energy_report=_energy_report(),
        external_validity_report=external_report,
        external_ladder_report=_external_ladder_report(),
        energy_measurement_report=_energy_measurement_report(),
        operational_report=_operational_report(),
    )

    assert report["passed"] is False
    assert "stage_2_limited_real_data_advantage" in report["failed_stages"]
    assert report["stages"][1]["checks"]["negative_control_abstention_integrity"] is False


def test_ann_efficiency_roadmap_accepts_real_joule_measurements():
    module = _load_gate_module()

    report = module.build_ann_efficiency_roadmap_report(
        energy_report=_energy_report(),
        external_validity_report=_external_validity_report(),
        external_ladder_report=_external_ladder_report(),
        energy_measurement_report=_energy_measurement_report(real_measurements=True),
        operational_report=_operational_report(),
    )

    assert report["passed"] is True
    stage_6 = report["stages"][5]
    assert stage_6["name"] == "stage_6_real_joule_measurement_readiness"
    assert stage_6["checks"]["real_joule_claim_guard"] is True
    assert stage_6["metrics"]["ann_to_sara_joule_efficiency_ratio"] == 4.0
    assert stage_6["metrics"]["min_paired_task_ann_to_sara_ratio"] == 4.0
    assert report["next_evidence_action_count"] == 0


def test_ann_efficiency_roadmap_blocks_unpaired_real_joule_claims():
    module = _load_gate_module()
    measurement_report = _energy_measurement_report(real_measurements=True)
    measurement_report["checks"]["paired_task_rows_balanced"] = False

    report = module.build_ann_efficiency_roadmap_report(
        energy_report=_energy_report(),
        external_validity_report=_external_validity_report(),
        external_ladder_report=_external_ladder_report(),
        energy_measurement_report=measurement_report,
        operational_report=_operational_report(),
    )

    assert report["passed"] is False
    assert "stage_6_real_joule_measurement_readiness" in report["failed_stages"]
    assert report["stages"][5]["checks"]["real_joule_claim_guard"] is False
