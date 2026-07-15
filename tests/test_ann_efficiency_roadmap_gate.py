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


def test_roadmap_gate_numeric_normalization_rejects_non_finite_values():
    module = _load_gate_module()
    assert module._float({"value": "NaN"}, "value") == 0.0
    assert module._float({"value": "Infinity"}, "value") == 0.0
    assert module._float({"value": True}, "value") == 0.0


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
        "reference_readiness": {
            "status": "partial",
            "references": [
                {
                    "reference_id": "ann_cross_encoder_reference",
                    "label": "Local Cross-Encoder Reference",
                    "available": False,
                    "reason": "not_configured",
                },
                {
                    "reference_id": "ann_pretrained_embedding_reference",
                    "label": "Local Embedding Reference",
                    "available": False,
                    "reason": "not_configured",
                },
            ],
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
            "maintenance_trace_rows_present": bool(real_measurements),
            "sara_maintenance_event_cost_per_success": 0.06 if real_measurements else 0.0,
            "ann_maintenance_event_cost_per_success": 0.02 if real_measurements else 0.0,
        },
        "maintenance_alignment": (
            {
                "available": True,
                "sara_physical_maintenance_event_cost_per_selected": 0.05,
                "reference_maintenance_event_cost_per_selected": 0.04,
                "maintenance_event_cost_per_selected_delta": 0.01,
                "maintenance_event_cost_per_selected_ratio": 1.25,
            }
            if real_measurements
            else {"available": False}
        ),
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
        report["measurement_session_progress"] = {
            "planned_pair_count": 1,
            "complete_valid_pair_count": 0,
            "partial_pair_count": 0,
            "invalid_pair_count": 0,
            "missing_pair_count": 1,
            "orphan_pair_count": 0,
            "pair_statuses": [
                {
                    "status": "missing_pair",
                    "priority": "high",
                    "task": "real_data_external_validity",
                    "pair_id": "ann-efficiency-real-joule-real_data_external_validity-pair-1",
                    "replicate_index": 1,
                    "pair_command": (
                        "python scripts/sara_cli.py run-physical-energy-pair "
                        "--pair-id ann-efficiency-real-joule-real_data_external_validity-pair-1 "
                        "--replicate-index 1"
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
        report["measurement_session_progress"] = {
            "planned_pair_count": 0,
            "complete_valid_pair_count": 0,
            "partial_pair_count": 0,
            "invalid_pair_count": 0,
            "missing_pair_count": 0,
            "orphan_pair_count": 0,
            "pair_statuses": [],
        }
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


def _internal_maintenance_report():
    return {
        "passed": True,
        "observed_only": True,
        "counts": {
            "maintenance_selected_count": 4,
            "maintenance_refresh_count": 2,
        },
        "normalized_metrics": {
            "maintenance_event_cost_per_selected": 1.5,
        },
        "metrics": {
            "maintenance_self_state_continuity_observed": 1.0,
            "maintenance_event_cost_efficiency_observed": 1.0,
        },
    }


def _event_memory_maintenance_coupling_reference():
    return {
        "available": True,
        "passed": True,
        "observed_only": True,
        "profile_count": 3,
        "best_profile_id": "wide",
        "compression_to_maintenance_correlation": 0.51,
        "best_profile_compression_efficiency_per_maintenance": 0.19,
        "best_profile_multimodal_bundle_compression_contribution": 1.83,
        "best_profile_self_state_continuity": 0.83,
        "best_profile_episode_compression_ratio": 3.67,
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
    assert report["artifact_state"]["energy_efficiency_benchmark"] == "passed"
    assert report["artifact_state"]["real_data_external_validity"] == "passed"
    assert report["artifact_state"]["real_data_external_validity_ladder"] == "passed"
    assert report["artifact_state"]["energy_measurement_readiness"] == "present"
    assert report["artifact_state"]["internal_maintenance_efficiency"] == "missing"
    assert report["artifact_state"]["event_memory_maintenance_coupling"] == "missing"
    assert report["artifact_state"]["operational_readiness"] == "passed"
    assert report["next_evidence_action_count"] == 5
    assert report["next_evidence_actions"][0]["source"] == "energy_measurement_session_progress"
    assert report["next_evidence_actions"][0]["category"] == "missing_pair"
    assert report["next_evidence_actions"][0]["pair_id"].startswith("ann-efficiency-real-joule")
    assert report["next_evidence_actions"][1]["source"] == "external_reference_readiness"
    assert report["next_evidence_actions"][1]["category"] == "configure_reference"
    assert report["next_evidence_actions"][3]["category"] == "missing_internal_maintenance_reference"
    assert report["next_evidence_actions"][4]["category"] == "missing_event_memory_maintenance_coupling_reference"
    assert all(stage["passed"] for stage in report["stages"])
    stage_6 = report["stages"][5]
    assert stage_6["metrics"]["maintenance_trace_rows_present"] == 0.0
    assert stage_6["metrics"]["internal_maintenance_event_cost_per_selected"] == 0.0

    summary = module.format_ann_efficiency_roadmap_summary(report)
    assert "SARA ANN Efficiency Roadmap Gate" in summary
    assert "- artifact_state: proxy=passed, phase8_single=passed, phase8_ladder=passed, phase6=present, maintenance=missing, coupling=missing, operational=passed" in summary
    assert "stage_3_scale_ladder_advantage: PASS" in summary
    assert "Next Evidence Actions: 5" in summary
    assert "run-physical-energy-pair" in summary
    assert "Configure Local Cross-Encoder Reference" in summary


def test_ann_efficiency_roadmap_gate_surfaces_maintenance_metrics_when_present():
    module = _load_gate_module()

    report = module.build_ann_efficiency_roadmap_report(
        energy_report=_energy_report(),
        external_validity_report=_external_validity_report(),
        external_ladder_report=_external_ladder_report(),
        energy_measurement_report=dict(
            _energy_measurement_report(real_measurements=True),
            event_memory_maintenance_coupling_reference=_event_memory_maintenance_coupling_reference(),
        ),
        internal_maintenance_report=_internal_maintenance_report(),
        operational_report=_operational_report(),
    )

    stage_6 = report["stages"][5]
    assert stage_6["metrics"]["maintenance_trace_rows_present"] == 1.0
    assert stage_6["metrics"]["sara_maintenance_event_cost_per_success"] == 0.06
    assert stage_6["metrics"]["ann_maintenance_event_cost_per_success"] == 0.02
    assert stage_6["metrics"]["internal_maintenance_event_cost_per_selected"] == 1.5
    assert stage_6["metrics"]["internal_maintenance_event_cost_efficiency_observed"] == 1.0
    assert stage_6["metrics"]["physical_internal_maintenance_alignment_available"] == 1.0
    assert stage_6["metrics"]["physical_internal_maintenance_alignment_ratio"] == 1.25
    assert stage_6["metrics"]["event_memory_maintenance_coupling_available"] == 1.0
    assert stage_6["metrics"]["event_memory_maintenance_best_efficiency"] == 0.19
    assert stage_6["metrics"]["event_memory_maintenance_best_bundle_compression_contribution"] == 1.83


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
        energy_measurement_report=dict(
            _energy_measurement_report(real_measurements=True),
            event_memory_maintenance_coupling_reference=_event_memory_maintenance_coupling_reference(),
        ),
        internal_maintenance_report=_internal_maintenance_report(),
        operational_report=_operational_report(),
    )

    assert report["passed"] is True
    stage_6 = report["stages"][5]
    assert stage_6["name"] == "stage_6_real_joule_measurement_readiness"
    assert stage_6["checks"]["real_joule_claim_guard"] is True
    assert stage_6["metrics"]["ann_to_sara_joule_efficiency_ratio"] == 4.0
    assert stage_6["metrics"]["min_paired_task_ann_to_sara_ratio"] == 4.0
    assert report["next_evidence_action_count"] == 2
    assert report["next_evidence_actions"][0]["source"] == "external_reference_readiness"


def test_ann_efficiency_roadmap_surfaces_maintenance_alignment_drift_action():
    module = _load_gate_module()
    measurement_report = _energy_measurement_report(real_measurements=True)
    measurement_report["maintenance_alignment"] = {
        "available": True,
        "sara_physical_maintenance_event_cost_per_selected": 0.09,
        "reference_maintenance_event_cost_per_selected": 0.04,
        "maintenance_event_cost_per_selected_delta": 0.05,
        "maintenance_event_cost_per_selected_ratio": 2.25,
    }
    measurement_report["event_memory_maintenance_coupling_reference"] = _event_memory_maintenance_coupling_reference()

    report = module.build_ann_efficiency_roadmap_report(
        energy_report=_energy_report(),
        external_validity_report=_external_validity_report(),
        external_ladder_report=_external_ladder_report(),
        energy_measurement_report=measurement_report,
        internal_maintenance_report=_internal_maintenance_report(),
        operational_report=_operational_report(),
    )

    action = next(
        action
        for action in report["next_evidence_actions"]
        if action["category"] == "maintenance_alignment_drift"
    )
    assert action["priority"] == "high"
    assert action["severity"] == "high"


def test_ann_efficiency_roadmap_blocks_unpaired_real_joule_claims():
    module = _load_gate_module()
    measurement_report = _energy_measurement_report(real_measurements=True)
    measurement_report["checks"]["paired_task_rows_balanced"] = False

    report = module.build_ann_efficiency_roadmap_report(
        energy_report=_energy_report(),
        external_validity_report=_external_validity_report(),
        external_ladder_report=_external_ladder_report(),
        energy_measurement_report=dict(
            measurement_report,
            event_memory_maintenance_coupling_reference=_event_memory_maintenance_coupling_reference(),
        ),
        internal_maintenance_report=_internal_maintenance_report(),
        operational_report=_operational_report(),
    )

    assert report["passed"] is False
    assert "stage_6_real_joule_measurement_readiness" in report["failed_stages"]
    assert report["stages"][5]["checks"]["real_joule_claim_guard"] is False


def test_ann_efficiency_roadmap_gate_requests_internal_maintenance_when_missing():
    module = _load_gate_module()

    report = module.build_ann_efficiency_roadmap_report(
        energy_report=_energy_report(),
        external_validity_report=_external_validity_report(),
        external_ladder_report=_external_ladder_report(),
        energy_measurement_report=_energy_measurement_report(real_measurements=True),
        operational_report=_operational_report(),
    )

    assert report["artifact_state"]["internal_maintenance_efficiency"] == "missing"
    assert any(
        action["category"] == "missing_internal_maintenance_reference"
        for action in report["next_evidence_actions"]
    )


def test_ann_efficiency_roadmap_gate_requests_event_memory_maintenance_coupling_when_missing():
    module = _load_gate_module()

    report = module.build_ann_efficiency_roadmap_report(
        energy_report=_energy_report(),
        external_validity_report=_external_validity_report(),
        external_ladder_report=_external_ladder_report(),
        energy_measurement_report=_energy_measurement_report(real_measurements=True),
        internal_maintenance_report=_internal_maintenance_report(),
        operational_report=_operational_report(),
    )

    assert report["artifact_state"]["event_memory_maintenance_coupling"] == "missing"
    assert any(
        action["category"] == "missing_event_memory_maintenance_coupling_reference"
        for action in report["next_evidence_actions"]
    )


def test_ann_efficiency_roadmap_gate_requests_event_memory_maintenance_coupling_when_weak():
    module = _load_gate_module()
    weak_reference = _event_memory_maintenance_coupling_reference()
    weak_reference["best_profile_compression_efficiency_per_maintenance"] = 0.0
    weak_reference["best_profile_self_state_continuity"] = 0.3

    report = module.build_ann_efficiency_roadmap_report(
        energy_report=_energy_report(),
        external_validity_report=_external_validity_report(),
        external_ladder_report=_external_ladder_report(),
        energy_measurement_report=dict(
            _energy_measurement_report(real_measurements=True),
            event_memory_maintenance_coupling_reference=weak_reference,
        ),
        internal_maintenance_report=_internal_maintenance_report(),
        operational_report=_operational_report(),
    )

    assert any(
        action["category"] == "weak_event_memory_maintenance_coupling_reference"
        for action in report["next_evidence_actions"]
    )


def test_ann_efficiency_roadmap_gate_requests_bundle_compression_repair_when_bundle_contribution_is_weak():
    module = _load_gate_module()
    weak_reference = _event_memory_maintenance_coupling_reference()
    weak_reference["best_profile_multimodal_bundle_compression_contribution"] = 0.1

    report = module.build_ann_efficiency_roadmap_report(
        energy_report=_energy_report(),
        external_validity_report=_external_validity_report(),
        external_ladder_report=_external_ladder_report(),
        energy_measurement_report=dict(
            _energy_measurement_report(real_measurements=True),
            event_memory_maintenance_coupling_reference=weak_reference,
        ),
        internal_maintenance_report=_internal_maintenance_report(),
        operational_report=_operational_report(),
    )

    action = next(
        action
        for action in report["next_evidence_actions"]
        if action["category"] == "weak_bundle_compression_contribution"
    )
    assert action["bundle_contribution"] == 0.1
    assert action["return_phase"] == "phase7"
    assert action["return_lane"] == "phase7_source_aware_bundle_fixtures"


def test_ann_efficiency_roadmap_gate_preserves_weak_pair_severity_from_measurement_plan():
    module = _load_gate_module()
    measurement_report = _energy_measurement_report(real_measurements=True)
    measurement_report["measurement_session_progress"] = {
        "planned_pair_count": 0,
        "complete_valid_pair_count": 0,
        "partial_pair_count": 0,
        "invalid_pair_count": 0,
        "missing_pair_count": 0,
        "orphan_pair_count": 0,
        "invalid_measurement_row_count": 0,
        "pair_statuses": [],
        "orphan_pairs": [],
    }
    measurement_report["measurement_plan"] = {
        "pending_pair_count": 0,
        "weak_pair_count": 1,
        "pending_pairs": [],
        "weak_pairs": [
            {
                "task": "real_data_external_validity",
                "ann_to_sara_joule_efficiency_ratio": 0.8,
                "required_min": 2.0,
                "relative_ratio": 0.4,
                "ratio_gap": 1.2,
                "severity": "critical",
                "priority": "high",
                "next_action": "Repeat this paired measurement with more replicates.",
            }
        ],
    }
    measurement_report["event_memory_maintenance_coupling_reference"] = (
        _event_memory_maintenance_coupling_reference()
    )

    report = module.build_ann_efficiency_roadmap_report(
        energy_report=_energy_report(),
        external_validity_report=_external_validity_report(),
        external_ladder_report=_external_ladder_report(),
        energy_measurement_report=measurement_report,
        internal_maintenance_report=_internal_maintenance_report(),
        operational_report=_operational_report(),
    )

    action = next(
        action
        for action in report["next_evidence_actions"]
        if action["category"] == "weak_joule_pair"
    )
    assert action["priority"] == "high"
    assert action["severity"] == "critical"
    assert action["relative_ratio"] == 0.4
    assert action["ratio_gap"] == 1.2


def test_ann_efficiency_roadmap_gate_surfaces_orphan_pair_repair_actions():
    module = _load_gate_module()
    measurement_report = _energy_measurement_report(real_measurements=True)
    measurement_report["measurement_session_progress"] = {
        "planned_pair_count": 1,
        "complete_valid_pair_count": 1,
        "partial_pair_count": 0,
        "invalid_pair_count": 0,
        "missing_pair_count": 0,
        "orphan_pair_count": 1,
        "invalid_measurement_row_count": 0,
        "pair_statuses": [],
        "orphan_pairs": [
            {
                "task": "extra_task",
                "pair_id": "orphan-pair",
                "replicate_index": 1,
                "present_systems": ["ann"],
            }
        ],
    }
    measurement_report["event_memory_maintenance_coupling_reference"] = (
        _event_memory_maintenance_coupling_reference()
    )

    report = module.build_ann_efficiency_roadmap_report(
        energy_report=_energy_report(),
        external_validity_report=_external_validity_report(),
        external_ladder_report=_external_ladder_report(),
        energy_measurement_report=measurement_report,
        internal_maintenance_report=_internal_maintenance_report(),
        operational_report=_operational_report(),
    )

    assert any(action["category"] == "orphan_pair" for action in report["next_evidence_actions"])
    assert report["stages"][5]["metrics"]["measurement_session_orphan_pair_count"] == 1.0


def test_ann_efficiency_roadmap_gate_surfaces_invalid_measurement_row_repair_actions():
    module = _load_gate_module()
    measurement_report = _energy_measurement_report(real_measurements=True)
    measurement_report["measurement_session_progress"] = {
        "planned_pair_count": 1,
        "complete_valid_pair_count": 0,
        "partial_pair_count": 0,
        "invalid_pair_count": 0,
        "missing_pair_count": 0,
        "orphan_pair_count": 0,
        "invalid_measurement_row_count": 2,
        "pair_statuses": [],
        "orphan_pairs": [],
    }
    measurement_report["event_memory_maintenance_coupling_reference"] = (
        _event_memory_maintenance_coupling_reference()
    )

    report = module.build_ann_efficiency_roadmap_report(
        energy_report=_energy_report(),
        external_validity_report=_external_validity_report(),
        external_ladder_report=_external_ladder_report(),
        energy_measurement_report=measurement_report,
        internal_maintenance_report=_internal_maintenance_report(),
        operational_report=_operational_report(),
    )

    assert any(
        action["category"] == "invalid_measurement_rows"
        for action in report["next_evidence_actions"]
    )
    assert report["stages"][5]["metrics"]["measurement_session_invalid_measurement_row_count"] == 2.0


def test_ann_efficiency_roadmap_gate_classifies_invalid_pair_run_order_conflict():
    module = _load_gate_module()
    measurement_report = _energy_measurement_report(real_measurements=True)
    measurement_report["measurement_session_progress"] = {
        "planned_pair_count": 1,
        "complete_valid_pair_count": 0,
        "partial_pair_count": 0,
        "invalid_pair_count": 1,
        "missing_pair_count": 0,
        "orphan_pair_count": 0,
        "invalid_measurement_row_count": 0,
        "pair_statuses": [
            {
                "status": "invalid_pair",
                "priority": "high",
                "task": "real_data_external_validity",
                "pair_id": "pair-1",
                "replicate_index": 1,
                "pair_command": "python scripts/sara_cli.py run-physical-energy-pair --pair-id pair-1",
                "invalid_reason_category": "run_order_conflict",
                "invalid_reason_fields": [],
            }
        ],
        "orphan_pairs": [],
    }
    measurement_report["event_memory_maintenance_coupling_reference"] = (
        _event_memory_maintenance_coupling_reference()
    )

    report = module.build_ann_efficiency_roadmap_report(
        energy_report=_energy_report(),
        external_validity_report=_external_validity_report(),
        external_ladder_report=_external_ladder_report(),
        energy_measurement_report=measurement_report,
        internal_maintenance_report=_internal_maintenance_report(),
        operational_report=_operational_report(),
    )

    action = next(
        action
        for action in report["next_evidence_actions"]
        if action["category"] == "invalid_pair_run_order_conflict"
    )
    assert action["priority"] == "medium"


def test_ann_efficiency_roadmap_gate_classifies_invalid_pair_fairness_mismatch():
    module = _load_gate_module()
    measurement_report = _energy_measurement_report(real_measurements=True)
    measurement_report["measurement_session_progress"] = {
        "planned_pair_count": 1,
        "complete_valid_pair_count": 0,
        "partial_pair_count": 0,
        "invalid_pair_count": 1,
        "missing_pair_count": 0,
        "orphan_pair_count": 0,
        "invalid_measurement_row_count": 0,
        "pair_statuses": [
            {
                "status": "invalid_pair",
                "priority": "high",
                "task": "real_data_external_validity",
                "pair_id": "pair-1",
                "replicate_index": 1,
                "pair_command": "python scripts/sara_cli.py run-physical-energy-pair --pair-id pair-1",
                "invalid_reason_category": "fairness_field_mismatch",
                "invalid_reason_fields": ["environment_fingerprint", "measurement_boundary"],
            }
        ],
        "orphan_pairs": [],
    }
    measurement_report["event_memory_maintenance_coupling_reference"] = (
        _event_memory_maintenance_coupling_reference()
    )

    report = module.build_ann_efficiency_roadmap_report(
        energy_report=_energy_report(),
        external_validity_report=_external_validity_report(),
        external_ladder_report=_external_ladder_report(),
        energy_measurement_report=measurement_report,
        internal_maintenance_report=_internal_maintenance_report(),
        operational_report=_operational_report(),
    )

    action = next(
        action
        for action in report["next_evidence_actions"]
        if action["category"] == "invalid_pair_fairness_mismatch"
    )
    assert action["priority"] == "high"
    assert "environment_fingerprint" in action["command"]
