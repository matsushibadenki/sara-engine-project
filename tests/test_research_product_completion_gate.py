import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "eval" / "research_product_completion_gate.py"
    spec = importlib.util.spec_from_file_location("research_product_completion_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _policy_text() -> str:
    return (
        "Do not make runtime learning depend on backpropagation.\n"
        "Do not make dense matrix operations the primary runtime design.\n"
        "Do not require GPUs for correctness or normal operation.\n"
        "Generated files must stay in managed directories.\n"
        "Prefer `src/sara_engine/utils/project_paths.py` for new read/write paths.\n"
    )


def _roadmap_report(passed=True):
    return {
        "passed": bool(passed),
        "closure_done_count": 4 if passed else 2,
        "unchecked_marker_count": 0,
        "candidate_line_count": 8,
    }


def _phase3_report(passed=True):
    return {
        "phase3_completion": {
            "passed": bool(passed),
            "completion_score": 1.0 if passed else 0.5,
            "checks": {"overall": bool(passed), "stage_b": bool(passed)},
        }
    }


def _phase4_report(passed=True):
    value = 1.0 if passed else 0.0
    return {
        "passed": bool(passed),
        "overall_score": value,
        "metrics": {
            "structural_plasticity_stability": value,
            "hippocampal_transfer_integrity": value,
            "scale_out_retention_integrity": value,
            "continual_drift_recovery_integrity": value,
        },
        "quality_metrics": {
            "structural_synapse_ratio": 1.0,
        },
    }


def _phase5_completion_report(passed=True):
    return {
        "passed": bool(passed),
        "phase5_overall_score": 1.0 if passed else 0.0,
        "failed_checks": [] if passed else ["phase5_overall_score"],
        "check_count": 3,
        "pass_count": 3 if passed else 2,
    }


def _operational_report(passed=True):
    required = [
        "phase3_accuracy",
        "phase3_completion",
        "phase4_completion",
        "phase5_entry_gate",
        "phase5_completion_gate",
        "external_validity",
        "external_validity_ladder",
        "release_gate",
        "production_profile",
    ]
    return {
        "passed": bool(passed),
        "strict_production": True,
        "readiness_score": 1.0 if passed else 0.5,
        "checks": {
            name: {"passed": bool(passed), "errors": [] if passed else ["failed"]}
            for name in required
        },
    }


def _ann_efficiency_roadmap_report(passed=True):
    stage_names = [
        "stage_1_instrumented_sparse_proxy",
        "stage_2_limited_real_data_advantage",
        "stage_3_scale_ladder_advantage",
        "stage_4_production_regression_guard",
        "stage_5_neuromorphic_transfer_readiness",
        "stage_6_real_joule_measurement_readiness",
    ]
    return {
        "passed": bool(passed),
        "completion_score": 1.0 if passed else 0.8,
        "stage_count": 6,
        "passed_stage_count": 6 if passed else 5,
        "stages": [
            {"name": name, "passed": bool(passed or index < 4)}
            for index, name in enumerate(stage_names)
        ],
    }


def _sparse_diffusion_block_report(passed=True):
    value = 1.0 if passed else 0.0
    return {
        "suite_name": "SparseDiffusionBlockReadiness",
        "passed": bool(passed),
        "overall_score": value,
        "block_count": 3,
        "metrics": {
            "sparse_diffusion_partition_integrity": value,
            "sparse_diffusion_independent_block_integrity": value,
            "sparse_diffusion_denoise_accuracy": value,
            "sparse_diffusion_event_cost_advantage": 2.25 if passed else 1.0,
            "sparse_diffusion_block_ablation_integrity": value,
            "sparse_diffusion_single_pass_recurrent_integrity": value,
            "sparse_diffusion_policy_compatibility": value,
        },
        "threshold_results": {
            "partition_integrity": bool(passed),
            "independent_block_integrity": bool(passed),
            "denoise_accuracy": bool(passed),
            "event_cost_advantage": bool(passed),
            "block_ablation_integrity": bool(passed),
            "single_pass_recurrent_integrity": bool(passed),
            "policy_compatibility": bool(passed),
        },
    }


def _energy_measurement_session_plan(passed=True):
    return {
        "schema": "sara-energy-measurement-session-plan-v2",
        "status": "pending_measurement",
        "session_id": "ann-efficiency-real-joule",
        "measurement_path": "data/raw/energy_measurements.jsonl",
        "min_ann_to_sara_ratio": 1.0,
        "planned_run_count": 1 if passed else 0,
        "pairing_matrix": {
            "tasks": ["real_data_external_validity"],
            "systems": ["sara", "ann"],
            "required_rows_per_task": 2,
            "required_paired_replicates_per_task": 3,
        },
        "fair_comparison_contract": {
            "protocol_version": "sara-energy-fair-comparison-v2",
            "aggregation": "per-task median joule_per_success with MAD",
        },
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
                    "--trial-count <trials> --joules <J> --source real_energy_session "
                    "--pair-id <pair-id> --environment-fingerprint <sha256> "
                    "--task-fixture-hash <sha256> --success-criterion-id <criterion> "
                    "--measurement-boundary <boundary> --measurement-tool <tool> "
                    "--run-order <1-or-2>"
                ),
            }
        ]
        if passed
        else [],
    }


def _rust_core_readiness_report(passed=True):
    return {
        "schema": "sara-rust-core-readiness-v1",
        "status": "needs_build_or_review",
        "source_readiness_passed": bool(passed),
        "built_extension_readiness_passed": False,
        "checks": {
            "versions_match": bool(passed),
            "cargo_feature_split_ready": bool(passed),
            "pymodule_exports_registered": bool(passed),
            "rust_core_comments_english": bool(passed),
            "batch_sdr_parallelized": bool(passed),
            "benchmark_report_present": bool(passed),
            "cargo_test_passed": bool(passed),
        },
        "export_contract": {
            "missing_from_pymodule_registration": [] if passed else ["SpikeEngine"],
        },
        "benchmark_report": {
            "present": bool(passed),
            "path": "workspace/evaluation/rust_core_benchmark.json",
        },
    }


def _adaptive_credit_field_report(passed=True):
    value = 1.0 if passed else 0.5
    return {
        "schema": "sara-adaptive-credit-field-benchmark-v1",
        "passed": bool(passed),
        "observed_only": True,
        "metrics": {
            "decision_integrity": 1.0 if passed else 0.5,
            "harmful_update_suppression": 1.0 if passed else 0.5,
            "sparse_advantage_case_ratio": 1.0 if passed else 0.5,
            "sparse_active_fraction_vs_naive": 0.5 if passed else 1.1,
            "quantized_behavior_match": 1.0 if passed else 0.5,
            "max_updated_routes": 2 if passed else 4,
        },
    }


def _adaptive_credit_event_memory_report(passed=True):
    return {
        "schema": "sara-adaptive-credit-event-memory-benchmark-v1",
        "passed": bool(passed),
        "observed_only": True,
        "metrics": {
            "harmful_block_preserved_count": 1 if passed else 0,
            "credit_entry_count": 1 if passed else 0,
            "credit_strong_entry_present": bool(passed),
            "credit_weak_entry_evicted": bool(passed),
        },
    }


def _research_fixture_readiness_report(passed=True):
    return {
        "schema": "sara-research-fixture-readiness-v1",
        "passed": bool(passed),
        "fixture_path": "data/processed/benchmark_fixtures/external_validity_cases.jsonl",
        "case_count": 8 if passed else 4,
        "task_types": [
            "qa",
            "negative",
            "partial",
            "contrastive",
            "noisy",
            "adversarial",
            "delayed",
        ]
        if passed
        else ["qa"],
        "missing_task_types": [] if passed else ["delayed"],
        "coverage": {
            "has_repository_safe_fixture": bool(passed),
            "has_noisy_case": bool(passed),
            "has_adversarial_case": bool(passed),
            "has_delayed_recall_case": bool(passed),
            "has_abstention_cases": bool(passed),
            "has_retrieval_cases": bool(passed),
        },
    }


def _autobot_gap_loop_readiness_report(passed=True):
    return {
        "schema": "sara-autobot-gap-loop-readiness-v1",
        "passed": bool(passed),
        "metrics": {
            "accepted_count": 8 if passed else 0,
            "collection_target_count": 2 if passed else 0,
            "requested_slot_count": 2 if passed else 0,
            "gap_material_built_count": 2 if passed else 0,
            "gap_material_skipped_count": 0 if passed else 2,
            "gap_curriculum_enqueued_count": 2 if passed else 0,
            "queue_pending": 2 if passed else 0,
            "gap_build_coverage": 1.0 if passed else 0.0,
            "gap_enqueue_coverage": 1.0 if passed else 0.0,
            "gap_skip_ratio": 0.0 if passed else 1.0,
            "repair_curriculum_share": 0.75 if passed else 0.0,
            "replay_curriculum_share": 0.25 if passed else 0.0,
        },
        "checks": {
            "loop_report_present": {"passed": bool(passed)},
            "dataset_report_present": {"passed": bool(passed)},
            "gap_report_present": {"passed": bool(passed)},
            "enqueue_report_present": {"passed": bool(passed)},
            "collection_targets_present": {"passed": bool(passed)},
            "loop_passed": {"passed": bool(passed)},
            "accepted_materials_ready": {"passed": bool(passed)},
            "gap_material_coverage_ready": {"passed": bool(passed)},
            "gap_enqueue_ready": {"passed": bool(passed)},
            "repair_curriculum_present": {"passed": bool(passed)},
        },
    }


def _source_texts():
    return {
        "fix_memory": "def fix_inference_memory(): pass\n dry_run ensure_parent_directory memory_fix_report",
        "sara_cli": "fix-memory",
        "tools": "fix-memory",
    }


def test_research_product_completion_gate_passes_all_green():
    module = _load_module()

    report = module.build_research_product_completion_report(
        policy_text=_policy_text(),
        roadmap_report=_roadmap_report(True),
        phase3_report=_phase3_report(True),
        phase4_report=_phase4_report(True),
        phase5_completion_report=_phase5_completion_report(True),
        operational_report=_operational_report(True),
        ann_efficiency_roadmap_report=_ann_efficiency_roadmap_report(True),
        sparse_diffusion_block_report=_sparse_diffusion_block_report(True),
        energy_measurement_session_plan=_energy_measurement_session_plan(True),
        adaptive_credit_field_report=_adaptive_credit_field_report(True),
        adaptive_credit_event_memory_report=_adaptive_credit_event_memory_report(True),
        rust_core_readiness_report=_rust_core_readiness_report(True),
        research_fixture_readiness_report=_research_fixture_readiness_report(True),
        autobot_gap_loop_readiness_report=_autobot_gap_loop_readiness_report(True),
        source_texts=_source_texts(),
    )
    summary = module.format_research_product_completion_summary(report)

    assert report["passed"] is True
    assert report["completion_score"] == 1.0
    assert report["failed_checks"] == []
    assert report["artifact_state"]["energy_measurement_session_plan"] == "present"
    assert report["artifact_state"]["research_fixture_readiness"] == "passed"
    assert report["artifact_state"]["autobot_gap_loop_readiness"] == "passed"
    assert "- artifact_state: phase6=present, phase8=passed, phase7=passed" in summary
    assert "- phase6_energy_metrics: status=pending_measurement, session_id=ann-efficiency-real-joule, planned_runs=1, task_count=1" in summary
    assert "- phase8_baseline_metrics: roadmap_completion=1.000, passed_stages=6/6, fixture_cases=8, fixture_task_types=7" in summary
    assert "- ann_efficiency_roadmap: PASS" in summary
    assert "- energy_measurement_session_plan: PASS" in summary
    assert "- sparse_diffusion_block_readiness: PASS" in summary
    assert "- adaptive_credit_field: PASS" in summary
    assert "- adaptive_credit_event_memory: PASS" in summary
    assert "- rust_core_readiness: PASS" in summary
    assert "- research_fixture_readiness: PASS" in summary
    assert "- autobot_gap_loop_readiness: PASS" in summary
    assert "- autobot_gap_loop_metrics: requested_slots=2, build_coverage=1.000, enqueue_coverage=1.000, skip_ratio=0.000, repair_share=0.750, replay_share=0.250" in summary
    assert "- adaptive_credit_metrics: decision_integrity=1.000, harmful_update_suppression=1.000, quantized_behavior_match=1.000, sparse_active_fraction_vs_naive=0.500" in summary
    assert "- adaptive_credit_event_memory_metrics: harmful_block_preserved_count=1, credit_strong_entry_present=True, credit_weak_entry_evicted=True, credit_entry_count=1" in summary
    assert "- memory_repair_operations: PASS" in summary
    assert "- neuromorphic_hal_smoke: PASS" in summary


def test_research_product_completion_gate_reports_operational_failure():
    module = _load_module()

    report = module.build_research_product_completion_report(
        policy_text=_policy_text(),
        roadmap_report=_roadmap_report(True),
        phase3_report=_phase3_report(True),
        phase4_report=_phase4_report(True),
        phase5_completion_report=_phase5_completion_report(True),
        operational_report=_operational_report(False),
        ann_efficiency_roadmap_report=_ann_efficiency_roadmap_report(True),
        sparse_diffusion_block_report=_sparse_diffusion_block_report(True),
        energy_measurement_session_plan=_energy_measurement_session_plan(True),
        adaptive_credit_field_report=_adaptive_credit_field_report(True),
        adaptive_credit_event_memory_report=_adaptive_credit_event_memory_report(True),
        rust_core_readiness_report=_rust_core_readiness_report(True),
        research_fixture_readiness_report=_research_fixture_readiness_report(True),
        autobot_gap_loop_readiness_report=_autobot_gap_loop_readiness_report(True),
        source_texts=_source_texts(),
    )

    assert report["passed"] is False
    assert "operational_strict_production" in report["failed_checks"]
    assert report["checks"]["operational_strict_production"]["errors"]


def test_research_product_completion_gate_rejects_missing_memory_surface():
    module = _load_module()

    report = module.build_research_product_completion_report(
        policy_text=_policy_text(),
        roadmap_report=_roadmap_report(True),
        phase3_report=_phase3_report(True),
        phase4_report=_phase4_report(True),
        phase5_completion_report=_phase5_completion_report(True),
        operational_report=_operational_report(True),
        ann_efficiency_roadmap_report=_ann_efficiency_roadmap_report(True),
        sparse_diffusion_block_report=_sparse_diffusion_block_report(True),
        energy_measurement_session_plan=_energy_measurement_session_plan(True),
        adaptive_credit_field_report=_adaptive_credit_field_report(True),
        adaptive_credit_event_memory_report=_adaptive_credit_event_memory_report(True),
        rust_core_readiness_report=_rust_core_readiness_report(True),
        research_fixture_readiness_report=_research_fixture_readiness_report(True),
        autobot_gap_loop_readiness_report=_autobot_gap_loop_readiness_report(True),
        source_texts={"fix_memory": "", "sara_cli": "", "tools": ""},
    )

    assert report["passed"] is False
    assert "memory_repair_operations" in report["failed_checks"]


def test_research_product_completion_gate_rejects_ann_efficiency_roadmap_failure():
    module = _load_module()

    report = module.build_research_product_completion_report(
        policy_text=_policy_text(),
        roadmap_report=_roadmap_report(True),
        phase3_report=_phase3_report(True),
        phase4_report=_phase4_report(True),
        phase5_completion_report=_phase5_completion_report(True),
        operational_report=_operational_report(True),
        ann_efficiency_roadmap_report=_ann_efficiency_roadmap_report(False),
        sparse_diffusion_block_report=_sparse_diffusion_block_report(True),
        energy_measurement_session_plan=_energy_measurement_session_plan(True),
        adaptive_credit_field_report=_adaptive_credit_field_report(True),
        adaptive_credit_event_memory_report=_adaptive_credit_event_memory_report(True),
        rust_core_readiness_report=_rust_core_readiness_report(True),
        research_fixture_readiness_report=_research_fixture_readiness_report(True),
        autobot_gap_loop_readiness_report=_autobot_gap_loop_readiness_report(True),
        source_texts=_source_texts(),
    )

    assert report["passed"] is False
    assert "ann_efficiency_roadmap" in report["failed_checks"]
    assert report["checks"]["ann_efficiency_roadmap"]["errors"]


def test_research_product_completion_gate_rejects_sparse_diffusion_failure():
    module = _load_module()

    report = module.build_research_product_completion_report(
        policy_text=_policy_text(),
        roadmap_report=_roadmap_report(True),
        phase3_report=_phase3_report(True),
        phase4_report=_phase4_report(True),
        phase5_completion_report=_phase5_completion_report(True),
        operational_report=_operational_report(True),
        ann_efficiency_roadmap_report=_ann_efficiency_roadmap_report(True),
        sparse_diffusion_block_report=_sparse_diffusion_block_report(False),
        energy_measurement_session_plan=_energy_measurement_session_plan(True),
        adaptive_credit_field_report=_adaptive_credit_field_report(True),
        adaptive_credit_event_memory_report=_adaptive_credit_event_memory_report(True),
        rust_core_readiness_report=_rust_core_readiness_report(True),
        research_fixture_readiness_report=_research_fixture_readiness_report(True),
        autobot_gap_loop_readiness_report=_autobot_gap_loop_readiness_report(True),
        source_texts=_source_texts(),
    )

    assert report["passed"] is False
    assert "sparse_diffusion_block_readiness" in report["failed_checks"]
    assert report["checks"]["sparse_diffusion_block_readiness"]["errors"]


def test_research_product_completion_gate_rejects_bad_energy_measurement_session_plan():
    module = _load_module()

    report = module.build_research_product_completion_report(
        policy_text=_policy_text(),
        roadmap_report=_roadmap_report(True),
        phase3_report=_phase3_report(True),
        phase4_report=_phase4_report(True),
        phase5_completion_report=_phase5_completion_report(True),
        operational_report=_operational_report(True),
        ann_efficiency_roadmap_report=_ann_efficiency_roadmap_report(True),
        sparse_diffusion_block_report=_sparse_diffusion_block_report(True),
        energy_measurement_session_plan=_energy_measurement_session_plan(False),
        adaptive_credit_field_report=_adaptive_credit_field_report(True),
        adaptive_credit_event_memory_report=_adaptive_credit_event_memory_report(True),
        rust_core_readiness_report=_rust_core_readiness_report(True),
        research_fixture_readiness_report=_research_fixture_readiness_report(True),
        autobot_gap_loop_readiness_report=_autobot_gap_loop_readiness_report(True),
        source_texts=_source_texts(),
    )

    assert report["passed"] is False
    assert "energy_measurement_session_plan" in report["failed_checks"]
    assert report["checks"]["energy_measurement_session_plan"]["errors"]


def test_research_product_completion_gate_rejects_rust_core_readiness_failure():
    module = _load_module()

    report = module.build_research_product_completion_report(
        policy_text=_policy_text(),
        roadmap_report=_roadmap_report(True),
        phase3_report=_phase3_report(True),
        phase4_report=_phase4_report(True),
        phase5_completion_report=_phase5_completion_report(True),
        operational_report=_operational_report(True),
        ann_efficiency_roadmap_report=_ann_efficiency_roadmap_report(True),
        sparse_diffusion_block_report=_sparse_diffusion_block_report(True),
        energy_measurement_session_plan=_energy_measurement_session_plan(True),
        adaptive_credit_field_report=_adaptive_credit_field_report(True),
        adaptive_credit_event_memory_report=_adaptive_credit_event_memory_report(True),
        rust_core_readiness_report=_rust_core_readiness_report(False),
        research_fixture_readiness_report=_research_fixture_readiness_report(True),
        autobot_gap_loop_readiness_report=_autobot_gap_loop_readiness_report(True),
        source_texts=_source_texts(),
    )

    assert report["passed"] is False
    assert "rust_core_readiness" in report["failed_checks"]
    assert report["checks"]["rust_core_readiness"]["errors"]


def test_research_product_completion_gate_rejects_research_fixture_failure():
    module = _load_module()

    report = module.build_research_product_completion_report(
        policy_text=_policy_text(),
        roadmap_report=_roadmap_report(True),
        phase3_report=_phase3_report(True),
        phase4_report=_phase4_report(True),
        phase5_completion_report=_phase5_completion_report(True),
        operational_report=_operational_report(True),
        ann_efficiency_roadmap_report=_ann_efficiency_roadmap_report(True),
        sparse_diffusion_block_report=_sparse_diffusion_block_report(True),
        energy_measurement_session_plan=_energy_measurement_session_plan(True),
        adaptive_credit_field_report=_adaptive_credit_field_report(True),
        adaptive_credit_event_memory_report=_adaptive_credit_event_memory_report(True),
        rust_core_readiness_report=_rust_core_readiness_report(True),
        research_fixture_readiness_report=_research_fixture_readiness_report(False),
        autobot_gap_loop_readiness_report=_autobot_gap_loop_readiness_report(True),
        source_texts=_source_texts(),
    )

    assert report["passed"] is False
    assert "research_fixture_readiness" in report["failed_checks"]
    assert report["checks"]["research_fixture_readiness"]["errors"]


def test_research_product_completion_gate_rejects_autobot_gap_loop_readiness_failure():
    module = _load_module()

    report = module.build_research_product_completion_report(
        policy_text=_policy_text(),
        roadmap_report=_roadmap_report(True),
        phase3_report=_phase3_report(True),
        phase4_report=_phase4_report(True),
        phase5_completion_report=_phase5_completion_report(True),
        operational_report=_operational_report(True),
        ann_efficiency_roadmap_report=_ann_efficiency_roadmap_report(True),
        sparse_diffusion_block_report=_sparse_diffusion_block_report(True),
        energy_measurement_session_plan=_energy_measurement_session_plan(True),
        adaptive_credit_field_report=_adaptive_credit_field_report(True),
        adaptive_credit_event_memory_report=_adaptive_credit_event_memory_report(True),
        rust_core_readiness_report=_rust_core_readiness_report(True),
        research_fixture_readiness_report=_research_fixture_readiness_report(True),
        autobot_gap_loop_readiness_report=_autobot_gap_loop_readiness_report(False),
        source_texts=_source_texts(),
    )

    assert report["passed"] is False
    assert report["artifact_state"]["autobot_gap_loop_readiness"] == "failed"
    assert "autobot_gap_loop_readiness" in report["failed_checks"]
    assert report["checks"]["autobot_gap_loop_readiness"]["errors"]


def test_research_product_completion_gate_rejects_adaptive_credit_field_failure():
    module = _load_module()

    report = module.build_research_product_completion_report(
        policy_text=_policy_text(),
        roadmap_report=_roadmap_report(True),
        phase3_report=_phase3_report(True),
        phase4_report=_phase4_report(True),
        phase5_completion_report=_phase5_completion_report(True),
        operational_report=_operational_report(True),
        ann_efficiency_roadmap_report=_ann_efficiency_roadmap_report(True),
        sparse_diffusion_block_report=_sparse_diffusion_block_report(True),
        energy_measurement_session_plan=_energy_measurement_session_plan(True),
        adaptive_credit_field_report=_adaptive_credit_field_report(False),
        adaptive_credit_event_memory_report=_adaptive_credit_event_memory_report(True),
        rust_core_readiness_report=_rust_core_readiness_report(True),
        research_fixture_readiness_report=_research_fixture_readiness_report(True),
        autobot_gap_loop_readiness_report=_autobot_gap_loop_readiness_report(True),
        source_texts=_source_texts(),
    )

    assert report["passed"] is False
    assert "adaptive_credit_field" in report["failed_checks"]
    assert report["checks"]["adaptive_credit_field"]["errors"]


def test_research_product_completion_gate_rejects_adaptive_credit_event_memory_failure():
    module = _load_module()

    report = module.build_research_product_completion_report(
        policy_text=_policy_text(),
        roadmap_report=_roadmap_report(True),
        phase3_report=_phase3_report(True),
        phase4_report=_phase4_report(True),
        phase5_completion_report=_phase5_completion_report(True),
        operational_report=_operational_report(True),
        ann_efficiency_roadmap_report=_ann_efficiency_roadmap_report(True),
        sparse_diffusion_block_report=_sparse_diffusion_block_report(True),
        energy_measurement_session_plan=_energy_measurement_session_plan(True),
        adaptive_credit_field_report=_adaptive_credit_field_report(True),
        adaptive_credit_event_memory_report=_adaptive_credit_event_memory_report(False),
        rust_core_readiness_report=_rust_core_readiness_report(True),
        research_fixture_readiness_report=_research_fixture_readiness_report(True),
        autobot_gap_loop_readiness_report=_autobot_gap_loop_readiness_report(True),
        source_texts=_source_texts(),
    )

    assert report["passed"] is False
    assert "adaptive_credit_event_memory" in report["failed_checks"]
    assert report["checks"]["adaptive_credit_event_memory"]["errors"]
