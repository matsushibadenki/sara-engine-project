import importlib.util
import json
import os
import sys
from unittest.mock import Mock


def _load_suite_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "research_benchmark_suite.py")
    )
    spec = importlib.util.spec_from_file_location("research_benchmark_suite", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_research_benchmark_suite_dry_run_writes_manifest(tmp_path):
    suite = _load_suite_module()
    manifest_path = suite.workspace_path("evaluation", "test_research_benchmark_manifest.json")
    summary_path = suite.workspace_path("evaluation", "test_research_benchmark_summary.txt")

    exit_code = suite.main(
        [
            "--dry-run",
            "--rust-iterations",
            "3",
            "--manifest-path",
            manifest_path,
            "--summary-path",
            summary_path,
        ]
    )

    assert exit_code == 0
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["schema"] == "sara-research-benchmark-manifest-v1"
    assert manifest["dry_run"] is True
    assert manifest["rust_iterations"] == 3
    assert manifest["passed"] is True
    assert manifest["artifact_state"]["autobot_gap_loop_readiness"] in {"missing", "passed", "failed"}
    assert "operational_readiness" in manifest["artifact_state"]
    assert [item["command_id"] for item in manifest["commands"]] == [
        "research_fixture_readiness",
        "rust_core_readiness",
        "rust_core_benchmark",
        "neuromorphic_capability_matrix",
        "own_latent_learning",
            "own_latent_manifest",
            "phase14_completion",
            "internal_practical_integration",
            "autobot_gap_loop_prepare",
            "gap_materials_closed_loop",
            "autobot_gap_loop_readiness",
        "dendritic_feedback_gate",
        "phase15_completion",
        "sparse_plan_trace_verifier",
        "sparse_reasoning_prior",
        "resonance_credit",
        "synesthetic_multimodal_binding",
        "phase16_completion",
        "resonance_credit_integration",
        "phase17_completion",
        "adaptive_credit_field",
        "adaptive_credit_event_memory",
        "event_state_cache",
        "event_state_cache_integration",
        "phase18_completion",
        "phase19_liquid_time_constant",
        "phase19_completion",
        "phase20_semantic_echo_field",
        "phase20_completion",
        "concept_revalidation_fixture_builder",
        "persistent_self_state",
        "idle_replay",
        "internal_maintenance_efficiency",
        "event_memory_ingest_pipeline",
        "event_memory_maintenance_coupling",
        "sara_ann_comparison",
        "phase13_capability_expansion",
        "research_product_completion",
        "v1_release_gate",
    ]
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = handle.read()
    assert "Artifact state: phase6=" in summary
    assert "Phase 6 energy metrics:" in summary
    assert "Phase 8 baseline metrics:" in summary
    assert "Phase 7 loop metrics:" in summary
    assert "Self-state maintenance:" in summary
    assert "Adaptive credit:" in summary
    assert "requested_slots=" in summary
    assert "Gap loop readiness: state=" in summary
    assert "What is proven:" in summary


def test_research_benchmark_suite_records_command_failure(monkeypatch, tmp_path):
    suite = _load_suite_module()
    mock_run = Mock()
    mock_run.return_value.returncode = 2
    monkeypatch.setattr(suite.subprocess, "run", mock_run)

    item = suite.BenchmarkCommand(
        command_id="failing_command",
        purpose="Exercise failure recording.",
        command=[sys.executable, "missing.py"],
        managed_outputs=[str(tmp_path / "missing.json")],
    )

    result = suite._run_command(item, dry_run=False)

    assert result["returncode"] == 2
    assert result["status"] == "failed"
    assert result["managed_outputs_present"][str(tmp_path / "missing.json")] is False


def test_research_benchmark_suite_exposes_concept_revalidation_evidence(monkeypatch):
    suite = _load_suite_module()
    original_loader = suite._load_json_if_present
    original_list_loader = suite._load_json_list_if_present

    def _stub_loader(path):
        if str(path).endswith("event_state_cache_integration_benchmark.json"):
            return {
                "passed": True,
                "metrics": {
                    "source_aware_logarithmic_delayed_recall": 1.0,
                    "round_trip_integrity": 1.0,
                    "concept_revalidation_case_count": 3,
                    "concept_revalidation_recovery_rate": 0.666667,
                    "concept_revalidation_blocked_count": 1,
                    "concept_revalidation_source_diversity_blocked_count": 0,
                    "concept_revalidation_revision_conflict_blocked_count": 0,
                    "concept_revalidation_counterexample_blocked_count": 0,
                    "concept_revalidation_attempt_budget_blocked_count": 1,
                },
                "next_actions": [
                    {
                        "priority": 4,
                        "reason": "attempt_budget",
                        "action": "manual_review_high_stall_candidates",
                    }
                ],
            }
        if str(path).endswith("concept_revalidation_fixture_builder.json"):
            return {
                "passed": True,
                "case_count": 4,
                "case_type_counts": {
                    "recoverable_revision_conflict": 1,
                    "blocked_source_diversity": 1,
                    "blocked_counterexample_pressure": 1,
                    "blocked_attempt_budget": 1,
                },
                "manifest_material_type_counts": {
                    "qa_pair": 2,
                    "source_claim": 1,
                    "contrastive_pair": 1,
                },
                "next_actions": [
                    {
                        "priority": 5,
                        "reason": "source_diversity",
                        "action": "collect_additional_distinct_sources",
                        "case_type": "blocked_source_diversity",
                        "case_count": 1,
                    }
                ],
                "expansion_plan": [
                    {
                        "action": "collect_additional_distinct_sources",
                        "case_type": "blocked_source_diversity",
                        "priority": 5,
                        "target_case_count": 1,
                        "preferred_material_types": [
                            "source_claim",
                            "qa_pair",
                            "transcript_segment",
                        ],
                        "available_material_types": {
                            "source_claim": 1,
                            "qa_pair": 2,
                        },
                        "missing_material_types": ["transcript_segment"],
                        "guidance": "Increase distinct source_ref coverage for repeated relation candidates.",
                    }
                ],
            }
        if str(path).endswith("own_latent_manifest_builder.json"):
            return {
                "passed": True,
                "manifest_count": 6,
                "fixture_feedback_loaded": True,
                "fixture_material_coverage_gap_count": 2,
                "fixture_material_request_count": 2,
                "fixture_expansion_plan": [
                    {
                        "action": "collect_additional_distinct_sources",
                        "preferred_material_types": [
                            "source_claim",
                            "qa_pair",
                            "transcript_segment",
                        ],
                        "missing_material_types_now": ["transcript_segment"],
                    }
                ],
            }
        if str(path).endswith("gap_materials_closed_loop_benchmark.json"):
            return {
                "passed": True,
                "baseline_fixture_material_coverage_gap_count": 4,
                "augmented_fixture_material_coverage_gap_count": 0,
                "coverage_gap_reduction": 4,
                "bundle_relevant_request_coverage": 1.0,
                "bundle_relevant_built_request_ids": [
                    "fixture_counterexample_gap",
                    "fixture_source_diversity_gap",
                ],
            }
        if str(path).endswith("autobot_gap_loop_readiness.json"):
            return {
                "passed": True,
                "metrics": {
                    "requested_slot_count": 2,
                    "gap_build_coverage": 1.0,
                    "fixture_request_count": 2,
                    "fixture_requested_slot_count": 2,
                    "fixture_gap_material_built_count": 2,
                    "fixture_gap_build_coverage": 1.0,
                    "fixture_source_domain_count": 2,
                    "fixture_source_lineage_coverage": 1.0,
                    "fixture_candidate_source_domain_count": 2,
                    "fixture_accepted_source_domain_count": 2,
                    "fixture_collection_time_coverage": 1.0,
                    "gap_enqueue_coverage": 1.0,
                    "gap_skip_ratio": 0.0,
                    "repair_curriculum_share": 0.75,
                    "replay_curriculum_share": 0.25,
                },
                "checks": {
                    "fixture_source_isolation_ready": {"passed": True},
                    "fixture_source_lineage_ready": {"passed": True},
                    "fixture_collection_time_ready": {"passed": True},
                },
                "fixture_isolation_audit": {
                    "missing_axes": [],
                },
                "fixture_request_isolation_audit": {
                    "fixture_counterexample_gap": {
                        "missing_axes": [],
                        "row_count": 1,
                        "lineage_coverage": 1.0,
                        "collection_time_coverage": 1.0,
                    },
                    "fixture_source_diversity_gap": {
                        "missing_axes": [],
                        "row_count": 1,
                        "lineage_coverage": 1.0,
                        "collection_time_coverage": 1.0,
                    },
                },
                "fixture_lane": {
                    "requested_slots_by_request": {
                        "fixture_counterexample_gap": 1,
                        "fixture_source_diversity_gap": 1,
                    },
                    "built_by_request": {
                        "fixture_counterexample_gap": 1,
                        "fixture_source_diversity_gap": 1,
                    },
                    "skipped_by_request": {
                        "fixture_counterexample_gap": 0,
                        "fixture_source_diversity_gap": 0,
                    },
                },
                "fixture_repair_actions": [
                    {
                        "request_id": "fixture_counterexample_gap",
                        "priority": "high",
                        "command": "python bot/gap_materials_builder.py --targets-path \"/tmp/targets.json\"",
                    }
                ],
            }
        if str(path).endswith("sara_ann_comparison_report.json"):
            return {
                "passed": False,
                "status": "proxy_only_or_partial_reference_surface",
                "completion_score": 0.7,
                "next_action_count": 2,
                "best_available_offline_reference": {
                    "label": "BM25 Offline Baseline",
                },
                "metrics": {
                    "ann_to_sara_joule_efficiency_ratio": 0.0,
                },
            }
        if str(path).endswith("persistent_self_state_benchmark.json"):
            return {
                "passed": True,
                "observed_only": True,
                "metrics": {
                    "persistent_self_state_idle_activity": 1.0,
                    "persistent_self_state_continuity": 1.0,
                    "persistent_self_state_memory_reactivation": 1.0,
                    "persistent_self_state_internal_prediction": 1.0,
                },
            }
        if str(path).endswith("idle_replay_benchmark.json"):
            return {
                "passed": True,
                "observed_only": True,
                "metrics": {
                    "idle_replay_candidate_selection_observed": 1.0,
                    "idle_replay_budget_observed": 1.0,
                    "idle_replay_self_state_alignment_observed": 1.0,
                    "idle_replay_memory_reactivation_observed": 1.0,
                    "idle_replay_state_continuity_observed": 1.0,
                },
            }
        if str(path).endswith("internal_maintenance_efficiency_benchmark.json"):
            return {
                "passed": True,
                "observed_only": True,
                "counts": {
                    "maintenance_selected_count": 4,
                    "maintenance_refresh_count": 2,
                },
                "normalized_metrics": {
                    "maintenance_event_cost": 6.0,
                    "maintenance_event_cost_per_selected": 1.5,
                },
            }
        if str(path).endswith("event_memory_ingest_pipeline.json"):
            return {
                "passed": True,
                "metrics": {
                    "episode_compression_ratio": 5.0,
                    "relation_verification_yield": 1.0,
                    "self_state_continuity": 0.285714,
                    "multimodal_bundle_promotion_rate": 1.0,
                    "multimodal_bundle_relation_verification_yield": 1.0,
                    "multimodal_bundle_compression_contribution": 5.0,
                },
                "traces": {
                    "multimodal_bundle_admission": {
                        "promotion_allowed_count": 1,
                    }
                },
            }
        if str(path).endswith("event_memory_maintenance_coupling_benchmark.json"):
            return {
                "passed": True,
                "best_profile": {
                    "profile_id": "balanced",
                },
                "metrics": {
                    "compression_to_maintenance_correlation": -0.25,
                    "best_profile_compression_efficiency_per_maintenance": 1.75,
                    "best_profile_self_state_continuity": 0.9,
                    "best_profile_episode_compression_ratio": 4.0,
                    "best_profile_multimodal_bundle_compression_contribution": 1.5,
                },
            }
        if str(path).endswith("adaptive_credit_field_benchmark.json"):
            return {
                "passed": True,
                "metrics": {
                    "sparse_active_fraction_vs_naive": 0.333333,
                    "quantized_behavior_match": 1.0,
                },
            }
        if str(path).endswith("adaptive_credit_event_memory_benchmark.json"):
            return {
                "passed": True,
                "metrics": {
                    "credit_strong_entry_present": True,
                    "credit_weak_entry_evicted": True,
                    "harmful_block_preserved_count": 1,
                },
            }
        if str(path).endswith("operational_readiness_report.json"):
            return {
                "passed": False,
                "checks": {
                    "adaptive_credit_field": {"passed": False},
                    "adaptive_credit_event_memory": {"passed": False},
                },
                "error_details": [
                    {"category": "adaptive_credit_field_validation"},
                    {"category": "adaptive_credit_event_memory_validation"},
                ],
                "recovery_actions": [
                    {
                        "command": "python scripts/eval/adaptive_credit_field_benchmark.py",
                        "affected_checks": ["adaptive_credit_field"],
                    },
                    {
                        "command": "python scripts/eval/adaptive_credit_event_memory_benchmark.py",
                        "affected_checks": ["adaptive_credit_event_memory", "event_memory_ingest_pipeline"],
                    },
                ],
                "failure_focus": {
                    "primary_category": "adaptive_credit_field_validation",
                },
                "bundle_repair_log_summary": {
                    "entry_count": 3,
                    "recovered_count": 1,
                    "max_gap_reduction": 4,
                    "isolation_clear_release_success_count": 1,
                    "isolation_clear_release_request_ids": [
                        "fixture_source_diversity_gap"
                    ],
                },
                "bundle_isolation_blocked_request_count": 1,
                "bundle_isolation_blocked_request_ids": ["fixture_counterexample_gap"],
                "bundle_isolation_blocked_missing_axes": ["source_lineage"],
                "runbook_actions": [
                    {
                        "source": "autobot_bundle_fixture_repair",
                        "return_phase": "phase7",
                        "reason": "bundle_gap_followup; return_phase=phase7",
                    }
                ],
                "repair_retry_queue": [
                    {
                        "source": "runbook_action:autobot_bundle_fixture_repair",
                        "bundle_closed_loop_overlap": True,
                        "bundle_recovered_before": False,
                        "request_id": "fixture_source_diversity_gap",
                        "isolation_review_churn_without_resolution": True,
                        "return_phase": "phase7",
                    },
                    {
                        "source": "runbook_action:autobot_bundle_fixture_repair",
                        "bundle_closed_loop_overlap": True,
                        "bundle_recovered_before": True,
                        "request_id": "fixture_counterexample_gap",
                        "isolation_reblocked_after_resolution": True,
                    },
                ],
                "bundle_retry_queue_summary": {
                    "entry_count": 2,
                    "fresh_count": 1,
                    "recovered_before_count": 1,
                    "isolation_review_churn_count": 1,
                    "isolation_review_churn_request_ids": ["fixture_source_diversity_gap"],
                    "isolation_reblocked_count": 1,
                    "isolation_reblocked_request_ids": ["fixture_counterexample_gap"],
                },
            }
        return original_loader(path)

    def _stub_list_loader(path):
        if str(path).endswith("operational_repair_execution_log.json"):
            return [
                {
                    "source": "adaptive_credit_repair",
                    "status": "success",
                    "covered_checks": ["adaptive_credit_field"],
                },
                {
                    "source": "adaptive_credit_repair",
                    "status": "success",
                    "covered_checks": ["adaptive_credit_event_memory", "event_memory_ingest_pipeline"],
                },
            ]
        return original_list_loader(path)

    monkeypatch.setattr(suite, "_load_json_if_present", _stub_loader)
    monkeypatch.setattr(suite, "_load_json_list_if_present", _stub_list_loader)

    manifest = suite.build_manifest(
        command_results=[],
        dry_run=True,
        rust_iterations=1,
    )

    evidence = manifest["evidence"]
    assert evidence["event_state_cache_concept_revalidation_case_count"] == 3
    assert evidence["event_state_cache_concept_revalidation_recovery_rate"] == 0.666667
    assert evidence["event_state_cache_concept_revalidation_blocked_count"] == 1
    assert evidence["event_state_cache_concept_source_diversity_blocked_count"] == 0
    assert evidence["event_state_cache_concept_revision_conflict_blocked_count"] == 0
    assert evidence["event_state_cache_concept_counterexample_blocked_count"] == 0
    assert evidence["event_state_cache_concept_attempt_budget_blocked_count"] == 1
    assert evidence["event_state_cache_concept_next_actions"][0]["action"] == "manual_review_high_stall_candidates"
    assert evidence["concept_revalidation_fixture_builder_passed"] is True
    assert evidence["concept_revalidation_fixture_case_count"] == 4
    assert evidence["concept_revalidation_fixture_case_type_counts"]["blocked_source_diversity"] == 1
    assert evidence["concept_revalidation_fixture_manifest_material_type_counts"]["qa_pair"] == 2
    assert evidence["concept_revalidation_fixture_next_actions"][0]["action"] == "collect_additional_distinct_sources"
    assert evidence["concept_revalidation_fixture_expansion_plan"][0]["missing_material_types"] == ["transcript_segment"]
    assert evidence["own_latent_fixture_feedback_loaded"] is True
    assert evidence["own_latent_fixture_material_coverage_gap_count"] == 2
    assert evidence["own_latent_fixture_material_request_count"] == 2
    assert evidence["own_latent_fixture_expansion_plan"][0]["missing_material_types_now"] == ["transcript_segment"]
    assert evidence["gap_materials_closed_loop_passed"] is True
    assert evidence["gap_materials_closed_loop_baseline_gap_count"] == 4
    assert evidence["gap_materials_closed_loop_augmented_gap_count"] == 0
    assert evidence["gap_materials_closed_loop_gap_reduction"] == 4
    assert evidence["autobot_gap_loop_readiness_passed"] is True
    assert evidence["autobot_gap_loop_requested_slot_count"] == 2
    assert evidence["autobot_gap_loop_build_coverage"] == 1.0
    assert evidence["autobot_gap_loop_fixture_request_count"] == 2
    assert evidence["autobot_gap_loop_fixture_requested_slot_count"] == 2
    assert evidence["autobot_gap_loop_fixture_gap_material_built_count"] == 2
    assert evidence["autobot_gap_loop_fixture_build_coverage"] == 1.0
    assert evidence["autobot_gap_loop_fixture_source_domain_count"] == 2
    assert evidence["autobot_gap_loop_fixture_source_lineage_coverage"] == 1.0
    assert evidence["autobot_gap_loop_fixture_candidate_source_domain_count"] == 2
    assert evidence["autobot_gap_loop_fixture_accepted_source_domain_count"] == 2
    assert evidence["autobot_gap_loop_fixture_collection_time_coverage"] == 1.0
    assert evidence["autobot_gap_loop_fixture_source_isolation_ready"] is True
    assert evidence["autobot_gap_loop_fixture_source_lineage_ready"] is True
    assert evidence["autobot_gap_loop_fixture_collection_time_ready"] is True
    assert evidence["autobot_gap_loop_fixture_missing_isolation_axes"] == []
    assert evidence["autobot_gap_loop_fixture_request_isolation_audit"]["fixture_counterexample_gap"]["missing_axes"] == []
    assert evidence["autobot_gap_loop_fixture_requested_slots_by_request"]["fixture_counterexample_gap"] == 1
    assert evidence["autobot_gap_loop_fixture_built_by_request"]["fixture_source_diversity_gap"] == 1
    assert evidence["autobot_gap_loop_fixture_skipped_by_request"]["fixture_counterexample_gap"] == 0
    assert evidence["autobot_gap_loop_fixture_repair_action_count"] == 1
    assert evidence["autobot_gap_loop_fixture_repair_request_ids"] == ["fixture_counterexample_gap"]
    assert evidence["autobot_gap_loop_enqueue_coverage"] == 1.0
    assert evidence["autobot_gap_loop_skip_ratio"] == 0.0
    assert evidence["autobot_gap_loop_repair_curriculum_share"] == 0.75
    assert evidence["autobot_gap_loop_replay_curriculum_share"] == 0.25
    assert manifest["artifact_state"]["gap_materials_closed_loop"] == "passed"
    assert manifest["artifact_state"]["autobot_gap_loop_readiness"] == "passed"
    assert manifest["artifact_state"]["persistent_self_state"] == "passed"
    assert manifest["artifact_state"]["idle_replay"] == "passed"
    assert manifest["artifact_state"]["internal_maintenance_efficiency"] == "passed"
    assert manifest["artifact_state"]["event_memory_ingest_pipeline"] == "passed"
    assert manifest["artifact_state"]["event_memory_maintenance_coupling"] == "passed"
    assert manifest["artifact_state"]["adaptive_credit_field"] == "passed"
    assert manifest["artifact_state"]["adaptive_credit_event_memory"] == "passed"
    assert manifest["artifact_state"]["sara_ann_comparison"] == "failed"
    assert evidence["persistent_self_state_passed"] is True
    assert evidence["persistent_self_state_idle_activity"] == 1.0
    assert evidence["idle_replay_passed"] is True
    assert evidence["idle_replay_self_state_alignment"] == 1.0
    assert evidence["internal_maintenance_efficiency_passed"] is True
    assert evidence["internal_maintenance_event_cost_per_selected"] == 1.5
    assert evidence["event_memory_ingest_pipeline_passed"] is True
    assert evidence["event_memory_episode_compression_ratio"] == 5.0
    assert evidence["event_memory_relation_verification_yield"] == 1.0
    assert evidence["event_memory_multimodal_bundle_relation_verification_yield"] == 1.0
    assert evidence["event_memory_multimodal_bundle_compression_contribution"] == 5.0
    assert evidence["event_memory_maintenance_coupling_passed"] is True
    assert evidence["event_memory_maintenance_best_profile"] == "balanced"
    assert evidence["event_memory_maintenance_best_efficiency"] == 1.75
    assert evidence["event_memory_maintenance_best_bundle_compression_contribution"] == 1.5
    assert evidence["event_memory_bundle_support_gap_present"] is False
    assert evidence["event_memory_bundle_support_gap_trigger"] == ""
    assert evidence["event_memory_bundle_support_repair_target"] is None
    assert evidence["event_memory_bundle_support_fixture_repair_action_count"] == 0
    assert evidence["event_memory_bundle_support_fixture_request_ids"] == []
    assert evidence["event_memory_bundle_support_fixture_coverage_ready"] is True
    assert evidence["event_memory_bundle_support_closed_loop_overlap_count"] == 0
    assert evidence["event_memory_bundle_support_closed_loop_overlap_ids"] == []
    assert evidence["event_memory_bundle_support_closed_loop_gap_reduction"] == 0
    assert evidence["event_memory_bundle_support_closed_loop_coverage_ready"] is True
    assert evidence["event_memory_bundle_support_overlap_request_isolation_audit"] == {}
    assert evidence["event_memory_bundle_support_overlap_missing_isolation_axes"] == []
    assert evidence["event_memory_bundle_support_overlap_isolation_risk_count"] == 0
    assert evidence["event_memory_bundle_support_overlap_highest_risk_axis"] == ""
    assert evidence["event_memory_bundle_support_overlap_risk_priority"] == "none"
    assert evidence["event_memory_maintenance_best_continuity"] == 0.9
    assert evidence["adaptive_credit_field_passed"] is True
    assert evidence["adaptive_credit_field_sparse_active_fraction"] == 0.333333
    assert evidence["adaptive_credit_field_quantized_match"] == 1.0
    assert evidence["adaptive_credit_event_memory_passed"] is True
    assert evidence["adaptive_credit_event_memory_strong_entry_present"] is True
    assert evidence["adaptive_credit_event_memory_weak_entry_evicted"] is True
    assert evidence["adaptive_credit_event_memory_harmful_block_preserved_count"] == 1
    assert evidence["adaptive_credit_operational_visibility"] is True
    assert evidence["adaptive_credit_operational_error_count"] == 2
    assert evidence["adaptive_credit_operational_repair_action_count"] == 2
    assert evidence["adaptive_credit_operational_primary_focus"] == "adaptive_credit_field_validation"
    assert evidence["adaptive_credit_repair_log_entry_count"] == 2
    assert evidence["adaptive_credit_repair_log_success_count"] == 2
    assert evidence["adaptive_credit_repair_log_pending_count"] == 0
    assert evidence["adaptive_credit_repair_log_failure_count"] == 0
    assert evidence["adaptive_credit_repair_log_recovered"] is True
    assert evidence["adaptive_credit_repair_log_chronic"] is False
    assert evidence["operational_bundle_repair_log_entry_count"] == 3
    assert evidence["operational_bundle_repair_log_recovered_count"] == 1
    assert evidence["operational_bundle_repair_log_max_gap_reduction"] == 4
    assert evidence["operational_bundle_isolation_clear_release_success_count"] == 1
    assert evidence["operational_bundle_isolation_clear_release_request_ids"] == [
        "fixture_source_diversity_gap"
    ]
    assert evidence["operational_bundle_isolation_resolved_request_ids"] == [
        "fixture_source_diversity_gap"
    ]
    assert evidence["operational_bundle_retry_queue_fresh_count"] == 1
    assert evidence["operational_bundle_retry_queue_recovered_before_count"] == 1
    assert evidence["operational_bundle_retry_queue_isolation_review_churn_count"] == 1
    assert evidence["operational_bundle_retry_queue_isolation_review_churn_request_ids"] == [
        "fixture_source_diversity_gap"
    ]
    assert evidence["operational_bundle_retry_queue_isolation_reblocked_count"] == 1
    assert evidence["operational_bundle_retry_queue_isolation_reblocked_request_ids"] == [
        "fixture_counterexample_gap"
    ]
    assert evidence["operational_bundle_phase7_routed_action_count"] == 1
    assert evidence["operational_bundle_phase7_routed_retry_count"] == 1
    assert evidence["operational_bundle_phase7_isolation_ready"] is True
    assert evidence["operational_bundle_phase7_lineage_ready"] is True
    assert evidence["operational_bundle_phase7_collection_time_ready"] is True
    assert evidence["operational_bundle_phase7_missing_isolation_axes"] == []
    assert evidence["operational_bundle_phase7_request_isolation_audit"]["fixture_counterexample_gap"]["missing_axes"] == []
    assert evidence["operational_bundle_isolation_blocked_request_count"] == 1
    assert evidence["operational_bundle_isolation_blocked_request_ids"] == ["fixture_counterexample_gap"]
    assert evidence["operational_bundle_isolation_blocked_missing_axes"] == ["source_lineage"]
    assert evidence["operational_bundle_overlap_blocked_request_ids"] == []
    assert evidence["sara_ann_comparison_status"] == "proxy_only_or_partial_reference_surface"
    assert evidence["sara_ann_best_offline_reference"] == "BM25 Offline Baseline"


def test_research_benchmark_suite_marks_bundle_support_gap_and_phase7_repair_target(monkeypatch, tmp_path):
    suite = _load_suite_module()
    original_loader = suite._load_json_if_present
    original_list_loader = suite._load_json_list_if_present

    def _stub_loader(path):
        payload = original_loader(path)
        if str(path).endswith("event_memory_maintenance_coupling_benchmark.json"):
            return {
                "passed": True,
                "best_profile": {"profile_id": "balanced"},
                "metrics": {
                    "compression_to_maintenance_correlation": -0.25,
                    "best_profile_compression_efficiency_per_maintenance": 1.75,
                    "best_profile_self_state_continuity": 0.9,
                    "best_profile_episode_compression_ratio": 4.0,
                    "best_profile_multimodal_bundle_compression_contribution": 0.1,
                },
            }
        if str(path).endswith("autobot_gap_loop_readiness.json"):
            return {
                "passed": True,
                "checks": {
                    "fixture_source_isolation_ready": {"passed": True},
                    "fixture_source_lineage_ready": {"passed": True},
                    "fixture_collection_time_ready": {"passed": True},
                },
                "fixture_isolation_audit": {
                    "missing_axes": [],
                },
                "fixture_request_isolation_audit": {
                    "fixture_counterexample_gap": {
                        "missing_axes": [],
                        "row_count": 0,
                        "lineage_coverage": 1.0,
                        "collection_time_coverage": 1.0,
                    }
                },
                "fixture_repair_actions": [
                    {
                        "request_id": "fixture_counterexample_gap",
                        "priority": "high",
                        "command": "Review fixture request fixture_counterexample_gap (missing_types=counterexample) and rerun python bot/gap_materials_builder.py --targets-path \"/tmp/targets.json\"",
                    }
                ],
            }
        if str(path).endswith("gap_materials_closed_loop_benchmark.json"):
            return {
                "passed": True,
                "baseline_fixture_material_coverage_gap_count": 4,
                "augmented_fixture_material_coverage_gap_count": 0,
                "coverage_gap_reduction": 4,
                "bundle_relevant_request_coverage": 1.0,
                "bundle_relevant_built_request_ids": [
                    "fixture_counterexample_gap",
                ],
            }
        return payload

    monkeypatch.setattr(suite, "_load_json_if_present", _stub_loader)
    monkeypatch.setattr(suite, "_load_json_list_if_present", original_list_loader)

    manifest = suite.build_manifest(
        command_results=[{"command_id": "dummy", "status": "passed"}],
        dry_run=True,
        rust_iterations=1,
    )
    evidence = manifest["evidence"]
    assert evidence["event_memory_bundle_support_gap_present"] is True
    assert evidence["event_memory_bundle_support_gap_trigger"] == "maintenance_coupling"
    assert evidence["event_memory_bundle_support_repair_target"] == "phase7_source_aware_bundle_fixtures"
    assert evidence["event_memory_bundle_support_fixture_repair_action_count"] == 1
    assert evidence["event_memory_bundle_support_fixture_request_ids"] == ["fixture_counterexample_gap"]
    assert evidence["event_memory_bundle_support_fixture_coverage_ready"] is True
    assert evidence["event_memory_bundle_support_closed_loop_overlap_count"] == 1
    assert evidence["event_memory_bundle_support_closed_loop_overlap_ids"] == ["fixture_counterexample_gap"]
    assert evidence["event_memory_bundle_support_closed_loop_gap_reduction"] == 4
    assert evidence["event_memory_bundle_support_closed_loop_coverage_ready"] is True
    assert evidence["event_memory_bundle_support_overlap_request_isolation_audit"]["fixture_counterexample_gap"]["missing_axes"] == []
    assert evidence["event_memory_bundle_support_overlap_missing_isolation_axes"] == []
    assert evidence["event_memory_bundle_support_overlap_isolation_risk_count"] == 0
    assert evidence["event_memory_bundle_support_overlap_highest_risk_axis"] == ""
    assert evidence["event_memory_bundle_support_overlap_risk_priority"] == "none"
    assert any(
        "Phase 7 source-aware bundle-fixture strengthening" in item
        for item in manifest["what_is_not_proven"]
    )

    manifest_path = suite.workspace_path("evaluation", "test_research_benchmark_bundle_gap_manifest.json")
    summary_path = suite.workspace_path("evaluation", "test_research_benchmark_bundle_gap_summary.txt")
    suite.write_outputs(manifest, manifest_path, summary_path)
    try:
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = handle.read()
        assert "bundle_gap=True" in summary
        assert "bundle_gap_trigger=maintenance_coupling" in summary
        assert "bundle_repair_target=phase7_source_aware_bundle_fixtures" in summary
        assert "bundle_fixture_repairs=1" in summary
        assert "bundle_closed_loop_overlap=1" in summary
        assert "bundle_closed_loop_gap_reduction=4" in summary
        assert "bundle_overlap_isolation_risk=0" in summary
        assert "bundle_overlap_missing_axes=none" in summary
        assert "bundle_overlap_highest_risk_axis=" in summary
        assert "bundle_overlap_risk_priority=none" in summary
        assert "Operational bundle repair:" in summary
        assert "phase7_isolation_ready=True" in summary
        assert "phase7_lineage_ready=True" in summary
        assert "phase7_collection_time_ready=True" in summary
        assert "phase7_missing_axes=none" in summary
        assert "churn_retry=" in summary
        assert "churn_request_ids=" in summary
        assert "reblocked_retry=" in summary
        assert "reblocked_request_ids=" in summary
        assert "blocked_request_count=" in summary
        assert "blocked_request_ids=" in summary
        assert "blocked_missing_axes=" in summary
        assert "overlap_blocked_request_ids=" in summary
        assert "clear_release_success_count=" in summary
        assert "clear_release_request_ids=" in summary
        assert "resolved_request_ids=" in summary
        assert "phase7_routed_actions=missing_artifact" not in summary
    finally:
        for path in (manifest_path, summary_path):
            if os.path.exists(path):
                os.remove(path)


def test_research_benchmark_suite_marks_bundle_overlap_isolation_risk_as_not_proven(monkeypatch):
    suite = _load_suite_module()
    original_loader = suite._load_json_if_present
    original_list_loader = suite._load_json_list_if_present

    def _stub_loader(path):
        payload = original_loader(path)
        if str(path).endswith("event_memory_maintenance_coupling_benchmark.json"):
            return {
                "passed": True,
                "best_profile": {"profile_id": "balanced"},
                "metrics": {
                    "compression_to_maintenance_correlation": -0.25,
                    "best_profile_compression_efficiency_per_maintenance": 1.75,
                    "best_profile_self_state_continuity": 0.9,
                    "best_profile_episode_compression_ratio": 4.0,
                    "best_profile_multimodal_bundle_compression_contribution": 0.1,
                },
            }
        if str(path).endswith("autobot_gap_loop_readiness.json"):
            return {
                "passed": True,
                "fixture_isolation_audit": {"missing_axes": ["collection_time"]},
                "fixture_request_isolation_audit": {
                    "fixture_counterexample_gap": {
                        "missing_axes": ["collection_time"],
                        "row_count": 1,
                        "lineage_coverage": 1.0,
                        "collection_time_coverage": 0.0,
                    }
                },
                "fixture_repair_actions": [
                    {"request_id": "fixture_counterexample_gap", "priority": "high", "command": "python bot/gap_materials_builder.py"}
                ],
            }
        if str(path).endswith("gap_materials_closed_loop_benchmark.json"):
            return {
                "passed": True,
                "baseline_fixture_material_coverage_gap_count": 4,
                "augmented_fixture_material_coverage_gap_count": 0,
                "coverage_gap_reduction": 4,
                "bundle_relevant_request_coverage": 1.0,
                "bundle_relevant_built_request_ids": ["fixture_counterexample_gap"],
            }
        return payload

    monkeypatch.setattr(suite, "_load_json_if_present", _stub_loader)
    monkeypatch.setattr(suite, "_load_json_list_if_present", original_list_loader)

    manifest = suite.build_manifest(
        command_results=[{"command_id": "dummy", "status": "passed"}],
        dry_run=True,
        rust_iterations=1,
    )

    assert manifest["evidence"]["event_memory_bundle_support_overlap_isolation_risk_count"] == 1
    assert manifest["evidence"]["event_memory_bundle_support_overlap_missing_isolation_axes"] == [
        "collection_time"
    ]
    assert manifest["evidence"]["event_memory_bundle_support_overlap_highest_risk_axis"] == "collection_time"
    assert manifest["evidence"]["event_memory_bundle_support_overlap_risk_priority"] == "medium"
    assert manifest["evidence"]["operational_bundle_overlap_blocked_request_ids"] == []
    assert any(
        "incomplete Phase 7 isolation audit axes (collection_time)" in item
        for item in manifest["what_is_not_proven"]
    )
    assert any(
        "prioritized as medium due to collection_time gaps" in item
        for item in manifest["what_is_not_proven"]
    )
    assert any(
        "blocked overlap repair requests are fixture_counterexample_gap" in item
        for item in manifest["what_is_not_proven"]
    )
