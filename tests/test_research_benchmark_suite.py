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
    assert manifest["artifact_state"]["autobot_gap_loop_readiness"] == "missing"
    assert [item["command_id"] for item in manifest["commands"]] == [
        "research_fixture_readiness",
        "rust_core_readiness",
        "rust_core_benchmark",
        "neuromorphic_capability_matrix",
        "own_latent_learning",
        "own_latent_manifest",
        "gap_materials_closed_loop",
        "autobot_gap_loop_readiness",
        "dendritic_feedback_gate",
        "sparse_plan_trace_verifier",
        "sparse_reasoning_prior",
        "resonance_credit",
        "synesthetic_multimodal_binding",
        "resonance_credit_integration",
        "event_state_cache",
        "event_state_cache_integration",
        "concept_revalidation_fixture_builder",
        "research_product_completion",
        "v1_release_gate",
    ]
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = handle.read()
    assert "Artifact state: phase6=" in summary
    assert "Phase 6 energy metrics:" in summary
    assert "Phase 8 baseline metrics:" in summary
    assert "Phase 7 loop metrics:" in summary
    assert "requested_slots=missing_artifact" in summary
    assert "Gap loop readiness: state=missing" in summary
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
            }
        if str(path).endswith("autobot_gap_loop_readiness.json"):
            return {
                "passed": True,
                "metrics": {
                    "requested_slot_count": 2,
                    "gap_build_coverage": 1.0,
                    "gap_enqueue_coverage": 1.0,
                    "gap_skip_ratio": 0.0,
                    "repair_curriculum_share": 0.75,
                    "replay_curriculum_share": 0.25,
                },
            }
        return original_loader(path)

    monkeypatch.setattr(suite, "_load_json_if_present", _stub_loader)

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
    assert evidence["autobot_gap_loop_enqueue_coverage"] == 1.0
    assert evidence["autobot_gap_loop_skip_ratio"] == 0.0
    assert evidence["autobot_gap_loop_repair_curriculum_share"] == 0.75
    assert evidence["autobot_gap_loop_replay_curriculum_share"] == 0.25
    assert manifest["artifact_state"]["gap_materials_closed_loop"] == "passed"
    assert manifest["artifact_state"]["autobot_gap_loop_readiness"] == "passed"
