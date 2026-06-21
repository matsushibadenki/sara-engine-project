import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import processed_data_path, workspace_path


def _load_module():
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "scripts",
            "eval",
            "event_state_cache_integration_benchmark.py",
        )
    )
    spec = importlib.util.spec_from_file_location(
        "event_state_cache_integration_benchmark",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_event_state_cache_integration_uses_managed_sources():
    module = _load_module()
    concept_fixture_path = processed_data_path(
        "benchmark_fixtures",
        "test_event_state_cache_concept_revalidation_cases.jsonl",
    )
    os.makedirs(os.path.dirname(concept_fixture_path), exist_ok=True)
    fixture_rows = [
        {
            "schema": "sara-concept-revalidation-case-v1",
            "case_id": "recoverable_revision_conflict",
            "case_type": "recoverable_revision_conflict",
            "concept_key": "predicts:vision:visual_cluster_018->audio:audio_cluster_044",
            "expected_outcome": "admit",
            "queue_entry": {
                "concept_key": "predicts:vision:visual_cluster_018->audio:audio_cluster_044",
                "decision": "quarantine_source_revision_conflict",
                "supporting_relation_ids": [
                    "predicts:vision:visual_cluster_018->audio:audio_cluster_044"
                ],
                "source_refs": ["https://example.org/a"],
                "source_hashes": ["hash-a"],
                "revision_conflict_count": 1,
                "contradiction_score": 0.2,
                "next_action": "wait_for_source_revision_resolution",
                "attempt_count": 0,
                "blocked_at_segment": 1,
                "last_review_segment": 1,
                "retry_after_segment": 2,
            },
            "relations": [
                {
                    "record_id": "fixture-rel-a",
                    "relation": "predicts",
                    "source_event_id": "vision:visual_cluster_018",
                    "target_event_id": "audio:audio_cluster_044",
                    "delay_lower_ms": 60,
                    "delay_upper_ms": 140,
                    "confidence": 0.88,
                    "evidence_count": 5,
                    "counterexample_count": 0,
                    "prediction_gain": 0.18,
                    "lineage": {
                        "source_ref": "https://example.org/a",
                        "source_hash": "hash-a",
                        "extractor_name": "prediction_gain",
                        "extractor_version": "v1",
                    },
                },
                {
                    "record_id": "fixture-rel-b",
                    "relation": "predicts",
                    "source_event_id": "vision:visual_cluster_018",
                    "target_event_id": "audio:audio_cluster_044",
                    "delay_lower_ms": 60,
                    "delay_upper_ms": 140,
                    "confidence": 0.88,
                    "evidence_count": 5,
                    "counterexample_count": 0,
                    "prediction_gain": 0.18,
                    "lineage": {
                        "source_ref": "https://example.org/b",
                        "source_hash": "hash-b",
                        "extractor_name": "prediction_gain",
                        "extractor_version": "v1",
                    },
                },
            ],
        }
    ]
    with open(concept_fixture_path, "w", encoding="utf-8") as handle:
        for row in fixture_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_path = workspace_path(
        "evaluation",
        "test_event_state_cache_integration.json",
    )
    exit_code = module.main(
        [
            "--manifest-path",
            processed_data_path("autobot", "latent_manifest.jsonl"),
            "--trace-path",
            workspace_path(
                "evaluation",
                "test_event_state_cache_integration_traces.jsonl",
            ),
            "--round-trip-state-path",
            workspace_path(
                "evaluation",
                "test_event_state_cache_round_trip.json",
            ),
            "--concept-queue-path",
            workspace_path(
                "evaluation",
                "test_event_state_cache_concept_queue.json",
            ),
            "--concept-review-report-path",
            workspace_path(
                "evaluation",
                "test_event_state_cache_concept_review.json",
            ),
            "--concept-fixture-path",
            concept_fixture_path,
            "--report-path",
            report_path,
            "--summary-path",
            workspace_path(
                "evaluation",
                "test_event_state_cache_integration.txt",
            ),
        ]
    )

    assert exit_code == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["passed"] is True
    assert report["metrics"]["source_aware_logarithmic_delayed_recall"] == 1.0
    assert report["metrics"]["round_trip_integrity"] == 1.0
    assert report["metrics"]["corrupted_state_rejection"] == 1.0
    assert report["metrics"]["missing_report_freeze_integrity"] == 1.0
    assert report["metrics"]["concept_revalidation_case_count"] >= 1
    assert report["metrics"]["concept_revalidation_fixture_case_count"] == 1
    assert report["metrics"]["concept_revalidation_ready_count"] >= 1
    assert report["metrics"]["concept_revalidation_blocked_count"] == 0
    assert report["metrics"]["concept_revalidation_attempt_budget_blocked_count"] == 0
    assert report["metrics"]["concept_revalidation_source_diversity_blocked_count"] == 0
    assert report["metrics"]["concept_revalidation_revision_conflict_blocked_count"] == 0
    assert report["metrics"]["concept_revalidation_counterexample_blocked_count"] == 0
    assert report["metrics"]["concept_revalidation_cooldown_blocked_count"] == 0
    assert report["metrics"]["concept_revalidation_other_blocked_count"] == 0
    assert report["metrics"]["concept_revalidation_admitted_count"] >= 1
    assert report["metrics"]["concept_revalidation_recovery_rate"] == 1.0
    assert report["metrics"]["concept_revalidation_recovered_integrity"] == 1.0
    assert report["metrics"]["concept_revalidation_queue_drained"] == 1.0
    assert report["next_actions"][0]["action"] == "scale_revalidation_case_coverage"
    assert report["concept_fixture_mode"] == "external_fixture"


def test_event_state_cache_integration_reports_mixed_fixture_outcomes():
    module = _load_module()
    concept_fixture_path = processed_data_path(
        "benchmark_fixtures",
        "test_event_state_cache_concept_revalidation_mixed_cases.jsonl",
    )
    os.makedirs(os.path.dirname(concept_fixture_path), exist_ok=True)
    fixture_rows = [
        {
            "schema": "sara-concept-revalidation-case-v1",
            "case_id": "recoverable_revision_conflict",
            "case_type": "recoverable_revision_conflict",
            "concept_key": "predicts:vision:visual_cluster_018->audio:audio_cluster_044",
            "expected_outcome": "admit",
            "queue_entry": {
                "concept_key": "predicts:vision:visual_cluster_018->audio:audio_cluster_044",
                "decision": "quarantine_source_revision_conflict",
                "supporting_relation_ids": [
                    "predicts:vision:visual_cluster_018->audio:audio_cluster_044"
                ],
                "source_refs": ["https://example.org/a"],
                "source_hashes": ["hash-a"],
                "revision_conflict_count": 1,
                "contradiction_score": 0.2,
                "next_action": "wait_for_source_revision_resolution",
                "attempt_count": 0,
                "blocked_at_segment": 1,
                "last_review_segment": 1,
                "retry_after_segment": 2,
            },
            "relations": [
                {
                    "record_id": "fixture-rel-a",
                    "relation": "predicts",
                    "source_event_id": "vision:visual_cluster_018",
                    "target_event_id": "audio:audio_cluster_044",
                    "delay_lower_ms": 60,
                    "delay_upper_ms": 140,
                    "confidence": 0.88,
                    "evidence_count": 5,
                    "counterexample_count": 0,
                    "prediction_gain": 0.18,
                    "lineage": {
                        "source_ref": "https://example.org/a",
                        "source_hash": "hash-a",
                        "extractor_name": "prediction_gain",
                        "extractor_version": "v1",
                    },
                },
                {
                    "record_id": "fixture-rel-b",
                    "relation": "predicts",
                    "source_event_id": "vision:visual_cluster_018",
                    "target_event_id": "audio:audio_cluster_044",
                    "delay_lower_ms": 60,
                    "delay_upper_ms": 140,
                    "confidence": 0.88,
                    "evidence_count": 5,
                    "counterexample_count": 0,
                    "prediction_gain": 0.18,
                    "lineage": {
                        "source_ref": "https://example.org/b",
                        "source_hash": "hash-b",
                        "extractor_name": "prediction_gain",
                        "extractor_version": "v1",
                    },
                },
            ],
        },
        {
            "schema": "sara-concept-revalidation-case-v1",
            "case_id": "blocked_source_diversity",
            "case_type": "blocked_source_diversity",
            "concept_key": "predicts:vision:visual_cluster_019->audio:audio_cluster_045",
            "expected_outcome": "blocked",
            "queue_entry": {
                "concept_key": "predicts:vision:visual_cluster_019->audio:audio_cluster_045",
                "decision": "reject_insufficient_source_diversity",
                "supporting_relation_ids": [
                    "predicts:vision:visual_cluster_019->audio:audio_cluster_045"
                ],
                "source_refs": ["https://example.org/a"],
                "source_hashes": ["hash-a"],
                "revision_conflict_count": 0,
                "contradiction_score": 0.0,
                "next_action": "collect_more_distinct_sources",
                "attempt_count": 0,
                "blocked_at_segment": 1,
                "last_review_segment": 1,
                "retry_after_segment": 2,
            },
            "relations": [
                {
                    "record_id": "fixture-rel-c",
                    "relation": "predicts",
                    "source_event_id": "vision:visual_cluster_019",
                    "target_event_id": "audio:audio_cluster_045",
                    "delay_lower_ms": 60,
                    "delay_upper_ms": 140,
                    "confidence": 0.88,
                    "evidence_count": 5,
                    "counterexample_count": 0,
                    "prediction_gain": 0.18,
                    "lineage": {
                        "source_ref": "https://example.org/a",
                        "source_hash": "hash-a",
                        "extractor_name": "prediction_gain",
                        "extractor_version": "v1",
                    },
                }
            ],
        },
    ]
    with open(concept_fixture_path, "w", encoding="utf-8") as handle:
        for row in fixture_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report_path = workspace_path(
        "evaluation",
        "test_event_state_cache_integration_mixed.json",
    )
    exit_code = module.main(
        [
            "--manifest-path",
            processed_data_path("autobot", "latent_manifest.jsonl"),
            "--trace-path",
            workspace_path(
                "evaluation",
                "test_event_state_cache_integration_mixed_traces.jsonl",
            ),
            "--round-trip-state-path",
            workspace_path(
                "evaluation",
                "test_event_state_cache_integration_mixed_round_trip.json",
            ),
            "--concept-queue-path",
            workspace_path(
                "evaluation",
                "test_event_state_cache_integration_mixed_queue.json",
            ),
            "--concept-review-report-path",
            workspace_path(
                "evaluation",
                "test_event_state_cache_integration_mixed_review.json",
            ),
            "--concept-fixture-path",
            concept_fixture_path,
            "--report-path",
            report_path,
            "--summary-path",
            workspace_path(
                "evaluation",
                "test_event_state_cache_integration_mixed.txt",
            ),
        ]
    )

    assert exit_code == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["passed"] is True
    assert report["metrics"]["concept_revalidation_case_count"] == 2
    assert report["metrics"]["concept_revalidation_expected_recoverable_case_count"] == 1
    assert report["metrics"]["concept_revalidation_expected_blocked_case_count"] == 1
    assert report["metrics"]["concept_revalidation_admitted_count"] == 1
    assert report["metrics"]["concept_revalidation_blocked_count"] == 1
    assert report["metrics"]["concept_revalidation_source_diversity_blocked_count"] == 1
    assert report["metrics"]["concept_revalidation_recovery_rate"] == 1.0
    assert report["metrics"]["concept_revalidation_blocked_integrity"] == 1.0
    assert report["next_actions"][0]["action"] == "collect_additional_distinct_sources"
