import json
import os

from sara_engine.ingest import make_candidate_relation
from sara_engine.memory.concept_admission import ConceptRevalidationEntry
from sara_engine.memory.concept_queue_store import (
    load_revalidation_queue,
    run_persisted_concept_review_cycle,
    save_revalidation_queue,
)
from sara_engine.utils.project_paths import workspace_path


def _entry(**overrides):
    values = {
        "concept_key": "predicts:vision:visual_cluster_018->audio:audio_cluster_044",
        "decision": "quarantine_source_revision_conflict",
        "supporting_relation_ids": ("predicts:vision:visual_cluster_018->audio:audio_cluster_044",),
        "source_refs": ("episode-1",),
        "source_hashes": ("hash-a",),
        "revision_conflict_count": 1,
        "contradiction_score": 0.2,
        "next_action": "wait_for_source_revision_resolution",
        "attempt_count": 0,
        "blocked_at_segment": 3,
        "last_review_segment": 3,
        "retry_after_segment": 4,
    }
    values.update(overrides)
    return ConceptRevalidationEntry(**values)


def _relation(
    *,
    record_id: str,
    source_ref: str,
    source_hash: str,
):
    return make_candidate_relation(
        {
            "record_id": record_id,
            "relation": "predicts",
            "source_event_id": "vision:visual_cluster_018",
            "target_event_id": "audio:audio_cluster_044",
            "delay_lower_ms": 60,
            "delay_upper_ms": 140,
            "confidence": 0.88,
            "source_ref": source_ref,
            "source_hash": source_hash,
            "extractor_name": "prediction_gain",
            "extractor_version": "v1",
            "evidence_count": 5,
            "counterexample_count": 0,
            "prediction_gain": 0.18,
        }
    )


def test_concept_queue_store_round_trips_entries_in_workspace():
    queue_path = workspace_path("memory", f"test_revalidation_queue_{os.getpid()}.json")
    original = (_entry(), _entry(concept_key="predicts:vision:cluster_x->audio:cluster_y", source_refs=("episode-2",)))

    saved_path = save_revalidation_queue(original, queue_path)
    restored = load_revalidation_queue(saved_path)

    assert saved_path == queue_path
    assert restored == original


def test_concept_queue_store_rejects_unknown_schema():
    queue_path = workspace_path("memory", f"test_revalidation_bad_schema_{os.getpid()}.json")
    with open(queue_path, "w", encoding="utf-8") as handle:
        json.dump({"schema": "unknown", "entries": []}, handle)

    try:
        load_revalidation_queue(queue_path)
    except ValueError as exc:
        assert "schema" in str(exc)
    else:
        raise AssertionError("unknown queue schema must be rejected")


def test_run_persisted_concept_review_cycle_updates_queue_and_report():
    queue_path = workspace_path("memory", f"test_revalidation_cycle_{os.getpid()}.json")
    report_path = workspace_path("memory", f"test_revalidation_cycle_report_{os.getpid()}.json")
    save_revalidation_queue((_entry(),), queue_path)
    relations = [
        _relation(record_id="rel-1", source_ref="episode-1", source_hash="hash-a"),
        _relation(record_id="rel-2", source_ref="episode-2", source_hash="hash-b"),
    ]

    result = run_persisted_concept_review_cycle(
        relations,
        current_segment=6,
        queue_path=queue_path,
        report_path=report_path,
    )

    assert len(result.admission_plan.admitted_candidates) == 1
    updated_queue = load_revalidation_queue(queue_path)
    assert updated_queue == ()
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["ready_count"] == 1
    assert report["admitted_candidate_count"] == 1
    assert report["revalidation_queue_count"] == 0
    assert report["ready_mean_credit_score"] > 0.0
    assert report["blocked_mean_credit_score"] == 0.0


def test_review_report_surfaces_manual_review_candidates():
    queue_path = workspace_path("memory", f"test_revalidation_manual_{os.getpid()}.json")
    report_path = workspace_path("memory", f"test_revalidation_manual_report_{os.getpid()}.json")
    save_revalidation_queue((_entry(attempt_count=3),), queue_path)

    result = run_persisted_concept_review_cycle(
        [],
        current_segment=10,
        queue_path=queue_path,
        report_path=report_path,
    )

    assert len(result.schedule.blocked_queue) == 1
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["manual_review_candidates"][0]["next_action"] == "manual_review"
