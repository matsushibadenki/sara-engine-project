from sara_engine.ingest import FrequentSequence, make_candidate_relation
from sara_engine.memory.concept_admission import ConceptRevalidationEntry
from sara_engine.memory.concept_review_loop import ConceptReviewLoop


def _relation(
    *,
    record_id: str,
    source_ref: str,
    source_hash: str,
    source_event_id: str = "vision:visual_cluster_018",
    target_event_id: str = "audio:audio_cluster_044",
    evidence_count: int = 5,
    counterexample_count: int = 0,
    prediction_gain: float = 0.18,
):
    return make_candidate_relation(
        {
            "record_id": record_id,
            "relation": "predicts",
            "source_event_id": source_event_id,
            "target_event_id": target_event_id,
            "delay_lower_ms": 60,
            "delay_upper_ms": 140,
            "confidence": 0.88,
            "source_ref": source_ref,
            "source_hash": source_hash,
            "extractor_name": "prediction_gain",
            "extractor_version": "v1",
            "evidence_count": evidence_count,
            "counterexample_count": counterexample_count,
            "prediction_gain": prediction_gain,
        }
    )


def _entry(decision: str, **overrides):
    values = {
        "concept_key": "predicts:vision:visual_cluster_018->audio:audio_cluster_044",
        "decision": decision,
        "supporting_relation_ids": ("predicts:vision:visual_cluster_018->audio:audio_cluster_044",),
        "source_refs": ("episode-1",),
        "source_hashes": ("hash-a",),
        "revision_conflict_count": 1,
        "contradiction_score": 0.2,
        "next_action": "wait",
        "attempt_count": 0,
        "blocked_at_segment": 3,
        "last_review_segment": 3,
        "retry_after_segment": 4,
    }
    values.update(overrides)
    return ConceptRevalidationEntry(**values)


def test_review_loop_rebuilds_ready_concept_into_admission_candidate():
    queue = [_entry("quarantine_source_revision_conflict")]
    relations = [
        _relation(record_id="rel-1", source_ref="episode-1", source_hash="hash-a"),
        _relation(record_id="rel-2", source_ref="episode-2", source_hash="hash-b"),
    ]

    result = ConceptReviewLoop().run(queue, relations, current_segment=6)

    assert len(result.schedule.ready_queue) == 1
    assert len(result.admission_plan.admitted_candidates) == 1
    assert result.admission_plan.admitted_candidates[0].verified is True
    assert result.next_revalidation_queue == ()


def test_review_loop_keeps_blocked_entries_in_next_queue():
    queue = [_entry("reject_insufficient_source_diversity", retry_after_segment=8, last_review_segment=7)]
    relations = [
        _relation(record_id="rel-1", source_ref="episode-1", source_hash="hash-a"),
    ]

    result = ConceptReviewLoop().run(queue, relations, current_segment=7)

    assert len(result.schedule.blocked_queue) == 1
    assert len(result.next_revalidation_queue) == 1
    assert result.next_revalidation_queue[0].decision == "reject_insufficient_source_diversity"
    assert result.next_revalidation_queue[0].retry_after_segment == 8


def test_review_loop_returns_missing_support_to_revalidation():
    queue = [_entry("reject_missing_support", next_action="rebuild_supporting_relations", attempt_count=1)]
    relations = []

    result = ConceptReviewLoop().run(queue, relations, current_segment=9)

    assert len(result.schedule.ready_queue) == 0
    assert len(result.admission_plan.admitted_candidates) == 0
    assert len(result.next_revalidation_queue) == 1
    assert result.next_revalidation_queue[0].decision == "reject_missing_support"


def test_review_loop_carries_forward_attempt_budget_block():
    queue = [_entry("reject_missing_support", attempt_count=3, next_action="rebuild_supporting_relations")]

    result = ConceptReviewLoop().run(queue, [], current_segment=10)

    assert len(result.schedule.blocked_queue) == 1
    assert result.schedule.blocked_queue[0].decision == "blocked_attempt_budget"
    assert result.next_revalidation_queue[0].decision == "blocked_attempt_budget"
    assert result.next_revalidation_queue[0].next_action == "manual_review"


def test_review_loop_can_rebuild_ready_concept_with_sequence_backing():
    queue = [_entry("quarantine_source_revision_conflict")]
    relations = [
        _relation(record_id="rel-1", source_ref="episode-1", source_hash="hash-a"),
        _relation(record_id="rel-2", source_ref="episode-2", source_hash="hash-b"),
    ]
    sequences = [
        FrequentSequence(
            sequence_key="visual_cluster_018 -> audio_cluster_044",
            labels=("visual_cluster_018", "audio_cluster_044"),
            support_episode_count=2,
            occurrence_count=2,
            source_count=2,
            mean_span_ms=50.0,
            parent_episode_ids=("episode-1", "episode-2"),
        )
    ]

    result = ConceptReviewLoop().run(
        queue,
        relations,
        current_segment=6,
        frequent_sequences=sequences,
    )

    assert len(result.admission_plan.admitted_candidates) == 1
    assert result.schedule.ready_queue[0].sequence_support_score > 0.3
