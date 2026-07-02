from sara_engine.ingest import FrequentSequence, make_candidate_relation
from sara_engine.dynamics import stable_self_state_id
from sara_engine.memory.concept_admission import ConceptRevalidationEntry
from sara_engine.memory.concept_revalidation import ConceptRevalidationScheduler


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


def test_revalidation_scheduler_releases_revision_conflict_after_resolution():
    entry = _entry("quarantine_source_revision_conflict")
    relations = [
        _relation(record_id="rel-1", source_ref="episode-1", source_hash="hash-a"),
        _relation(record_id="rel-2", source_ref="episode-2", source_hash="hash-b"),
    ]

    schedule = ConceptRevalidationScheduler().build_schedule(
        [entry],
        relations,
        current_segment=5,
    )

    assert len(schedule.ready_queue) == 1
    assert schedule.ready_queue[0].decision == "ready_revision_conflict_resolved"
    assert schedule.ready_queue[0].priority_score > 0.0
    assert schedule.ready_queue[0].credit_score > 0.0


def test_revalidation_scheduler_blocks_until_cooldown_passes():
    entry = _entry("reject_insufficient_source_diversity", retry_after_segment=8, last_review_segment=7)
    relations = [
        _relation(record_id="rel-1", source_ref="episode-1", source_hash="hash-a"),
        _relation(record_id="rel-2", source_ref="episode-2", source_hash="hash-b"),
    ]

    schedule = ConceptRevalidationScheduler().build_schedule(
        [entry],
        relations,
        current_segment=7,
    )

    assert len(schedule.ready_queue) == 0
    assert schedule.blocked_queue[0].decision == "blocked_cooldown"
    assert schedule.blocked_queue[0].retry_after_segment == 8


def test_revalidation_scheduler_requires_counterexample_pressure_to_drop():
    entry = _entry("quarantine_counterexample_pressure", contradiction_score=0.5, next_action="collect_counterexamples_and_retest")
    noisy_relations = [
        _relation(record_id="rel-1", source_ref="episode-1", source_hash="hash-a", evidence_count=4, counterexample_count=3),
        _relation(record_id="rel-2", source_ref="episode-2", source_hash="hash-b", evidence_count=4, counterexample_count=3),
    ]
    healthy_relations = [
        _relation(record_id="rel-1", source_ref="episode-1", source_hash="hash-a", evidence_count=5, counterexample_count=0),
        _relation(record_id="rel-2", source_ref="episode-2", source_hash="hash-b", evidence_count=5, counterexample_count=0),
    ]

    blocked = ConceptRevalidationScheduler().build_schedule([entry], noisy_relations, current_segment=5)
    ready = ConceptRevalidationScheduler().build_schedule([entry], healthy_relations, current_segment=5)

    assert blocked.blocked_queue[0].decision == "hold_missing_requirements"
    assert ready.ready_queue[0].decision == "ready_counterexample_pressure_reduced"


def test_revalidation_scheduler_respects_attempt_budget():
    entry = _entry("reject_missing_support", attempt_count=3, next_action="rebuild_supporting_relations")

    schedule = ConceptRevalidationScheduler(max_attempts=3).build_schedule(
        [entry],
        [],
        current_segment=10,
    )

    assert len(schedule.ready_queue) == 0
    assert schedule.blocked_queue[0].decision == "blocked_attempt_budget"
    assert schedule.blocked_queue[0].next_action == "manual_review"


def test_revalidation_scheduler_marks_entry_dispatched():
    entry = _entry("reject_insufficient_source_diversity", attempt_count=1, last_review_segment=4, retry_after_segment=5)

    updated = ConceptRevalidationScheduler(min_cooldown_segments=2).mark_dispatched(entry, current_segment=9)

    assert updated.attempt_count == 2
    assert updated.last_review_segment == 9
    assert updated.retry_after_segment == 11


def test_revalidation_scheduler_uses_sequence_support_in_ready_priority():
    entry = _entry("quarantine_source_revision_conflict")
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

    plain = ConceptRevalidationScheduler().build_schedule(
        [entry],
        relations,
        current_segment=5,
    )
    backed = ConceptRevalidationScheduler().build_schedule(
        [entry],
        relations,
        current_segment=5,
        frequent_sequences=sequences,
    )

    assert backed.ready_queue[0].sequence_support_count == 1
    assert backed.ready_queue[0].sequence_support_score > 0.3
    assert backed.ready_queue[0].priority_score > plain.ready_queue[0].priority_score
    assert backed.ready_queue[0].credit_score > 0.0


def test_revalidation_scheduler_uses_self_state_alignment_in_ready_priority():
    entry = _entry("quarantine_source_revision_conflict")
    relations = [
        _relation(record_id="rel-1", source_ref="episode-1", source_hash="hash-a"),
        _relation(record_id="rel-2", source_ref="episode-2", source_hash="hash-b"),
    ]
    self_state_ids = (
        stable_self_state_id("vision:visual_cluster_018"),
        stable_self_state_id("audio:audio_cluster_044"),
    )

    plain = ConceptRevalidationScheduler().build_schedule(
        [entry],
        relations,
        current_segment=5,
    )
    aligned = ConceptRevalidationScheduler().build_schedule(
        [entry],
        relations,
        current_segment=5,
        self_state_ids=self_state_ids,
    )

    assert aligned.ready_queue[0].self_state_alignment_score == 1.0
    assert aligned.ready_queue[0].priority_score > plain.ready_queue[0].priority_score


def test_revalidation_scheduler_surfaces_multimodal_bundle_affinity_in_credit_summary():
    entry = _entry(
        "quarantine_source_revision_conflict",
        concept_key="predicts:bundle:0:123456->audio:audio_cluster_044",
    )
    relations = [
        _relation(
            record_id="rel-1",
            source_ref="bundle::episode-1",
            source_hash="hash-a",
            source_event_id="bundle:0:123456",
            prediction_gain=0.2,
        ),
        _relation(
            record_id="rel-2",
            source_ref="episode-2",
            source_hash="hash-b",
            source_event_id="bundle:0:123456",
            prediction_gain=0.2,
        ),
    ]

    schedule = ConceptRevalidationScheduler().build_schedule(
        [entry],
        relations,
        current_segment=5,
    )

    assert schedule.ready_queue[0].multimodal_bundle_affinity == 1.0


def test_revalidation_scheduler_uses_multimodal_bundle_affinity_in_ready_priority():
    plain_entry = _entry(
        "quarantine_source_revision_conflict",
        concept_key="predicts:vision:visual_cluster_018->audio:audio_cluster_044",
    )
    bundle_entry = _entry(
        "quarantine_source_revision_conflict",
        concept_key="predicts:bundle:0:123456->audio:audio_cluster_044",
    )
    plain_relations = [
        _relation(record_id="plain-1", source_ref="episode-1", source_hash="hash-a", prediction_gain=0.2),
        _relation(record_id="plain-2", source_ref="episode-2", source_hash="hash-b", prediction_gain=0.2),
    ]
    bundle_relations = [
        _relation(
            record_id="bundle-1",
            source_ref="bundle::episode-1",
            source_hash="hash-a",
            source_event_id="bundle:0:123456",
            prediction_gain=0.2,
        ),
        _relation(
            record_id="bundle-2",
            source_ref="episode-2",
            source_hash="hash-b",
            source_event_id="bundle:0:123456",
            prediction_gain=0.2,
        ),
    ]

    plain = ConceptRevalidationScheduler().build_schedule([plain_entry], plain_relations, current_segment=5)
    bundle = ConceptRevalidationScheduler().build_schedule([bundle_entry], bundle_relations, current_segment=5)

    assert bundle.ready_queue[0].priority_score > plain.ready_queue[0].priority_score
