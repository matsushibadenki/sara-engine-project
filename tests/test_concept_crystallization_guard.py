from sara_engine.ingest import (
    ConceptCrystallizationGuard,
    FrequentSequence,
    RelationStabilityAssessor,
    make_candidate_relation,
)


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


def _concepts(relations):
    return RelationStabilityAssessor(
        min_contexts=2,
        min_stability_score=0.35,
        min_mean_prediction_gain=0.05,
    ).crystallization_candidates(relations)


def test_guard_accepts_stable_candidate_with_multi_source_support():
    relations = [
        _relation(record_id="rel-1", source_ref="episode-1", source_hash="hash-a"),
        _relation(record_id="rel-2", source_ref="episode-2", source_hash="hash-b"),
        _relation(record_id="rel-3", source_ref="episode-3", source_hash="hash-c"),
    ]
    candidates = _concepts(relations)

    audits = ConceptCrystallizationGuard().audit_candidates(relations, candidates)

    assert len(audits) == 1
    assert audits[0].accepted is True
    assert audits[0].decision == "accept_concept_candidate"


def test_guard_quarantines_revision_conflicts_within_same_source_ref():
    relations = [
        _relation(record_id="rel-1", source_ref="episode-1", source_hash="hash-a"),
        _relation(record_id="rel-2", source_ref="episode-1", source_hash="hash-b"),
        _relation(record_id="rel-3", source_ref="episode-2", source_hash="hash-c"),
    ]
    candidates = _concepts(relations)

    audits = ConceptCrystallizationGuard().audit_candidates(relations, candidates)

    assert len(audits) == 1
    assert audits[0].accepted is False
    assert audits[0].decision == "quarantine_source_revision_conflict"
    assert audits[0].revision_conflict_count == 1
    assert "episode-1" in audits[0].trace["revision_conflict_refs"]


def test_guard_quarantines_counterexample_pressure():
    relations = [
        _relation(
            record_id="rel-1",
            source_ref="episode-1",
            source_hash="hash-a",
            evidence_count=4,
            counterexample_count=3,
        ),
        _relation(
            record_id="rel-2",
            source_ref="episode-2",
            source_hash="hash-b",
            evidence_count=4,
            counterexample_count=3,
        ),
    ]
    candidates = _concepts(relations)

    audits = ConceptCrystallizationGuard(max_counterexample_rate=0.25).audit_candidates(relations, candidates)

    assert len(audits) == 1
    assert audits[0].accepted is False
    assert audits[0].decision == "quarantine_counterexample_pressure"
    assert audits[0].contradiction_score > 0.40


def test_guard_filters_out_unaccepted_candidates():
    accepted_relations = [
        _relation(record_id="ok-1", source_ref="episode-1", source_hash="hash-a"),
        _relation(record_id="ok-2", source_ref="episode-2", source_hash="hash-b"),
    ]
    conflicted_relations = [
        _relation(
            record_id="bad-1",
            source_ref="episode-10",
            source_hash="hash-x",
            source_event_id="vision:visual_cluster_099",
            target_event_id="audio:audio_cluster_199",
        ),
        _relation(
            record_id="bad-2",
            source_ref="episode-10",
            source_hash="hash-y",
            source_event_id="vision:visual_cluster_099",
            target_event_id="audio:audio_cluster_199",
        ),
        _relation(
            record_id="bad-3",
            source_ref="episode-11",
            source_hash="hash-z",
            source_event_id="vision:visual_cluster_099",
            target_event_id="audio:audio_cluster_199",
        ),
    ]
    all_relations = accepted_relations + conflicted_relations
    candidates = _concepts(accepted_relations) + _concepts(conflicted_relations)

    accepted = ConceptCrystallizationGuard().accepted_candidates(all_relations, candidates)

    assert len(accepted) == 1
    assert accepted[0].concept_key == "predicts:vision:visual_cluster_018->audio:audio_cluster_044"


def test_guard_can_require_sequence_backing_for_concept_admission():
    relations = [
        _relation(record_id="rel-1", source_ref="episode-1", source_hash="hash-a"),
        _relation(record_id="rel-2", source_ref="episode-2", source_hash="hash-b"),
    ]
    candidates = _concepts(relations)
    guard = ConceptCrystallizationGuard(min_sequence_support_score=0.3)

    rejected = guard.audit_candidates(relations, candidates)
    assert rejected[0].accepted is False
    assert rejected[0].decision == "reject_weak_sequence_support"

    sequences = [
        FrequentSequence(
            sequence_key="visual_cluster_018 -> audio_cluster_044",
            labels=("visual_cluster_018", "audio_cluster_044"),
            support_episode_count=2,
            occurrence_count=2,
            source_count=2,
            mean_span_ms=60.0,
            parent_episode_ids=("episode-1", "episode-2"),
        )
    ]
    accepted = guard.audit_candidates(relations, candidates, frequent_sequences=sequences)
    assert accepted[0].accepted is True
    assert accepted[0].sequence_support_score > 0.3
