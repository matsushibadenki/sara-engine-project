from sara_engine.ingest import (
    ConceptCrystallizationGuard,
    FrequentSequence,
    RelationStabilityAssessor,
    make_candidate_relation,
)
from sara_engine.memory.concept_admission import ConceptAdmissionPlanner
from sara_engine.memory.event_state_cache import VerifiedHierarchicalEventStateCache


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


def test_concept_admission_planner_promotes_audited_candidate_into_event_state():
    relations = [
        _relation(record_id="rel-1", source_ref="episode-1", source_hash="hash-a"),
        _relation(record_id="rel-2", source_ref="episode-2", source_hash="hash-b"),
        _relation(record_id="rel-3", source_ref="episode-3", source_hash="hash-c"),
    ]
    plan = ConceptAdmissionPlanner().build_plan(relations, _concepts(relations), time_segment=7)

    assert len(plan.admitted_candidates) == 1
    candidate = plan.admitted_candidates[0]
    assert candidate.verified is True
    assert candidate.observed is True
    assert candidate.source_backed is True
    assert candidate.own_latent_id == "predicts:vision:visual_cluster_018->audio:audio_cluster_044"
    assert candidate.source_revision.startswith("concept-rev:")
    assert candidate.sequence_support_score == 0.0
    assert candidate.credit_score > 0.0
    assert candidate.credit_confidence > 0.0
    assert len(plan.revalidation_queue) == 0


def test_concept_admission_planner_routes_conflicted_candidate_to_revalidation():
    relations = [
        _relation(record_id="rel-1", source_ref="episode-1", source_hash="hash-a"),
        _relation(record_id="rel-2", source_ref="episode-1", source_hash="hash-b"),
        _relation(record_id="rel-3", source_ref="episode-2", source_hash="hash-c"),
    ]
    plan = ConceptAdmissionPlanner().build_plan(relations, _concepts(relations), time_segment=7)

    assert len(plan.admitted_candidates) == 0
    assert len(plan.revalidation_queue) == 1
    queued = plan.revalidation_queue[0]
    assert queued.decision == "quarantine_source_revision_conflict"
    assert queued.next_action == "wait_for_source_revision_resolution"


def test_concept_admission_candidates_can_enter_verified_event_state_cache():
    relations = [
        _relation(record_id="rel-1", source_ref="episode-1", source_hash="hash-a"),
        _relation(record_id="rel-2", source_ref="episode-2", source_hash="hash-b"),
    ]
    plan = ConceptAdmissionPlanner().build_plan(relations, _concepts(relations), time_segment=5)
    cache = VerifiedHierarchicalEventStateCache(retention_profile="logarithmic")

    result = cache.admit(plan.admitted_candidates[0])

    assert result.accepted is True
    retrieval = cache.retrieve(plan.admitted_candidates[0].signature, own_latent_id=plan.admitted_candidates[0].own_latent_id)
    assert retrieval.abstained is False
    assert retrieval.matches[0]["entry_id"] == plan.admitted_candidates[0].entry_id


def test_concept_admission_uses_custom_guard_thresholds():
    relations = [
        _relation(record_id="rel-1", source_ref="episode-1", source_hash="hash-a"),
        _relation(record_id="rel-2", source_ref="episode-2", source_hash="hash-b"),
    ]
    planner = ConceptAdmissionPlanner(
        guard=ConceptCrystallizationGuard(min_distinct_source_refs=3)
    )

    plan = planner.build_plan(relations, _concepts(relations), time_segment=9)

    assert len(plan.admitted_candidates) == 0
    assert len(plan.revalidation_queue) == 1
    assert plan.revalidation_queue[0].decision == "reject_insufficient_source_diversity"


def test_concept_admission_can_use_sequence_backing():
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
            mean_span_ms=55.0,
            parent_episode_ids=("episode-1", "episode-2"),
        )
    ]
    planner = ConceptAdmissionPlanner(
        guard=ConceptCrystallizationGuard(min_sequence_support_score=0.3)
    )

    plan = planner.build_plan(
        relations,
        _concepts(relations),
        time_segment=11,
        frequent_sequences=sequences,
    )

    assert len(plan.admitted_candidates) == 1
    assert plan.admitted_candidates[0].sequence_support_score > 0.3
    assert plan.admitted_candidates[0].sequence_support_count == 1
    assert len(plan.revalidation_queue) == 0


def test_concept_admission_prioritizes_bundle_supported_candidate_when_credit_is_higher():
    relations = [
        _relation(
            record_id="bundle-1",
            source_ref="bundle::episode-1",
            source_hash="hash-a",
            source_event_id="bundle:0:123456",
            prediction_gain=0.22,
        ),
        _relation(
            record_id="bundle-2",
            source_ref="episode-2",
            source_hash="hash-b",
            source_event_id="bundle:0:123456",
            prediction_gain=0.22,
        ),
        _relation(
            record_id="plain-1",
            source_ref="episode-3",
            source_hash="hash-c",
            source_event_id="vision:visual_cluster_018",
            target_event_id="audio:audio_cluster_099",
            prediction_gain=0.18,
        ),
        _relation(
            record_id="plain-2",
            source_ref="episode-4",
            source_hash="hash-d",
            source_event_id="vision:visual_cluster_018",
            target_event_id="audio:audio_cluster_099",
            prediction_gain=0.18,
        ),
    ]

    plan = ConceptAdmissionPlanner().build_plan(relations, _concepts(relations), time_segment=12)

    assert len(plan.admitted_candidates) == 2
    assert "bundle:" in plan.admitted_candidates[0].own_latent_id
    assert plan.admitted_candidates[0].credit_score >= plan.admitted_candidates[1].credit_score
