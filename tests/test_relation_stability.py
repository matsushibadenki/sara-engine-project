from sara_engine.ingest import (
    RelationStabilityAssessor,
    make_candidate_relation,
)


def _relation(
    *,
    record_id: str,
    source_ref: str,
    source_hash: str,
    evidence_count: int,
    counterexample_count: int,
    prediction_gain: float,
):
    return make_candidate_relation(
        {
            "record_id": record_id,
            "relation": "predicts",
            "source_event_id": "vision:visual_cluster_018",
            "target_event_id": "audio:audio_cluster_044",
            "delay_lower_ms": 80,
            "delay_upper_ms": 180,
            "confidence": 0.85,
            "source_ref": source_ref,
            "source_hash": source_hash,
            "extractor_name": "prediction_gain",
            "extractor_version": "v1",
            "evidence_count": evidence_count,
            "counterexample_count": counterexample_count,
            "prediction_gain": prediction_gain,
        }
    )


def test_relation_stability_summary_aggregates_contexts_sources_and_gain():
    assessor = RelationStabilityAssessor(min_contexts=2, min_stability_score=0.35, min_mean_prediction_gain=0.05)
    relations = [
        _relation(
            record_id="rel-1",
            source_ref="episode-1",
            source_hash="source-a",
            evidence_count=5,
            counterexample_count=1,
            prediction_gain=0.20,
        ),
        _relation(
            record_id="rel-2",
            source_ref="episode-2",
            source_hash="source-b",
            evidence_count=4,
            counterexample_count=0,
            prediction_gain=0.16,
        ),
    ]

    summaries = assessor.summarize(relations)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.relation_key == "predicts:vision:visual_cluster_018->audio:audio_cluster_044"
    assert summary.context_count == 2
    assert summary.source_count == 2
    assert summary.evidence_count == 9
    assert summary.counterexample_count == 1
    assert summary.mean_prediction_gain > 0.17
    assert summary.min_prediction_gain == 0.16
    assert summary.stability_score > 0.60


def test_relation_stability_requires_multi_context_support_for_crystallization():
    assessor = RelationStabilityAssessor(min_contexts=2, min_stability_score=0.35, min_mean_prediction_gain=0.05)
    single_context = [
        _relation(
            record_id="rel-single",
            source_ref="episode-1",
            source_hash="source-a",
            evidence_count=8,
            counterexample_count=0,
            prediction_gain=0.24,
        )
    ]

    candidates = assessor.crystallization_candidates(single_context)

    assert candidates == []


def test_relation_stability_emits_concept_candidate_for_stable_cross_context_relation():
    assessor = RelationStabilityAssessor(min_contexts=2, min_stability_score=0.35, min_mean_prediction_gain=0.05)
    relations = [
        _relation(
            record_id="rel-1",
            source_ref="episode-1",
            source_hash="source-a",
            evidence_count=6,
            counterexample_count=1,
            prediction_gain=0.19,
        ),
        _relation(
            record_id="rel-2",
            source_ref="episode-2",
            source_hash="source-b",
            evidence_count=5,
            counterexample_count=0,
            prediction_gain=0.22,
        ),
        _relation(
            record_id="rel-3",
            source_ref="episode-3",
            source_hash="source-c",
            evidence_count=4,
            counterexample_count=1,
            prediction_gain=0.17,
        ),
    ]

    candidates = assessor.crystallization_candidates(relations)

    assert len(candidates) == 1
    candidate = candidates[0]
    payload = candidate.to_dict()
    assert payload["record_type"] == "concept_crystal_candidate"
    assert payload["concept_key"] == "predicts:vision:visual_cluster_018->audio:audio_cluster_044"
    assert payload["evidence_count"] == 15
    assert payload["counterexample_count"] == 2
    assert payload["prediction_gain"] > 0.19
    assert payload["lineage"]["extractor_name"] == "relation_stability"


def test_relation_stability_penalizes_counterexamples():
    assessor = RelationStabilityAssessor(min_contexts=2, min_stability_score=0.35, min_mean_prediction_gain=0.05)
    stable_relations = [
        _relation(
            record_id="stable-1",
            source_ref="episode-1",
            source_hash="source-a",
            evidence_count=6,
            counterexample_count=0,
            prediction_gain=0.18,
        ),
        _relation(
            record_id="stable-2",
            source_ref="episode-2",
            source_hash="source-b",
            evidence_count=5,
            counterexample_count=0,
            prediction_gain=0.18,
        ),
    ]
    noisy_relations = [
        _relation(
            record_id="noisy-1",
            source_ref="episode-1",
            source_hash="source-a",
            evidence_count=6,
            counterexample_count=5,
            prediction_gain=0.18,
        ),
        _relation(
            record_id="noisy-2",
            source_ref="episode-2",
            source_hash="source-b",
            evidence_count=5,
            counterexample_count=4,
            prediction_gain=0.18,
        ),
    ]

    stable_score = assessor.summarize(stable_relations)[0].stability_score
    noisy_score = assessor.summarize(noisy_relations)[0].stability_score

    assert stable_score > noisy_score

