from sara_engine.ingest import FrequentSequence, make_candidate_relation, summarize_sequence_relation_support


def _relation():
    return make_candidate_relation(
        {
            "record_id": "rel-1",
            "relation": "predicts",
            "source_event_id": "vision:visual_cluster_018",
            "target_event_id": "audio:audio_cluster_044",
            "delay_lower_ms": 60,
            "delay_upper_ms": 140,
            "confidence": 0.88,
            "source_ref": "episode-1",
            "source_hash": "hash-a",
            "extractor_name": "prediction_gain",
            "extractor_version": "v1",
            "evidence_count": 5,
            "counterexample_count": 0,
            "prediction_gain": 0.18,
        }
    )


def test_sequence_relation_support_scores_ordered_sequence_matches():
    relation = _relation()
    sequences = [
        FrequentSequence(
            sequence_key="visual_cluster_018 -> audio_cluster_044",
            labels=("visual_cluster_018", "audio_cluster_044"),
            support_episode_count=2,
            occurrence_count=2,
            source_count=2,
            mean_span_ms=50.0,
            parent_episode_ids=("episode-a", "episode-b"),
        )
    ]

    support = summarize_sequence_relation_support([relation], sequences)[
        "predicts:vision:visual_cluster_018->audio:audio_cluster_044"
    ]

    assert support.supporting_sequence_count == 1
    assert support.ordered_match_count == 1
    assert support.supporting_episode_count == 2
    assert support.support_score > 0.4


def test_sequence_relation_support_returns_zero_when_no_sequence_matches():
    relation = _relation()
    sequences = [
        FrequentSequence(
            sequence_key="other_a -> other_b",
            labels=("other_a", "other_b"),
            support_episode_count=2,
            occurrence_count=2,
            source_count=2,
            mean_span_ms=50.0,
            parent_episode_ids=("episode-a", "episode-b"),
        )
    ]

    support = summarize_sequence_relation_support([relation], sequences)[
        "predicts:vision:visual_cluster_018->audio:audio_cluster_044"
    ]

    assert support.supporting_sequence_count == 0
    assert support.support_score == 0.0
