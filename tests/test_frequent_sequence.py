from sara_engine.ingest import EpisodeSegmenter, FrequentSequenceMiner, make_candidate_event, make_observed_event


def _observed(record_id: str, time_ms: int, label: str, *, source_ref: str, source_hash: str):
    return make_observed_event(
        {
            "record_id": record_id,
            "modality": "audio",
            "local_time_ms": time_ms,
            "label": label,
            "confidence": 0.9,
            "lineage": {
                "source_ref": source_ref,
                "source_hash": source_hash,
                "extractor_name": "temporal_eventizer",
                "extractor_version": "v1",
            },
        }
    )


def _candidate(record_id: str, time_ms: int, label: str, *, source_ref: str, source_hash: str):
    return make_candidate_event(
        {
            "record_id": record_id,
            "modality": "vision",
            "label": label,
            "local_time_ms": time_ms,
            "confidence": 0.81,
            "source_ref": source_ref,
            "source_hash": source_hash,
            "extractor_name": "candidate_proposals",
            "extractor_version": "v1",
            "evidence_count": 3,
        }
    )


def test_frequent_sequence_miner_finds_repeated_patterns_across_episodes():
    observed = [
        _observed("a1", 100, "audio_onset", source_ref="session-1", source_hash="hash-1"),
        _observed("a2", 140, "subtitle_boundary", source_ref="session-1", source_hash="hash-1"),
        _observed("b1", 100, "audio_onset", source_ref="session-2", source_hash="hash-2"),
        _observed("b2", 130, "subtitle_boundary", source_ref="session-2", source_hash="hash-2"),
    ]
    segmenter = EpisodeSegmenter(max_gap_ms=80, max_events_per_episode=4)
    episodes = segmenter.segment(observed)

    sequences = FrequentSequenceMiner(min_support_episodes=2, max_pattern_length=2, max_span_ms=100).mine(
        episodes,
        observed,
    )

    assert len(sequences) == 1
    sequence = sequences[0]
    assert sequence.labels == ("audio_onset", "subtitle_boundary")
    assert sequence.support_episode_count == 2
    assert sequence.occurrence_count == 2
    assert sequence.source_count == 2


def test_frequent_sequence_miner_can_include_candidate_events_without_changing_boundary():
    observed = [
        _observed("a1", 100, "audio_onset", source_ref="session-1", source_hash="hash-1"),
        _observed("a2", 150, "subtitle_boundary", source_ref="session-1", source_hash="hash-1"),
        _observed("b1", 100, "audio_onset", source_ref="session-2", source_hash="hash-2"),
        _observed("b2", 150, "subtitle_boundary", source_ref="session-2", source_hash="hash-2"),
    ]
    candidates = [
        _candidate("ca1", 180, "visual_cluster_018", source_ref="session-1", source_hash="hash-1"),
        _candidate("cb1", 180, "visual_cluster_018", source_ref="session-2", source_hash="hash-2"),
    ]
    segmenter = EpisodeSegmenter(max_gap_ms=120, max_events_per_episode=6)
    episodes = segmenter.segment(observed, candidate_events=candidates)

    sequences = FrequentSequenceMiner(min_support_episodes=2, max_pattern_length=3, max_span_ms=120).mine(
        episodes,
        observed,
        candidate_events=candidates,
    )

    labels = [sequence.labels for sequence in sequences]
    assert ("audio_onset", "subtitle_boundary", "visual_cluster_018") in labels


def test_frequent_sequence_miner_rejects_sequences_without_enough_episode_support():
    observed = [
        _observed("a1", 100, "audio_onset", source_ref="session-1", source_hash="hash-1"),
        _observed("a2", 140, "subtitle_boundary", source_ref="session-1", source_hash="hash-1"),
        _observed("a3", 300, "turn_pause", source_ref="session-1", source_hash="hash-1"),
    ]
    episodes = EpisodeSegmenter(max_gap_ms=60, max_events_per_episode=4).segment(observed)

    miner = FrequentSequenceMiner(min_support_episodes=2, max_pattern_length=2, max_span_ms=100)
    sequences = miner.mine(episodes, observed)

    assert sequences == []
    assert miner.last_trace.accepted_sequences == 0
