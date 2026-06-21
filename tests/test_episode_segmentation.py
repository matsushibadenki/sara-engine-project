from sara_engine.ingest import EpisodeSegmenter, make_candidate_event, make_observed_event


def _observed(record_id: str, time_ms: int, *, source_ref: str = "session-1", source_hash: str = "hash-1", modality: str = "audio"):
    return make_observed_event(
        {
            "record_id": record_id,
            "modality": modality,
            "local_time_ms": time_ms,
            "label": f"{modality}_change",
            "confidence": 0.9,
            "lineage": {
                "source_ref": source_ref,
                "source_hash": source_hash,
                "extractor_name": "temporal_eventizer",
                "extractor_version": "v1",
            },
        }
    )


def _candidate(record_id: str, time_ms: int, *, source_ref: str = "session-1", source_hash: str = "hash-1", modality: str = "vision", label: str = "visual_cluster_018"):
    return make_candidate_event(
        {
            "record_id": record_id,
            "modality": modality,
            "label": label,
            "local_time_ms": time_ms,
            "confidence": 0.82,
            "source_ref": source_ref,
            "source_hash": source_hash,
            "extractor_name": "candidate_proposals",
            "extractor_version": "v1",
            "evidence_count": 3,
            "prediction_gain": 0.1,
        }
    )


def test_episode_segmenter_groups_observed_and_candidate_events_into_shared_episode():
    observed = [_observed("obs-1", 100), _observed("obs-2", 180, modality="text")]
    candidates = [_candidate("cand-1", 210)]

    episodes = EpisodeSegmenter(max_gap_ms=80, max_events_per_episode=4).segment(
        observed,
        candidate_events=candidates,
    )

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.observed_event_ids == ("obs-1", "obs-2")
    assert episode.candidate_event_ids == ("cand-1",)
    assert episode.event_count == 3
    assert set(episode.modalities) == {"audio", "text", "vision"}


def test_episode_segmenter_splits_on_large_time_gap():
    observed = [
        _observed("obs-1", 100),
        _observed("obs-2", 140),
        _observed("obs-3", 500),
    ]
    segmenter = EpisodeSegmenter(max_gap_ms=120, max_events_per_episode=6)

    episodes = segmenter.segment(observed)

    assert len(episodes) == 2
    assert episodes[0].parent_ids == ("obs-1", "obs-2")
    assert episodes[1].parent_ids == ("obs-3",)
    assert segmenter.last_trace.gap_split_count == 1


def test_episode_segmenter_splits_on_source_boundary_and_capacity():
    observed = [
        _observed("obs-1", 100),
        _observed("obs-2", 120),
        _observed("obs-3", 140),
        _observed("obs-4", 200, source_ref="session-2", source_hash="hash-2"),
    ]
    segmenter = EpisodeSegmenter(max_gap_ms=100, max_events_per_episode=2)

    episodes = segmenter.segment(observed)

    assert len(episodes) == 3
    assert episodes[0].parent_ids == ("obs-1", "obs-2")
    assert episodes[1].parent_ids == ("obs-3",)
    assert episodes[2].parent_ids == ("obs-4",)
    assert segmenter.last_trace.overflow_split_count == 1
    assert segmenter.last_trace.source_split_count == 1
