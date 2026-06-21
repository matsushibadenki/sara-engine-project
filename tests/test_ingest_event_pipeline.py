from sara_engine.ingest import (
    PredictionGainEstimator,
    ProposalVerifier,
    ScalarChangeDetector,
    SynchronyDetector,
    TemporalEventizer,
)


def test_change_detector_emits_refractory_bounded_changes():
    detector = ScalarChangeDetector(threshold=0.2, refractory_ms=100, baseline_smoothing=0.5)
    samples = [
        {"time_ms": 0, "value": 0.0},
        {"time_ms": 50, "value": 0.1},
        {"time_ms": 120, "value": 0.5},
        {"time_ms": 180, "value": 0.6},
        {"time_ms": 260, "value": 0.0},
    ]

    changes = detector.detect(samples, stream_id="audio-1", modality="audio")

    assert len(changes) == 2
    assert changes[0].time_ms == 120
    assert changes[1].time_ms == 260


def test_temporal_eventizer_groups_nearby_changes_into_observed_events():
    detector = ScalarChangeDetector(threshold=0.2, refractory_ms=20, baseline_smoothing=0.2)
    samples = [
        {"time_ms": 0, "value": 0.0},
        {"time_ms": 50, "value": 0.3},
        {"time_ms": 90, "value": 0.55},
        {"time_ms": 300, "value": 0.0},
    ]
    changes = detector.detect(samples, stream_id="vision-1", modality="vision")

    eventizer = TemporalEventizer(merge_window_ms=60)
    events = eventizer.eventize(changes, source_ref="session-a", source_hash="hash-a")

    assert len(events) == 2
    assert events[0].record_type == "observed_event"
    assert events[0].label == "vision_change"
    assert eventizer.last_trace.emitted_count == 2


def test_prediction_gain_pipeline_proposes_and_verifies_relation():
    detector = ScalarChangeDetector(threshold=0.2, refractory_ms=10, baseline_smoothing=0.1)
    eventizer = TemporalEventizer(merge_window_ms=10)

    audio_changes = detector.detect(
        [
            {"time_ms": 0, "value": 0.0},
            {"time_ms": 100, "value": 0.6},
            {"time_ms": 200, "value": 0.0},
            {"time_ms": 400, "value": 0.7},
            {"time_ms": 500, "value": 0.0},
        ],
        stream_id="audio-1",
        modality="audio",
    )
    text_changes = detector.detect(
        [
            {"time_ms": 0, "value": 0.0},
            {"time_ms": 140, "value": 0.8},
            {"time_ms": 240, "value": 0.0},
            {"time_ms": 440, "value": 0.9},
            {"time_ms": 540, "value": 0.0},
        ],
        stream_id="text-1",
        modality="text",
    )

    events = eventizer.eventize(audio_changes, source_ref="session-b", source_hash="hash-b")
    events += eventizer.eventize(text_changes, source_ref="session-b", source_hash="hash-b")
    estimator = PredictionGainEstimator(min_support=2, min_gain=0.1, max_delay_ms=100)
    relations = estimator.propose_relations(events)

    assert len(relations) >= 1
    verifier = ProposalVerifier(min_confidence=0.5, min_evidence_count=2, min_prediction_gain=0.1)
    result = verifier.verify_relation(relations[0])

    assert result.accepted is True
    assert result.promoted_record is not None
    assert result.promoted_record["record_type"] == "verified_relation"


def test_synchrony_detector_links_nearby_cross_modal_events_only():
    detector = ScalarChangeDetector(threshold=0.2, refractory_ms=10, baseline_smoothing=0.1)
    eventizer = TemporalEventizer(merge_window_ms=10)

    audio_changes = detector.detect(
        [
            {"time_ms": 0, "value": 0.0},
            {"time_ms": 100, "value": 0.6},
            {"time_ms": 200, "value": 0.0},
        ],
        stream_id="audio-1",
        modality="audio",
    )
    text_changes = detector.detect(
        [
            {"time_ms": 0, "value": 0.0},
            {"time_ms": 135, "value": 0.8},
            {"time_ms": 240, "value": 0.0},
        ],
        stream_id="text-1",
        modality="text",
    )
    more_audio_changes = detector.detect(
        [
            {"time_ms": 0, "value": 0.0},
            {"time_ms": 145, "value": 0.7},
            {"time_ms": 260, "value": 0.0},
        ],
        stream_id="audio-2",
        modality="audio",
    )

    events = eventizer.eventize(audio_changes, source_ref="session-c", source_hash="hash-c")
    events += eventizer.eventize(text_changes, source_ref="session-c", source_hash="hash-c")
    events += eventizer.eventize(more_audio_changes, source_ref="session-c", source_hash="hash-c")

    synchrony = SynchronyDetector(window_ms=50, cross_modal_only=True)
    relations = synchrony.propose_relations(events)

    assert len(relations) >= 1
    assert all(item.relation == "synchronized_with" for item in relations)
    assert all(
        next(event for event in events if event.record_id == item.source_event_id).modality
        != next(event for event in events if event.record_id == item.target_event_id).modality
        for item in relations
    )
    assert synchrony.last_trace.accepted_pairs == len(relations)
