from sara_engine.multimodal.synesthetic_binding import (
    AudioEventAdapter,
    LanguageEventAdapter,
    SparseEventBundle,
    SparsePluggableCorticalColumn,
    SparseSynestheticLinker,
    SparseTemporalBinder,
    SparseThalamicGate,
    TactileEventAdapter,
    VisionEventAdapter,
)


def _events():
    binder = SparseTemporalBinder(window_ms=32.0)
    return [
        binder.normalize_event(
            modality="language",
            timestamp_ms=2.0,
            source_id="language-hard",
            sparse_signature=[1, 2],
            confidence=0.9,
            label="hard",
        ),
        binder.normalize_event(
            modality="vision",
            timestamp_ms=10.0,
            source_id="vision-hard",
            sparse_signature=[11, 12],
            confidence=0.9,
            label="hard",
        ),
        binder.normalize_event(
            modality="audio",
            timestamp_ms=15.0,
            source_id="audio-hard",
            sparse_signature=[21, 22],
            confidence=0.9,
            label="hard",
        ),
        binder.normalize_event(
            modality="tactile",
            timestamp_ms=20.0,
            source_id="tactile-hard",
            sparse_signature=[31, 32],
            confidence=0.9,
            label="hard",
        ),
    ]


def test_temporal_binder_places_modalities_in_same_chunk():
    events = _events()
    buckets = SparseTemporalBinder(window_ms=32.0).bind(events)

    assert list(buckets) == [0]
    assert {event.modality for event in buckets[0]} == {"language", "vision", "audio", "tactile"}
    assert all(event.event_id for event in buckets[0])


def test_temporal_binder_builds_auditable_event_bundles_without_payload_collapse():
    events = _events()
    bundles = SparseTemporalBinder(window_ms=32.0).bundle_events(events)

    assert len(bundles) == 1
    bundle = bundles[0]
    assert isinstance(bundle, SparseEventBundle)
    assert bundle.audit is not None
    assert bundle.audit.admitted is True
    assert bundle.audit.payload_separable is True
    assert set(bundle.modality_ids) == {"language", "vision", "audio", "tactile"}
    assert len(bundle.child_records) == 4
    assert all(record.event_id for record in bundle.child_records)


def test_pluggable_cortical_column_uses_same_learning_rule_for_modalities():
    column = SparsePluggableCorticalColumn(activation_threshold=0.8)
    results = [column.process(event) for event in _events()]

    assert {result["learning_rule"] for result in results} == {"shared_local_hebbian"}
    assert all(result["active_event_ids"] for result in results)
    assert column.state_budget_units() > 0


def test_synesthetic_linker_supports_non_language_routes():
    events = _events()
    linker = SparseSynestheticLinker(max_links_per_event=4)
    linker.update(events)

    prediction = linker.predict(events[2], target_modality="tactile")

    assert prediction["source_modality"] == "audio"
    assert prediction["target_modality"] == "tactile"
    assert prediction["predicted_missing_modality_events"]
    assert prediction["observed"] is False
    assert prediction["abstained"] is False


def test_thalamic_gate_equal_and_focused_modes_are_traceable():
    events = _events()
    gate = SparseThalamicGate(route_threshold=0.3)

    equal = gate.route(events, mode="equal")
    focused = gate.route(events, mode="focused", focused_modality="tactile")

    assert len(equal.routed_events) == 4
    assert len(focused.routed_events) == 4
    tactile_trace = next(row for row in focused.trace if row["modality"] == "tactile")
    assert tactile_trace["focus_gain"] == 0.2


def test_modality_adapters_share_ir_and_record_specialization_sources():
    binder = SparseTemporalBinder(window_ms=32.0)
    adapters = [
        LanguageEventAdapter(max_events=8),
        VisionEventAdapter(max_events=8),
        AudioEventAdapter(max_events=8),
        TactileEventAdapter(max_events=8),
    ]
    events = [
        adapter.encode(
            ["hard", "edge"],
            binder=binder,
            timestamp_ms=10.0,
            source_id=f"{adapter.modality}-source",
            source_ref="fixture://hard",
            latent_cluster_id="latent_hard",
            latent_signature=[7],
            topology_terms=["local"],
            gate_history_terms=["recent_success"],
        )
        for adapter in adapters
    ]

    assert {event.modality for event in events} == {"language", "vision", "audio", "tactile"}
    assert all(event.sparse_signature for event in events)
    assert all(event.source_ref == "fixture://hard" for event in events)
    assert all(
        set(event.specialization_factors)
        == {"input_statistics", "timing_profile", "topology", "own_latent", "gate_history"}
        for event in events
    )


def test_thalamic_gate_accepts_bounded_dendritic_route_hints():
    events = _events()
    result = SparseThalamicGate(route_threshold=0.3).route(
        events,
        route_hints={"audio-hard": 10.0, "vision-hard": -10.0},
    )

    audio_trace = next(row for row in result.trace if row["source_id"] == "audio-hard")
    vision_trace = next(row for row in result.trace if row["source_id"] == "vision-hard")
    assert audio_trace["route_hint"] == 0.25
    assert vision_trace["route_hint"] == -0.25
