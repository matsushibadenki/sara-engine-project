from sara_engine.nn.phase_synchronized_binding_trace import (
    PhaseSynchronizedBindingTrace,
    evaluate_phase_synchronized_binding_trace,
)


def test_phase_synchronized_binding_trace_binds_distant_same_phase_events() -> None:
    trace_builder = PhaseSynchronizedBindingTrace(
        phase_buckets=8,
        min_temporal_distance=32,
        max_bindings=4,
    )

    trace = trace_builder.build(
        anchor_events=[{"event_id": 10, "phase": 2, "step": 0}],
        candidate_events=[{"event_id": 20, "phase": 10, "step": 80}],
    )

    assert trace["state_budget_ok"] is True
    assert trace["bindings"] == [
        {
            "anchor_id": 10,
            "candidate_id": 20,
            "anchor_bucket": 2,
            "candidate_bucket": 2,
            "temporal_distance": 80,
            "same_phase_bucket": True,
            "distant_enough": True,
        }
    ]


def test_phase_synchronized_binding_trace_rejects_phase_and_distance_noise() -> None:
    trace_builder = PhaseSynchronizedBindingTrace(
        phase_buckets=8,
        min_temporal_distance=32,
        max_bindings=4,
    )

    trace = trace_builder.build(
        anchor_events=[{"event_id": 10, "phase": 2, "step": 0}],
        candidate_events=[
            {"event_id": 21, "phase": 3, "step": 80},
            {"event_id": 22, "phase": 10, "step": 8},
        ],
    )

    rejected_pairs = {
        (pair["anchor_id"], pair["candidate_id"]) for pair in trace["rejected"]
    }
    assert trace["bindings"] == []
    assert (10, 21) in rejected_pairs
    assert (10, 22) in rejected_pairs


def test_phase_synchronized_binding_trace_evaluation_reports_observed_metrics() -> None:
    report = evaluate_phase_synchronized_binding_trace()

    assert report["observed_only"] is True
    assert report["metrics"]["phase_binding_coincidence_integrity"] == 1.0
    assert report["metrics"]["phase_binding_state_budget_integrity"] == 1.0
    assert report["traces"]["bound_pairs"] == [(101, 301)]


def test_phase_synchronized_binding_trace_is_exposed_from_nn_package() -> None:
    import sara_engine.nn as nn

    trace = nn.PhaseSynchronizedBindingTrace().build(
        anchor_events=[{"event_id": 1, "phase": 0, "step": 0}],
        candidate_events=[{"event_id": 2, "phase": 8, "step": 64}],
    )

    assert trace["binding_count"] == 1
    assert nn.evaluate_phase_synchronized_binding_trace()["metrics"][
        "phase_binding_coincidence_integrity"
    ] == 1.0
