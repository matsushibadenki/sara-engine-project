from sara_engine.nn.delta_associative_memory import (
    DeltaAssociativeSpikeMemory,
    evaluate_delta_associative_spike_memory,
    evaluate_delta_memory_steering_trace,
)


def test_delta_associative_memory_writes_only_residual_events() -> None:
    memory = DeltaAssociativeSpikeMemory(capacity=4)

    trace = memory.update(
        context_events=[1, 2],
        predicted_events=[10],
        observed_events=[10, 11],
    )
    readout = memory.read([1, 2])

    assert trace["residual_ids"] == [11]
    assert trace["write_applied"] is True
    assert trace["state_budget_ok"] is True
    assert readout["predicted_ids"] == [11]
    assert readout["scores"][11] == 2.0


def test_delta_associative_memory_skips_predicted_observation_rewrites() -> None:
    memory = DeltaAssociativeSpikeMemory(capacity=4)
    memory.update(context_events=[1], predicted_events=[], observed_events=[10])

    trace = memory.update(
        context_events=[1],
        predicted_events=[10],
        observed_events=[10],
    )
    snapshot = memory.snapshot()

    assert trace["residual_ids"] == []
    assert trace["write_applied"] is False
    assert trace["interference_guard"] is True
    assert snapshot["state_units"] == 1
    assert snapshot["entries"][0]["weight"] == 1.0


def test_delta_associative_memory_respects_state_budget() -> None:
    memory = DeltaAssociativeSpikeMemory(capacity=2)

    memory.update(context_events=[1], predicted_events=[], observed_events=[10])
    memory.update(context_events=[2], predicted_events=[], observed_events=[20])
    trace = memory.update(context_events=[3], predicted_events=[], observed_events=[30])
    snapshot = memory.snapshot()

    assert trace["state_budget_ok"] is True
    assert trace["evicted_count"] == 1
    assert snapshot["state_units"] == 2
    assert all(entry["context_id"] in {2, 3} for entry in snapshot["entries"])


def test_delta_associative_memory_retention_gate_prunes_state() -> None:
    memory = DeltaAssociativeSpikeMemory(capacity=4, min_weight=0.25)
    memory.update(context_events=[1], predicted_events=[], observed_events=[10])

    trace = memory.update(
        context_events=[1],
        predicted_events=[10],
        observed_events=[10],
        retention_gate=0.1,
    )

    assert trace["state_units"] == 0
    assert memory.read([1])["predicted_ids"] == []


def test_delta_associative_memory_evaluation_reports_observed_metrics() -> None:
    report = evaluate_delta_associative_spike_memory()

    assert report["observed_only"] is True
    assert report["metrics"]["delta_memory_residual_write_integrity"] == 1.0
    assert report["metrics"]["delta_memory_retention_gate_stability"] == 1.0
    assert report["metrics"]["delta_memory_context_recall_without_text_reinjection"] == 1.0
    assert report["metrics"]["delta_memory_state_budget_integrity"] == 1.0
    assert report["metrics"]["delta_memory_interference_guard"] == 1.0


def test_delta_associative_memory_is_exposed_from_nn_package() -> None:
    import sara_engine.nn as nn

    memory = nn.DeltaAssociativeSpikeMemory(capacity=2)
    memory.update(context_events=[1], predicted_events=[], observed_events=[10])

    assert memory.read([1])["predicted_ids"] == [10]
    assert nn.evaluate_delta_memory_steering_trace()["metrics"]["delta_memory_steering_integrity"] == 1.0


def test_delta_associative_memory_builds_memory_steering_event() -> None:
    memory = DeltaAssociativeSpikeMemory(capacity=4)
    memory.update(context_events=[1, 2], predicted_events=[10], observed_events=[10, 11])

    event = memory.build_memory_steering_event([1, 2], branch_id="primary")

    assert event["event_type"] == "memory_steering_event"
    assert event["memory_type"] == "delta_associative_state"
    assert event["branch_id"] == "primary"
    assert event["steering_ids"] == [11]
    assert event["text_reinjection_used"] is False
    assert event["trace_complete"] is True


def test_delta_associative_memory_steering_evaluation_reports_stage_e_metrics() -> None:
    report = evaluate_delta_memory_steering_trace()

    assert report["observed_only"] is True
    assert report["metrics"]["delta_memory_steering_integrity"] == 1.0
    assert report["metrics"]["delta_memory_counterfactual_isolation"] == 1.0
    assert report["metrics"]["delta_memory_trace_observability"] == 1.0
    assert report["traces"]["primary_steering_event"]["steering_ids"] == [301]
    assert report["traces"]["counterfactual_steering_event"]["steering_ids"] == [302]
    assert report["traces"]["isolated_probe"]["steering_ids"] == []
