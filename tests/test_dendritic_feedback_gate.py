from sara_engine.learning.dendritic_feedback import (
    SparseDendriticFeedbackGate,
    precision_at_expected,
)


def test_dendritic_gate_promotes_supported_neighbor_event():
    gate = SparseDendriticFeedbackGate(event_budget=32)
    gate.update_local_links([1, 2], learning_rate=0.2)

    result = gate.gate(
        active_event_ids=[1, 2, 9],
        local_potentials={1: 0.95, 2: 0.92, 9: 1.01},
        recent_output_spikes=[1],
        neighbor_activity={2: [1], 9: [7, 8, 10]},
    )

    assert result.fallback_used is False
    assert 1 in result.gated_events
    assert 2 in result.gated_events
    assert result.convergence_steps == 1
    assert result.event_cost <= 32
    assert result.trace["trace_rows"]


def test_dendritic_gate_falls_back_when_event_budget_is_exceeded():
    gate = SparseDendriticFeedbackGate(event_budget=2)

    result = gate.gate(
        active_event_ids=[1, 2, 3],
        local_potentials={1: 1.1, 2: 0.2, 3: 0.2},
        neighbor_activity={2: [1, 3], 3: [1, 2]},
    )

    assert result.fallback_used is True
    assert result.gated_events == [1]
    assert result.event_cost <= 2
    assert result.trace["input_overflow"] is True


def test_dendritic_gate_homeostatic_clipping_bounds_state():
    gate = SparseDendriticFeedbackGate(homeostatic_clip=0.25)

    for _ in range(20):
        gate.update_local_links([4, 5], learning_rate=1.0)

    assert gate.state_budget_units() > 0
    assert all(abs(value) <= 0.25 for value in gate.local_weights.values())
    assert all(abs(value) <= 0.25 for value in gate.event_bias.values())


def test_dendritic_learning_rejects_work_before_state_growth():
    gate = SparseDendriticFeedbackGate(
        max_active_events=10,
        max_state_units=20,
        max_link_updates_per_call=12,
    )

    gate.update_local_links(range(100), learning_rate=0.2)

    assert gate.state_budget_units() <= 20
    assert gate.last_update_trace["input_overflow"] is True
    assert gate.last_update_trace["admitted_event_count"] == 10
    assert gate.last_update_trace["link_updates"] <= 12


def test_precision_at_expected_handles_empty_sets():
    assert precision_at_expected([], []) == 1.0
    assert precision_at_expected([], [1]) == 0.0
    assert precision_at_expected([1, 2], [2, 3]) == 0.5
