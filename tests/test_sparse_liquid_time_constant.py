from sara_engine.nn.sparse_liquid_time_constant import SparseLiquidTimeConstantNeuron


def test_event_updates_are_bounded_and_replay_deterministic() -> None:
    events = ((1, 0.0), (6, 1.0), (1, 0.0))
    first = SparseLiquidTimeConstantNeuron().run(events)
    replay = SparseLiquidTimeConstantNeuron().run(events)

    assert first == replay
    assert all(trace.event_cost == 4 for trace in first)
    assert all(trace.update_count == 1 for trace in first)
    assert all(-1.0 <= trace.state <= 1.0 for trace in first)
    assert all(2.0 <= trace.time_constant <= 24.0 for trace in first)


def test_fixed_control_disables_adaptive_threshold() -> None:
    events = ((1, 0.0), (6, 1.0))
    liquid = SparseLiquidTimeConstantNeuron(adaptive_threshold=True).run(events)
    fixed = SparseLiquidTimeConstantNeuron(adaptive_threshold=False).run(events)

    assert liquid[-1].spike is True
    assert fixed[-1].spike is False
