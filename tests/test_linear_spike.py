from sara_engine.nn.linear_spike import LinearSpike


def test_linear_spike_tracks_bounded_runtime_state_and_resets():
    layer = LinearSpike(in_features=4, out_features=3, density=1.0)

    out_spikes = layer.forward([0, 2, 99, -1], learning=False)

    assert layer.last_input_spikes == [0, 2]
    assert layer.last_output_spikes == out_spikes
    assert len(layer.last_potentials) == 3
    assert any(value > 0.0 for value in layer.last_potentials)

    layer.reset_state()

    assert layer.last_input_spikes == []
    assert layer.last_output_spikes == []
    assert layer.last_potentials == [0.0, 0.0, 0.0]
    assert layer.local_update_count == 0


def test_linear_spike_counts_local_updates_during_learning():
    layer = LinearSpike(in_features=3, out_features=2, density=1.0)
    initial_synapse_count = sum(len(row) for row in layer.weights)

    layer.forward([0, 1], learning=True)

    assert layer.local_update_count > 0
    assert layer.local_update_count <= initial_synapse_count
