# Directory Path: tests/test_modular_snn.py
# English Title: Modular SNN Tests
# Purpose/Content: Verifies sparse modular LIF layers and local STDP connections.

from snn_models.modular_snn import LIFLayer, STDPConnection


def test_modular_snn_connection_propagates_sparse_spikes() -> None:
    pre = LIFLayer(3, label="pre")
    post = LIFLayer(2, label="post")
    connection = STDPConnection(pre, post, conn_type="all_to_all", seed=1)

    pre.spikes[1] = True
    currents = connection.propagate()

    assert len(connection.synapses) == 6
    assert len(currents) == 2
    assert all(current > 0.0 for current in currents)


def test_modular_snn_stdp_keeps_weights_bounded() -> None:
    pre = LIFLayer(2, label="pre")
    post = LIFLayer(2, label="post")
    connection = STDPConnection(pre, post, conn_type="one_to_one", seed=2)

    pre.spikes = [True, False]
    pre.traces = [1.0, 0.0]
    post.spikes = [True, False]
    post.traces = [1.0, 0.0]

    connection.update_weights(w_min=0.0, w_max=0.2)

    assert len(connection.synapses) == 2
    assert all(0.0 <= weight <= 0.2 for _pre, _post, weight in connection.synapses)
