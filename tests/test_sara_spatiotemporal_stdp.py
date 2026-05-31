# Directory Path: tests/test_sara_spatiotemporal_stdp.py
# English Title: SARA Spatio-Temporal STDP Tests
# Purpose/Content: Verifies the sparse event-driven STDP network used by hippocampal consolidation.

from sara_engine.models.spatiotemporal_stdp import SpatioTemporalSNN


def test_spatiotemporal_snn_advances_with_sparse_activity_report() -> None:
    snn = SpatioTemporalSNN(n_in=16, n_low=8, n_high=4, n_ctx=3, seed=3)

    first = snn.step([1.0 if index in {3, 4, 5} else 0.0 for index in range(16)])
    second = snn.step([1.0 if index in {4, 5, 6} else 0.0 for index in range(16)])

    assert first["step_count"] == 1.0
    assert second["step_count"] == 2.0
    assert second["input_spikes"] >= 1.0
    assert second["synapse_count"] > 0.0
    assert 0.0 <= second["average_weight"] <= 2.0


def test_spatiotemporal_snn_keeps_sparse_weights_bounded_under_replay() -> None:
    snn = SpatioTemporalSNN(n_in=20, n_low=10, n_high=5, n_ctx=4, seed=5)

    for step in range(24):
        active = {step % 20, (step + 1) % 20, (step + 2) % 20}
        report = snn.step([1.0 if index in active else 0.0 for index in range(20)])

    assert report["step_count"] == 24.0
    assert 0.0 <= report["average_weight"] <= 2.0
    assert all(0.0 <= synapse.weight <= 2.0 for synapse in snn.in_low)
    assert all(0.0 <= synapse.weight <= 2.0 for synapse in snn.ctx_ctx)
