"""Training diagnostics cannot request or score a CartPole split."""
from itertools import product

import pytest

from sara_engine.evaluation.cartpole_training_diagnostics import (
    refractory_gate_probe, summarize_training,
)


def test_configured_neuron_is_exactly_a_two_step_refractory_gate():
    for length in range(9):
        for pattern in product((False, True), repeat=length):
            probe = refractory_gate_probe(pattern)
            assert probe["observed"] == probe["refractory_mask"]
            assert all(value == 0.0 for value in probe["membrane"])


def test_training_summary_uses_fixed_quartiles_only():
    episodes = [{"steps": index + 1, "input_events": 4 * (index + 1),
                 "spikes": 2 * (index + 1), "terminal_updates": 1}
                for index in range(128)]
    summary = summarize_training([{"run_id": 0, "training": {"C": episodes}}])
    assert summary[0]["conditions"]["C"] == {
        "quartile_mean_steps": [16.5, 48.5, 80.5, 112.5],
        "episodes_at_least_250_steps": 0,
        "spike_to_input_ratio": 0.5,
        "terminal_updates": 128,
    }
    with pytest.raises(ValueError):
        refractory_gate_probe((1,))
