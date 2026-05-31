import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.core.layers import DynamicLiquidLayer
from sara_engine.neuro.neuron import Neuron


def test_dynamic_liquid_layer_load_state_dict_handles_missing_keys():
    layer = DynamicLiquidLayer(input_size=4, hidden_size=8, decay=0.9)
    original_in_weights = [dict(item) for item in layer.in_weights]
    original_rec_weights = [dict(item) for item in layer.rec_weights]

    layer.load_state_dict(
        {
            "firing_rates": [0.1] * layer.size,
            "v": [0.0] * layer.size,
            "refractory": [0.0] * layer.size,
        }
    )

    assert layer.in_weights == original_in_weights
    assert layer.rec_weights == original_rec_weights


def test_neuron_step_keeps_strong_inhibitory_voltage_without_input():
    neuron = Neuron(neuron_id=1, is_inhibitory=True, num_branches=2)
    neuron.v = -1.5

    fired = neuron.step()

    assert fired is False
    assert neuron.v < -1.0
