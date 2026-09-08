import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.core.layers import DynamicLiquidLayer
from sara_engine.neuro.neuron import Neuron
from sara_engine.neuro.synapse import Synapse


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


def test_inhibitory_synapse_delivers_signed_current_to_branch():
    pre = Neuron(neuron_id=1)
    post = Neuron(neuron_id=2, num_branches=1)
    pre.spike = True
    synapse = Synapse(pre, post, post_branch_idx=0, is_inhibitory=True)
    synapse.weight = -1.0

    synapse.step()

    assert post.branches[0].current_input < 0.0
    assert post.active_branches == {0}
    assert post.step() is False
    assert post.v < 0.0


def test_dendritic_branch_combines_excitation_and_inhibition():
    neuron = Neuron(neuron_id=3, num_branches=1)
    neuron.v = 0.8
    neuron.add_input_to_branch(0, 0.3)
    neuron.add_input_to_branch(0, -0.6)

    assert neuron.step() is False
    assert neuron.v < 0.8 * neuron.leak
