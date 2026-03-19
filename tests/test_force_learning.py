import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.learning.force import ForceReadout
from sara_engine.encoders import TimeSeriesCurrentEncoder
from sara_engine.models.liquid_reservoir import LiquidReservoir
from sara_engine.models.lsm_network import LSMNetwork


def test_force_readout_tracks_linear_target_online():
    readout = ForceReadout(input_size=2, output_size=1, alpha=1.0)
    state = [1.0, -0.5]
    target = [0.75]

    first_error = abs(readout.predict(state)[0] - target[0])
    for _ in range(12):
        readout.update(state, target)
    final_error = abs(readout.predict(state)[0] - target[0])

    assert final_error < first_error
    assert final_error < 0.05


def test_time_series_current_encoder_produces_phase_sensitive_currents():
    encoder = TimeSeriesCurrentEncoder()

    rising = encoder.encode(0.4, 0.1, 6)
    falling = encoder.encode(0.4, 0.7, 6)

    assert len(rising) == 6
    assert rising[2] > 0.0
    assert rising[3] == 0.0
    assert falling[2] == 0.0
    assert falling[3] > 0.0


def test_lsm_network_exposes_filtered_reservoir_state_for_force_learning():
    network = LSMNetwork(n_input=4, n_liquid=6, n_output=2, enable_force_readout=True, force_output_dim=1, readout_decay=0.8)

    prediction = network.train_force([True, False, True, False], [0.5])
    state = network.get_reservoir_state()

    assert len(prediction) == 1
    assert len(state) == network.state_size
    assert max(state[:4]) > 0.0


def test_lsm_network_force_training_reduces_prediction_error_on_constant_drive():
    network = LSMNetwork(n_input=3, n_liquid=5, n_output=1, enable_force_readout=True, force_output_dim=1, readout_decay=0.85)
    target = [1.0]

    first_prediction = network.train_force([True, False, True], target)[0]
    for _ in range(25):
        network.train_force([True, False, True], target)
    final_prediction = network.predict_force([True, False, True])[0]

    assert abs(final_prediction - target[0]) < abs(first_prediction - target[0])


def test_liquid_reservoir_exposes_filtered_state_for_force_learning():
    reservoir = LiquidReservoir(
        n_neurons=10,
        p_connect=0.2,
        enable_force_readout=True,
        force_output_dim=1,
        readout_decay=0.8,
    )

    prediction = reservoir.train_force([1.0] + [0.0] * 9, [0.25])
    state = reservoir.get_reservoir_state()

    assert len(prediction) == 1
    assert len(state) == reservoir.state_size
    assert state[0] > 0.0


def test_liquid_reservoir_force_training_reduces_error_on_constant_current():
    reservoir = LiquidReservoir(
        n_neurons=12,
        p_connect=0.15,
        enable_force_readout=True,
        force_output_dim=1,
        readout_decay=0.9,
    )
    drive = [1.2] + [0.0] * 11
    target = [0.8]

    first_prediction = reservoir.train_force(drive, target)[0]
    for _ in range(20):
        reservoir.train_force(drive, target)
    final_prediction = reservoir.predict_force(drive)[0]

    assert abs(final_prediction - target[0]) < abs(first_prediction - target[0])


def test_reset_dynamic_state_preserves_force_readout_by_default():
    reservoir = LiquidReservoir(
        n_neurons=8,
        p_connect=0.1,
        enable_force_readout=True,
        force_output_dim=1,
    )
    drive = [1.0] + [0.0] * 7
    target = [0.6]

    for _ in range(10):
        reservoir.train_force(drive, target)
    assert reservoir.force_readout is not None
    learned_weights = [row[:] for row in reservoir.force_readout.weights]
    learned_bias = list(reservoir.force_readout.bias)

    reservoir.reset_dynamic_state()

    assert reservoir.force_readout.weights == learned_weights
    assert reservoir.force_readout.bias == learned_bias

    reservoir.reset_dynamic_state(reset_readout=True)
    assert reservoir.force_readout.weights != learned_weights
    assert reservoir.force_readout.bias != learned_bias
