import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.cognitive.global_workspace import GlobalWorkspace
from sara_engine.models.liquid_reservoir import LiquidReservoir
from sara_engine.models.lsm_network import LSMNetwork


def test_global_workspace_selects_and_homeostatically_recovers():
    workspace = GlobalWorkspace(num_candidates=4, inhibition_factor=0.2, winner_threshold=0.5, decay=0.8)

    winner = -1
    for _ in range(3):
        winner = workspace.step([1.2, 0.1, 0.0, 0.0])
    assert winner == 0

    raised = workspace.homeostasis.get_threshold(0)
    assert raised > 0.0

    for _ in range(60):
        workspace.step([0.0, 0.0, 0.0, 0.0])
    recovered = workspace.homeostasis.get_threshold(0)
    assert 0.0 <= recovered < raised


def test_liquid_reservoir_criticality_control_modifies_gain():
    reservoir = LiquidReservoir(n_neurons=12, p_connect=0.2)
    base_gain = reservoir.critical_gain
    reservoir._apply_criticality_control(1.8)
    assert reservoir.critical_gain < base_gain
    reservoir._apply_criticality_control(0.4)
    assert reservoir.critical_gain > 0.7


def test_lsm_network_criticality_control_shifts_thresholds():
    network = LSMNetwork(n_input=4, n_liquid=8, n_output=2)
    before = network.liquid[0].threshold
    network._apply_criticality_control(1.8)
    after_overfire = network.liquid[0].threshold
    assert after_overfire > before

    network._apply_criticality_control(0.2)
    after_underfire = network.liquid[0].threshold
    assert after_underfire < after_overfire
