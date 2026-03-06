import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.learning.homeostasis import NeuronActivityTracker, SynapticScalingManager


def test_activity_tracker_rate_is_bounded_and_decays():
    tracker = NeuronActivityTracker(decay=0.9)
    nid = 7

    for _ in range(500):
        tracker.step()
        tracker.update(nid, fired=True)

    rate = tracker.get_rate(nid)
    assert 0.0 < rate <= 1.0

    for _ in range(100):
        tracker.step()
    decayed = tracker.get_rate(nid)
    assert decayed < rate


def test_scaling_factor_is_clamped():
    manager = SynapticScalingManager(target_rate=0.05, scaling_lr=1.0, min_scale=0.95, max_scale=1.05)
    assert manager.compute_scaling_factor(current_rate=100.0) == 0.95
    assert manager.compute_scaling_factor(current_rate=-100.0) == 1.05
