import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.learning.homeostasis import (
    AdaptiveThresholdHomeostasis,
    NeuronActivityTracker,
    SynapticScalingManager,
)


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


def test_activity_tracker_global_rate_is_computable():
    tracker = NeuronActivityTracker(decay=0.9, slow_decay=0.99)
    for _ in range(20):
        tracker.step()
        tracker.update(1, fired=True)
        tracker.update(2, fired=False)
    g = tracker.get_global_rate()
    assert 0.0 <= g <= 1.0
    assert g > 0.0


def test_scaling_factor_has_deadband_and_population_feedback():
    manager = SynapticScalingManager(
        target_rate=0.05,
        scaling_lr=0.1,
        min_scale=0.9,
        max_scale=1.1,
        deadband=0.2,
        global_weight=0.5,
    )
    assert manager.compute_scaling_factor(current_rate=0.051, population_rate=0.052) == 1.0
    up = manager.compute_scaling_factor(current_rate=0.0, population_rate=0.0)
    down = manager.compute_scaling_factor(current_rate=0.5, population_rate=0.5)
    assert up > 1.0
    assert down < 1.0


def test_adaptive_threshold_homeostasis_raises_and_recovers():
    manager = AdaptiveThresholdHomeostasis(
        target_rate=0.1,
        adaptation_rate=0.5,
        decay=0.8,
        min_threshold=0.0,
        max_threshold=2.0,
    )

    for _ in range(8):
        manager.update([3], population_size=10)
    raised = manager.get_threshold(3)
    assert raised > 0.0

    for _ in range(20):
        manager.update([], population_size=10)
    recovered = manager.get_threshold(3)
    assert 0.0 <= recovered < raised


def test_adaptive_threshold_homeostasis_modulates_scores():
    manager = AdaptiveThresholdHomeostasis(
        target_rate=0.05,
        adaptation_rate=0.4,
        decay=0.9,
        min_threshold=0.0,
        max_threshold=3.0,
    )

    for _ in range(10):
        manager.update([9], population_size=100)

    raw = 1.0
    modulated = manager.modulate(9, raw, strength=1.0)
    assert 0.0 < modulated < raw
