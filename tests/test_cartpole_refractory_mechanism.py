"""Generator and feature invariants; no development arm is scored here."""
from collections import Counter

import pytest

from sara_engine.evaluation.cartpole_refractory_mechanism import (
    ARMS, LocalFeatureLearner, feature_for_arm, generate_episodes,
)


def test_balanced_splits_and_disjoint_absolute_time_nuisance():
    for seed in range(5):
        training = generate_episodes(split="training", seed=seed)
        development = generate_episodes(split="development", seed=seed)
        assert len(training) == 64 and len(development) == 32
        assert Counter(e.target for e in training) == {0: 32, 1: 32}
        assert Counter(e.target for e in development) == {0: 16, 1: 16}
        for split, episodes in (("training", training), ("development", development)):
            for target in (0, 1):
                selected = [e for e in episodes if e.target == target]
                assert Counter(e.index % 2 for e in selected) == {0: len(selected) // 2,
                                                                 1: len(selected) // 2}
                assert set(e.base_route for e in selected) == {0, 1, 2, 3}
                assert set(event.route for e in selected for event in e.events
                           if event.route >= 4) == (set(range(4, 8)) if split == "training"
                                                  else set(range(8, 12)))
        assert {event.time for e in training for event in e.events}.isdisjoint(
            {event.time for e in development for event in e.events}
        )


def test_feature_controls_separate_refractory_timing_from_balanced_mask():
    episodes = generate_episodes(split="training", seed=0)
    by_index = {e.index: e for e in episodes}
    for index, episode in by_index.items():
        assert feature_for_arm(episode, "scalar_count") == 2
        assert feature_for_arm(episode, "compact_gap") == (1 if episode.target == 0 else 2)
        assert feature_for_arm(episode, "spiking_refractory") == (1 if episode.target == 0 else 2)
        assert feature_for_arm(episode, "refractory_disabled") == 2
        assert feature_for_arm(episode, "timing_removed") == 2
        assert feature_for_arm(episode, "rate_matched_mask") == (1 if index % 2 == 0 else 2)
    assert len(ARMS) == 6


def test_local_update_is_bounded_and_prediction_does_not_learn():
    learner = LocalFeatureLearner()
    assert learner.predict(1) == 0
    learner.learn(1, 1)
    assert learner.predict(1) == 1
    before = learner.snapshot()
    learner.predict(1)
    assert learner.snapshot() == before
    for _ in range(20):
        learner.learn(1, 0)
        learner.learn(1, 1)
    assert all(-4 <= weight <= 4 for weight in learner.snapshot())
    with pytest.raises(ValueError):
        learner.predict(3)
