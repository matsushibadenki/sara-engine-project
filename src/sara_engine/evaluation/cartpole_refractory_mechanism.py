"""Synthetic refractory-versus-feature-loss mechanism study, not CartPole scoring."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from sara_engine.evaluation.event_unit_causal_isolation import UnitEvent
from sara_engine.neuro.neuron import Neuron


ARMS = (
    "scalar_count", "compact_gap", "spiking_refractory",
    "refractory_disabled", "rate_matched_mask", "timing_removed",
)


@dataclass(frozen=True)
class MechanismEpisode:
    identity: str
    split: str
    seed: int
    index: int
    events: tuple[UnitEvent, ...]
    target: int
    base_route: int


def generate_episodes(*, split: str, seed: int) -> list[MechanismEpisode]:
    """Balance near/far gaps, mask states, routes, and nuisance within each seed."""
    if split not in ("training", "development") or type(seed) is not int or not 0 <= seed < 5:
        raise ValueError("Invalid mechanism split or seed")
    count = 64 if split == "training" else 32
    offset = 1 if split == "training" else 20
    nuisance_base = 4 if split == "training" else 8
    episodes = []
    for index in range(count):
        target = (index // 2) % 2
        gap = 1 if target == 0 else 3
        start = offset + ((index // 4 + seed) % 4)
        base_route = index % 2 + 2 * ((index // 4) % 2)
        nuisance_route = nuisance_base + ((index // 4 + seed) % 4)
        events = tuple(sorted((
            UnitEvent(base_route, start, 0),
            UnitEvent(nuisance_route, start + 2, 0),
            UnitEvent(base_route, start + gap, 0),
        ), key=lambda event: (event.time, event.route)))
        episodes.append(MechanismEpisode(
            f"refractory-mechanism-v1:{split}:{seed}:{index}", split, seed,
            index, events, target, base_route,
        ))
    random.Random(700000 + seed + (0 if split == "training" else 1000)).shuffle(episodes)
    return episodes


def _base_times(episode: MechanismEpisode) -> tuple[int, int]:
    times = tuple(event.time for event in episode.events if event.route == episode.base_route)
    if len(times) != 2 or times[1] <= times[0]:
        raise ValueError("Expected two ordered base-route events")
    return times


def feature_for_arm(episode: MechanismEpisode, arm: str) -> int:
    """Emit one bounded local feature without reading the target."""
    if arm not in ARMS:
        raise ValueError("Unknown mechanism arm")
    first, second = _base_times(episode)
    if arm == "scalar_count":
        return 2
    if arm == "compact_gap":
        return 1 if second - first <= 2 else 2
    if arm == "rate_matched_mask":
        return 1 if episode.index % 2 == 0 else 2
    if arm == "timing_removed":
        second = first + 3
    neuron = Neuron(episode.base_route, num_branches=1)
    spikes = 0
    for tick in range(first, second + 1):
        if tick in (first, second):
            neuron.add_input_to_branch(0, 1.6)
        if arm == "refractory_disabled":
            neuron.refractory_time = 0
        spikes += neuron.step()
    return spikes


class LocalFeatureLearner:
    """The same two-action, bounded mistake update for every feature arm."""

    def __init__(self):
        self.weights = [0, 0, 0, 0]
        self.updates = 0

    def predict(self, feature: int) -> int:
        if feature not in (1, 2):
            raise ValueError("Feature must be one or two")
        offset = (feature - 1) * 2
        left, right = self.weights[offset], self.weights[offset + 1]
        return int(right > left)

    def learn(self, feature: int, target: int) -> int:
        if type(target) is not int or target not in (0, 1):
            raise ValueError("Target must be binary")
        prediction = self.predict(feature)
        if prediction != target:
            offset = (feature - 1) * 2
            target_index = offset + target
            predicted_index = offset + prediction
            self.weights[target_index] = min(4, self.weights[target_index] + 1)
            self.weights[predicted_index] = max(-4, self.weights[predicted_index] - 1)
            self.updates += 1
        return prediction

    def snapshot(self) -> tuple[int, int, int, int]:
        return (self.weights[0], self.weights[1], self.weights[2], self.weights[3])


def evaluate_arm(*, arm: str, seed: int) -> dict:
    """Score one synthetic development split only after protocol registration."""
    learner = LocalFeatureLearner()
    training = generate_episodes(split="training", seed=seed)
    development = generate_episodes(split="development", seed=seed)
    for episode in training:
        learner.learn(feature_for_arm(episode, arm), episode.target)
    before = learner.snapshot()
    development_features = tuple(feature_for_arm(episode, arm) for episode in development)
    predictions = tuple(learner.predict(feature) for feature in development_features)
    correct = sum(prediction == episode.target
                  for prediction, episode in zip(predictions, development))
    if learner.snapshot() != before:
        raise AssertionError("Development evaluation changed weights")
    return {"arm": arm, "seed": seed, "development_correct": correct,
            "development_count": len(development), "training_updates": learner.updates,
            "development_feature_one_count": development_features.count(1),
            "development_predictions": predictions, "final_weights": before}
