"""Development-only arms for the frozen minimal event-unit isolation study."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Iterable, Sequence

from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.neuro.neuron import Neuron


FAMILIES = (
    "order_sensitive_timing",
    "persistent_state",
    "refractory_suppression",
    "branch_specific_conjunction",
    "stateless_route_control",
)


@dataclass(frozen=True)
class UnitEvent:
    route: int
    time: int
    branch: int


@dataclass(frozen=True)
class UnitEpisode:
    identity: str
    family: str
    events: tuple[UnitEvent, ...]
    label: int


@dataclass(frozen=True)
class UnitPrediction:
    owner: int
    sequence: int
    predicted: int
    features: tuple[int, ...]
    event_work: int


def generate_episodes(
    *, seeds: Sequence[int], count_per_family: int, split: str, namespace: str | None = None
) -> list[UnitEpisode]:
    if split not in ("training", "development") or count_per_family < 1:
        raise ValueError("development generator arguments are invalid")
    episodes: list[UnitEpisode] = []
    for seed in seeds:
        for family_index, family in enumerate(FAMILIES):
            for index in range(count_per_family):
                rng = random.Random(seed * 1_000_003 + family_index * 10_007 + index * 101)
                label = rng.randrange(2)
                distractor = rng.randrange(4, 8)
                if family == "order_sensitive_timing":
                    routes = (0, 1) if label else (1, 0)
                    events = (UnitEvent(routes[0], 1, 0), UnitEvent(routes[1], 3 if label else 8, 0))
                elif family == "persistent_state":
                    first = 2 if label else 3
                    events = (UnitEvent(first, 1, 0), UnitEvent(distractor, 4, 0), UnitEvent(8, 10, 0))
                elif family == "refractory_suppression":
                    gap = 1 if label else 4
                    events = (UnitEvent(9, 1, 0), UnitEvent(9, 1 + gap, 0), UnitEvent(distractor, 8, 0))
                elif family == "branch_specific_conjunction":
                    pair = ((10, 0), (11, 1)) if label else ((10, 1), (11, 0))
                    events = (UnitEvent(pair[0][0], 1, pair[0][1]), UnitEvent(pair[1][0], 2, pair[1][1]))
                else:
                    route = 12 if label else 13
                    events = (UnitEvent(distractor, 1, 0), UnitEvent(route, 2, 0))
                episodes.append(UnitEpisode(
                    f"{namespace + ':' if namespace else ''}{split}:{seed}:{family}:{index}",
                    family,
                    events,
                    label,
                ))
    random.Random(sum(seeds) + (0 if split == "training" else 1)).shuffle(episodes)
    return episodes


class EventUnitLearner:
    """One bounded local update rule with arm-specific event feature generation."""

    ARM_NAMES = (
        "A_scalar_local",
        "B_compact_event",
        "C_stateful_spiking",
        "D_dendritic_structural",
    )

    INTERVENTIONS = (
        "none",
        "time_shuffle",
        "state_reset",
        "spike_count_preserving_shuffle",
        "refractory_disable",
        "branch_assignment_shuffle",
    )

    def __init__(
        self,
        arm: str,
        *,
        intervention: str = "none",
        max_routes: int = 512,
        max_features: int = 8192,
        capacity_reserve_bytes: int = 0,
    ) -> None:
        if arm not in self.ARM_NAMES:
            raise ValueError("unknown event-unit arm")
        if intervention not in self.INTERVENTIONS:
            raise ValueError("unknown event-unit intervention")
        if capacity_reserve_bytes < 0 or capacity_reserve_bytes > 4_194_304:
            raise ValueError("capacity reserve is invalid")
        self.arm = arm
        self.intervention = intervention
        self.max_routes = max_routes
        self.max_features = max_features
        self.weights: dict[int, float] = {}
        self._feature_ids: dict[tuple, int] = {}
        self._neurons: dict[int, Neuron] = {}
        self._pending: UnitPrediction | None = None
        self._sequence = 0
        self._capacity_reserve = bytearray(capacity_reserve_bytes)

    def _feature(self, key: tuple) -> int:
        if key not in self._feature_ids:
            if len(self._feature_ids) >= self.max_features:
                raise ValueError("feature budget exceeded")
            self._feature_ids[key] = len(self._feature_ids)
        return self._feature_ids[key]

    def _stateless_features(self, events: Iterable[UnitEvent]) -> tuple[tuple[int, ...], int]:
        admitted = tuple(events)
        if any(event.route < 0 or event.route >= self.max_routes for event in admitted):
            raise ValueError("route budget exceeded")
        return tuple(dict.fromkeys(self._feature(("route", event.route)) for event in admitted)), len(admitted)

    def _stateful_features(self, episode: UnitEpisode, *, dendritic: bool) -> tuple[tuple[int, ...], int]:
        # Episodes are causally isolated samples. Retain dynamics between events,
        # but never leak membrane or refractory state across episode boundaries.
        for neuron in self._neurons.values():
            neuron.v = 0.0
            neuron.spike = False
            neuron.refractory_time = 0
            neuron.active_branches.clear()
            for branch in neuron.branches:
                branch.current_input = 0.0
                branch.is_active = False
        features: list[int] = []
        previous: UnitEvent | None = None
        work = 0
        events = episode.events
        if self.intervention == "time_shuffle":
            events = tuple(UnitEvent(event.route, index + 1, event.branch) for index, event in enumerate(events))
        branch_order = [event.branch for event in events]
        shuffle_rng = random.Random(sum(ord(char) for char in episode.identity) + 91_601)
        if self.intervention == "branch_assignment_shuffle":
            random.Random(sum(ord(char) for char in episode.identity)).shuffle(branch_order)
        for event_index, raw_event in enumerate(events):
            event = UnitEvent(raw_event.route, raw_event.time, branch_order[event_index])
            if event.route < 0 or event.route >= self.max_routes:
                raise ValueError("route budget exceeded")
            neuron = self._neurons.get(event.route)
            if neuron is None:
                neuron = Neuron(event.route, num_branches=2 if dendritic else 1)
                self._neurons[event.route] = neuron
            if self.intervention in ("state_reset", "refractory_disable"):
                neuron.v = 0.0
                neuron.spike = False
                neuron.refractory_time = 0
            branch = event.branch if dendritic else 0
            neuron.add_input_to_branch(branch, 1.6)
            spiked = neuron.step()
            work += 5
            features.append(self._feature(("route", event.route)))
            if previous is not None and self.intervention != "state_reset":
                gap = event.time - previous.time
                gap_bucket = 0 if gap <= 1 else (1 if gap <= 3 else 2)
                if spiked:
                    pair = (previous.route, event.route)
                    if self.intervention == "spike_count_preserving_shuffle":
                        pair = (shuffle_rng.randrange(14), shuffle_rng.randrange(14))
                    features.append(self._feature(("time", pair[0], pair[1], gap_bucket)))
                if dendritic:
                    features.append(self._feature(
                        ("branch", previous.route, previous.branch, event.route, event.branch)
                    ))
            previous = event
        return tuple(dict.fromkeys(features)), work

    def predict(self, episode: UnitEpisode) -> UnitPrediction:
        if self._pending is not None:
            raise ValueError("resolve the pending prediction first")
        if self.arm in ("A_scalar_local", "B_compact_event"):
            features, work = self._stateless_features(episode.events)
        else:
            features, work = self._stateful_features(
                episode, dendritic=self.arm == "D_dendritic_structural"
            )
        score = sum(self.weights.get(feature, 0.0) for feature in features)
        self._sequence += 1
        receipt = UnitPrediction(id(self), self._sequence, int(score > 0.0), features, work + len(features))
        self._pending = receipt
        return receipt

    def observe(self, prediction: UnitPrediction, target: int) -> int:
        if prediction is not self._pending or prediction.owner != id(self):
            raise ValueError("prediction receipt is not pending")
        if target not in (0, 1):
            raise ValueError("target must be binary")
        updates = 0
        if prediction.predicted != target and prediction.features:
            delta = (0.3 if target else -0.3) / len(prediction.features)
            for feature in prediction.features:
                self.weights[feature] = max(-2.0, min(2.0, self.weights.get(feature, 0.0) + delta))
                updates += 1
        self._pending = None
        return updates

    def state_bytes(self) -> int:
        if self._pending is not None:
            raise ValueError("cannot measure pending learner state")
        return _deep_size((self.weights, self._feature_ids, self._neurons, self._capacity_reserve))


class EventUnitV2Learner(EventUnitLearner):
    """Factorial v2 learner separating temporal, refractory and branch state."""

    ARM_NAMES = (
        "B_compact_event",
        "C_temporal_state",
        "R_temporal_refractory",
        "D_temporal_dendritic",
    )

    def _temporal_features(self, episode: UnitEpisode, *, dendritic: bool) -> tuple[tuple[int, ...], int]:
        events = episode.events
        if self.intervention == "time_shuffle":
            events = tuple(UnitEvent(event.route, index + 1, event.branch) for index, event in enumerate(events))
        branch_order = [event.branch for event in events]
        if self.intervention == "branch_assignment_shuffle":
            random.Random(sum(ord(char) for char in episode.identity)).shuffle(branch_order)
        features: list[int] = []
        previous: UnitEvent | None = None
        for index, raw_event in enumerate(events):
            event = UnitEvent(raw_event.route, raw_event.time, branch_order[index])
            features.append(self._feature(("route", event.route)))
            if previous is not None and self.intervention != "state_reset":
                gap = event.time - previous.time
                gap_bucket = 0 if gap <= 1 else (1 if gap <= 3 else 2)
                features.append(self._feature(("time", previous.route, event.route, gap_bucket)))
                if dendritic:
                    features.append(self._feature(
                        ("branch", previous.route, previous.branch, event.route, event.branch)
                    ))
            previous = event
        work_per_event = 3 if dendritic else 2
        return tuple(dict.fromkeys(features)), len(events) * work_per_event + len(features)

    def predict(self, episode: UnitEpisode) -> UnitPrediction:
        if self._pending is not None:
            raise ValueError("resolve the pending prediction first")
        if self.arm == "B_compact_event":
            features, work = self._stateless_features(episode.events)
        elif self.arm == "C_temporal_state":
            features, work = self._temporal_features(episode, dendritic=False)
        elif self.arm == "R_temporal_refractory":
            features, work = self._stateful_features(episode, dendritic=False)
        else:
            features, work = self._temporal_features(episode, dendritic=True)
        score = sum(self.weights.get(feature, 0.0) for feature in features)
        self._sequence += 1
        receipt = UnitPrediction(id(self), self._sequence, int(score > 0.0), features, work)
        self._pending = receipt
        return receipt


def run_development_arm(
    arm: str,
    training: Sequence[UnitEpisode],
    development: Sequence[UnitEpisode],
    *,
    intervention: str = "none",
    outcome_shuffle_seed: int | None = None,
    capacity_reserve_bytes: int = 0,
) -> dict:
    learner = EventUnitLearner(
        arm, intervention=intervention, capacity_reserve_bytes=capacity_reserve_bytes
    )
    maximum_work = 0
    updates = 0
    outcomes = [episode.label for episode in training]
    development_outcomes = [episode.label for episode in development]
    if outcome_shuffle_seed is not None:
        random.Random(outcome_shuffle_seed).shuffle(outcomes)
        random.Random(outcome_shuffle_seed + 1).shuffle(development_outcomes)
    for episode, outcome in zip(training, outcomes):
        prediction = learner.predict(episode)
        maximum_work = max(maximum_work, prediction.event_work)
        updates += learner.observe(prediction, outcome)
    correct = 0
    rows = []
    family_totals = {family: 0 for family in FAMILIES}
    family_correct = {family: 0 for family in FAMILIES}
    for episode, observed_outcome in zip(development, development_outcomes):
        prediction = learner.predict(episode)
        maximum_work = max(maximum_work, prediction.event_work)
        correct += int(prediction.predicted == episode.label)
        family_totals[episode.family] += 1
        family_correct[episode.family] += int(prediction.predicted == episode.label)
        rows.append((episode.identity, prediction.predicted))
        updates += learner.observe(prediction, observed_outcome)
    digest = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    return {
        "accuracy": correct / len(development),
        "accuracy_by_family": {
            family: family_correct[family] / family_totals[family] for family in FAMILIES
        },
        "predictions": len(development),
        "updates": updates,
        "maximum_event_work": maximum_work,
        "state_bytes": learner.state_bytes(),
        "feature_count": len(learner._feature_ids),
        "neuron_count": len(learner._neurons),
        "prediction_rows": rows,
        "prediction_trace_sha256": digest,
    }


def run_v2_development_arm(
    arm: str,
    training: Sequence[UnitEpisode],
    development: Sequence[UnitEpisode],
    *,
    intervention: str = "none",
    outcome_shuffle_seed: int | None = None,
    capacity_reserve_bytes: int = 0,
) -> dict:
    learner = EventUnitV2Learner(
        arm, intervention=intervention, capacity_reserve_bytes=capacity_reserve_bytes
    )
    maximum_work = 0
    updates = 0
    training_outcomes = [episode.label for episode in training]
    development_outcomes = [episode.label for episode in development]
    if outcome_shuffle_seed is not None:
        random.Random(outcome_shuffle_seed).shuffle(training_outcomes)
        random.Random(outcome_shuffle_seed + 1).shuffle(development_outcomes)
    for episode, outcome in zip(training, training_outcomes):
        prediction = learner.predict(episode)
        maximum_work = max(maximum_work, prediction.event_work)
        updates += learner.observe(prediction, outcome)
    rows = []
    family_totals = {family: 0 for family in FAMILIES}
    family_correct = {family: 0 for family in FAMILIES}
    for episode, observed_outcome in zip(development, development_outcomes):
        prediction = learner.predict(episode)
        maximum_work = max(maximum_work, prediction.event_work)
        family_totals[episode.family] += 1
        family_correct[episode.family] += int(prediction.predicted == episode.label)
        rows.append((episode.identity, prediction.predicted))
        updates += learner.observe(prediction, observed_outcome)
    accuracy_by_family = {
        family: family_correct[family] / family_totals[family] for family in FAMILIES
    }
    return {
        "accuracy": sum(family_correct.values()) / len(development),
        "accuracy_by_family": accuracy_by_family,
        "predictions": len(development),
        "updates": updates,
        "maximum_event_work": maximum_work,
        "state_bytes": learner.state_bytes(),
        "feature_count": len(learner._feature_ids),
        "neuron_count": len(learner._neurons),
        "prediction_rows": rows,
        "prediction_trace_sha256": hashlib.sha256(
            json.dumps(rows, separators=(",", ":")).encode()
        ).hexdigest(),
    }


__all__ = [
    "EventUnitLearner",
    "EventUnitV2Learner",
    "FAMILIES",
    "UnitEpisode",
    "UnitEvent",
    "UnitPrediction",
    "generate_episodes",
    "run_development_arm",
    "run_v2_development_arm",
]
