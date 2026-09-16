"""Bounded development task for dependencies ambiguous to adjacent pair state."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Sequence

from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size

FAMILIES = (
    "delayed_parity",
    "nonadjacent_match",
    "three_event_composition",
    "branch_specific_conjunction",
    "pair_sufficient_control",
)
AMBIGUOUS_FAMILIES = FAMILIES[:3]


@dataclass(frozen=True)
class BeyondPairEvent:
    route: int
    branch: int = 0


@dataclass(frozen=True)
class BeyondPairEpisode:
    identity: str
    family: str
    events: tuple[BeyondPairEvent, ...]
    label: int


@dataclass(frozen=True)
class BeyondPairPrediction:
    owner: int
    predicted: int
    features: tuple[int, ...]
    work: int


def generate_beyond_pair(
    *, seeds: Sequence[int], count_per_family: int, split: str
) -> list[BeyondPairEpisode]:
    if split not in ("training", "development") or count_per_family < 2:
        raise ValueError("generator arguments are invalid")
    rows = []
    for seed in seeds:
        for family_index, family in enumerate(FAMILIES):
            for index in range(count_per_family):
                label = index % 2
                rng = random.Random(seed * 1_000_003 + family_index * 10_007 + index * 101)
                first = rng.randrange(2)
                distractor = 4 + rng.randrange(2)
                if family == "delayed_parity":
                    last = first ^ label
                    events = (BeyondPairEvent(first), BeyondPairEvent(distractor), BeyondPairEvent(last))
                elif family == "nonadjacent_match":
                    last = first if label else 1 - first
                    events = (BeyondPairEvent(first), BeyondPairEvent(distractor), BeyondPairEvent(last))
                elif family == "three_event_composition":
                    middle = rng.randrange(2)
                    last = first ^ middle ^ label
                    events = (BeyondPairEvent(first), BeyondPairEvent(2 + middle), BeyondPairEvent(last))
                elif family == "branch_specific_conjunction":
                    branches = (0, 1) if label else (1, 0)
                    events = (BeyondPairEvent(2, branches[0]), BeyondPairEvent(3, branches[1]))
                else:
                    events = (BeyondPairEvent(distractor), BeyondPairEvent(6 if label else 7))
                rows.append(BeyondPairEpisode(
                    f"event-unit-beyond-pair-v1:{split}:{seed}:{family}:{index}",
                    family,
                    events,
                    label,
                ))
    random.Random(sum(seeds) + (1 if split == "development" else 0)).shuffle(rows)
    return rows


class BeyondPairLearner:
    ARMS = ("P_temporal_pair", "T_bounded_triplet", "D_triplet_branch")
    INTERVENTIONS = ("none", "history_truncate", "event_order_shuffle", "distractor_shuffle", "branch_shuffle")

    def __init__(self, arm: str, *, intervention: str = "none", capacity_reserve_bytes: int = 0) -> None:
        if arm not in self.ARMS or intervention not in self.INTERVENTIONS:
            raise ValueError("arm or intervention is invalid")
        self.arm = arm
        self.intervention = intervention
        self.weights: dict[int, float] = {}
        self.feature_ids: dict[tuple, int] = {}
        self.pending: BeyondPairPrediction | None = None
        self.reserve = bytearray(capacity_reserve_bytes)

    def _feature(self, key: tuple) -> int:
        if key not in self.feature_ids:
            if len(self.feature_ids) >= 4096:
                raise ValueError("feature budget exceeded")
            self.feature_ids[key] = len(self.feature_ids)
        return self.feature_ids[key]

    def _encode(self, episode: BeyondPairEpisode) -> tuple[tuple[int, ...], int]:
        events = list(episode.events)
        rng = random.Random(sum(ord(char) for char in episode.identity) + 918_701)
        if self.intervention == "event_order_shuffle":
            rng.shuffle(events)
        if self.intervention == "distractor_shuffle":
            events = [BeyondPairEvent((9 + rng.randrange(8)) if event.route in (4, 5) else event.route, event.branch) for event in events]
        if self.intervention == "branch_shuffle":
            branches = [event.branch for event in events]; rng.shuffle(branches)
            events = [BeyondPairEvent(event.route, branches[index]) for index, event in enumerate(events)]
        keys = [("route", event.route) for event in events]
        keys.extend(("pair", events[index - 1].route, event.route) for index, event in enumerate(events) if index)
        use_triplet = self.arm != "P_temporal_pair" and self.intervention != "history_truncate"
        if use_triplet and len(events) >= 3:
            keys.extend(("triplet", events[index - 2].route, events[index - 1].route, event.route)
                        for index, event in enumerate(events) if index >= 2)
        if self.arm == "D_triplet_branch":
            keys.extend(("branch", events[index - 1].route, events[index - 1].branch, event.route, event.branch)
                        for index, event in enumerate(events) if index)
        features = tuple(dict.fromkeys(self._feature(key) for key in keys))
        return features, len(events) + len(keys)

    def predict(self, episode: BeyondPairEpisode) -> BeyondPairPrediction:
        if self.pending is not None:
            raise ValueError("resolve pending prediction first")
        features, work = self._encode(episode)
        score = sum(self.weights.get(feature, 0.0) for feature in features)
        receipt = BeyondPairPrediction(id(self), int(score > 0.0), features, work)
        self.pending = receipt
        return receipt

    def observe(self, receipt: BeyondPairPrediction, outcome: int) -> int:
        if receipt is not self.pending or receipt.owner != id(self) or outcome not in (0, 1):
            raise ValueError("invalid pending outcome")
        updates = 0
        if receipt.predicted != outcome:
            delta = (0.3 if outcome else -0.3) / len(receipt.features)
            for feature in receipt.features:
                self.weights[feature] = max(-2.0, min(2.0, self.weights.get(feature, 0.0) + delta))
                updates += 1
        self.pending = None
        return updates

    def state_bytes(self) -> int:
        return _deep_size((self.weights, self.feature_ids, self.reserve))


def run_beyond_pair(
    arm: str,
    training: Sequence[BeyondPairEpisode],
    development: Sequence[BeyondPairEpisode],
    *,
    intervention: str = "none",
    outcome_shuffle_seed: int | None = None,
    capacity_reserve_bytes: int = 0,
) -> dict:
    learner = BeyondPairLearner(arm, intervention=intervention, capacity_reserve_bytes=capacity_reserve_bytes)
    train_outcomes = [row.label for row in training]
    dev_outcomes = [row.label for row in development]
    if outcome_shuffle_seed is not None:
        random.Random(outcome_shuffle_seed).shuffle(train_outcomes)
        random.Random(outcome_shuffle_seed + 1).shuffle(dev_outcomes)
    updates = 0; max_work = 0
    for episode, outcome in zip(training, train_outcomes):
        receipt = learner.predict(episode); max_work = max(max_work, receipt.work)
        updates += learner.observe(receipt, outcome)
    totals = {family: 0 for family in FAMILIES}; correct = {family: 0 for family in FAMILIES}; rows = []
    for episode, outcome in zip(development, dev_outcomes):
        receipt = learner.predict(episode); max_work = max(max_work, receipt.work)
        totals[episode.family] += 1; correct[episode.family] += int(receipt.predicted == episode.label)
        rows.append((episode.identity, receipt.predicted)); updates += learner.observe(receipt, outcome)
    accuracy_by_family = {family: correct[family] / totals[family] for family in FAMILIES}
    return {
        "accuracy": sum(correct.values()) / len(development),
        "accuracy_by_family": accuracy_by_family,
        "ambiguous_accuracy": sum(correct[f] for f in AMBIGUOUS_FAMILIES) / sum(totals[f] for f in AMBIGUOUS_FAMILIES),
        "prediction_trace_sha256": hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest(),
        "prediction_rows": rows,
        "updates": updates,
        "maximum_event_work": max_work,
        "state_bytes": learner.state_bytes(),
        "feature_count": len(learner.feature_ids),
    }


__all__ = ["AMBIGUOUS_FAMILIES", "BeyondPairLearner", "FAMILIES", "generate_beyond_pair", "run_beyond_pair"]
