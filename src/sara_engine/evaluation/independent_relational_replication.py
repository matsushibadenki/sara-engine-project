"""Independent generator and local learners for relational replication v2."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import random
import sys
from typing import Sequence


CONTEXTS = (
    "ascending_or_equal",
    "same_parity",
    "bounded_jump",
    "interval_direction",
)


@dataclass(frozen=True)
class ReplicationEpisode:
    identity: str
    context: str
    values: tuple[int, ...]
    label: int


@dataclass(frozen=True)
class LocalReceipt:
    owner: int
    predicted: int
    feature: tuple[object, ...]
    work: int


def _generator_label(context: str, values: tuple[int, ...]) -> int:
    if context == "ascending_or_equal":
        return int(values[1] >= values[0])
    if context == "same_parity":
        return int(values[0] % 2 == values[1] % 2)
    if context == "bounded_jump":
        return int(abs(values[1] - values[0]) <= 2)
    if context == "interval_direction":
        return int(values[2] - values[1] > values[1] - values[0])
    raise ValueError(f"Unknown context: {context}")


def generate_independent_episodes(
    *,
    seeds: Sequence[int],
    values: Sequence[int],
    count_per_context: int,
    split: str,
    namespace: str,
) -> list[ReplicationEpisode]:
    """Generate balanced rows without importing the prior transition generator."""
    rows: list[ReplicationEpisode] = []
    half = count_per_context // 2
    if count_per_context % 2:
        raise ValueError("count_per_context must be even")
    for seed in seeds:
        for context_index, context in enumerate(CONTEXTS):
            width = 3 if context == "interval_direction" else 2
            pools = {0: [], 1: []}
            for candidate in itertools.product(values, repeat=width):
                pools[_generator_label(context, candidate)].append(candidate)
            rng = random.Random(seed * 1_000_003 + context_index * 10_007)
            selected: list[tuple[int, ...]] = []
            for label in (0, 1):
                pool = pools[label]
                rng.shuffle(pool)
                if split == "development":
                    if len(pool) < half:
                        raise ValueError("Insufficient unique development signatures")
                    selected.extend(pool[:half])
                else:
                    selected.extend(pool[index % len(pool)] for index in range(half))
            rng.shuffle(selected)
            for index, candidate in enumerate(selected):
                label = _generator_label(context, candidate)
                rows.append(
                    ReplicationEpisode(
                        f"{namespace}:{split}:{seed}:{context}:{index}",
                        context,
                        candidate,
                        label,
                    )
                )
    random.Random(sum(seeds) + (1 if split == "development" else 0)).shuffle(rows)
    return rows


class IndependentRelationalLearner:
    ARMS = ("categorical_zero_shot", "contextual_relational_zero_shot")

    def __init__(self, arm: str, *, intervention: str = "none", reserve: int = 0):
        if arm not in self.ARMS:
            raise ValueError("Unknown arm")
        if intervention not in (
            "none",
            "context_shuffle",
            "relation_shuffle",
            "context_relation_decouple",
        ):
            raise ValueError("Unknown intervention")
        self.arm = arm
        self.intervention = intervention
        self.weights: dict[tuple[object, ...], float] = {}
        self.pending: LocalReceipt | None = None
        self.reserve = bytearray(reserve)

    @staticmethod
    def _relation(values: tuple[int, ...]) -> tuple[object, ...]:
        first, second = values[:2]
        pair = (
            (second > first) - (second < first),
            int(first % 2 == second % 2),
            int(abs(second - first) <= 2),
        )
        if len(values) == 3:
            return pair + ((values[2] - values[1] > values[1] - values[0]),)
        return pair

    def predict(self, episode: ReplicationEpisode) -> LocalReceipt:
        if self.pending is not None:
            raise ValueError("Pending receipt must be observed")
        context = episode.context
        relation = self._relation(episode.values)
        rng = random.Random(sum(map(ord, episode.identity)) + 933_701)
        if self.intervention == "context_shuffle":
            context = CONTEXTS[rng.randrange(len(CONTEXTS))]
        if self.intervention == "relation_shuffle":
            relation = tuple(rng.randrange(2) for _ in relation)
        if self.arm == "categorical_zero_shot":
            feature: tuple[object, ...] = ("categorical", context, *episode.values)
        elif self.intervention == "context_relation_decouple":
            feature = ("relation_only", *relation)
        else:
            feature = ("context_relation", context, *relation)
        score = self.weights.get(feature, 0.0)
        receipt = LocalReceipt(id(self), int(score > 0.0), feature, len(episode.values) + 1)
        self.pending = receipt
        return receipt

    def observe(self, receipt: LocalReceipt, outcome: int) -> int:
        if receipt is not self.pending or receipt.owner != id(self):
            raise ValueError("Invalid receipt")
        changed = 0
        if receipt.predicted != outcome:
            delta = 1.0 if outcome else -1.0
            self.weights[receipt.feature] = max(
                -2.0, min(2.0, self.weights.get(receipt.feature, 0.0) + delta)
            )
            changed = 1
        self.pending = None
        return changed

    def state_bytes(self) -> int:
        return (
            sys.getsizeof(self.weights)
            + sum(sys.getsizeof(key) + sys.getsizeof(value) for key, value in self.weights.items())
            + sys.getsizeof(self.reserve)
        )


def run_independent_arm(
    arm: str,
    training: Sequence[ReplicationEpisode],
    development: Sequence[ReplicationEpisode],
    *,
    intervention: str = "none",
    outcome_shuffle_seed: int | None = None,
    reserve: int = 0,
) -> dict[str, object]:
    learner = IndependentRelationalLearner(arm, intervention=intervention, reserve=reserve)
    outcomes = [episode.label for episode in training]
    if outcome_shuffle_seed is not None:
        random.Random(outcome_shuffle_seed).shuffle(outcomes)
    updates = 0
    maximum_work = 0
    for episode, outcome in zip(training, outcomes):
        receipt = learner.predict(episode)
        maximum_work = max(maximum_work, receipt.work)
        updates += learner.observe(receipt, outcome)
    correct = {context: 0 for context in CONTEXTS}
    totals = {context: 0 for context in CONTEXTS}
    trace: list[tuple[str, int]] = []
    for episode in development:
        receipt = learner.predict(episode)
        maximum_work = max(maximum_work, receipt.work)
        correct[episode.context] += int(receipt.predicted == episode.label)
        totals[episode.context] += 1
        trace.append((episode.identity, receipt.predicted))
        learner.observe(receipt, receipt.predicted)
    return {
        "accuracy": sum(correct.values()) / sum(totals.values()),
        "accuracy_by_context": {
            context: correct[context] / totals[context] for context in CONTEXTS
        },
        "trace_sha256": hashlib.sha256(
            json.dumps(trace, separators=(",", ":")).encode()
        ).hexdigest(),
        "updates": updates,
        "development_updates": 0,
        "feature_count": len(learner.weights),
        "state_bytes": learner.state_bytes(),
        "maximum_event_work": maximum_work,
    }


__all__ = [
    "CONTEXTS",
    "IndependentRelationalLearner",
    "ReplicationEpisode",
    "generate_independent_episodes",
    "run_independent_arm",
]
