"""Frozen R1 causal temporal-learning benchmark.

The benchmark is intentionally small, event-driven, CPU-only, and free of matrix
operations or global backpropagation. Outcomes are supplied only after prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sara_engine.learning.three_factor_learning import ThreeFactorLearningManager
from sara_engine.neuro.neuron import Neuron


PRIMARY_FAMILIES = (
    "ordered_cue_short_gap",
    "ordered_cue_long_gap",
    "same_multiset_reversed_order",
    "interval_coded_cue",
    "distractor_resistant_cue",
    "controlled_rule_reversal",
)
CONTROL_FAMILY = "timing_irrelevant_control"


@dataclass(frozen=True)
class Event:
    symbol: int
    time: int


@dataclass(frozen=True)
class Episode:
    identity: str
    family: str
    events: Tuple[Event, ...]
    label: int
    reversal: bool = False


def _permutation(seed: int) -> Dict[int, int]:
    values = list(range(12))
    random.Random(seed).shuffle(values)
    return {source: target for source, target in enumerate(values)}


def generate_episode(
    *, seed: int, index: int, family: str, generator_family: str, split: str
) -> Episode:
    rng = random.Random((seed * 1_000_003) + (index * 101) + sum(ord(c) for c in family))
    label = rng.randrange(2)
    cue_a, cue_b = (0, 1) if label else (1, 0)
    gap = 2
    if family == "ordered_cue_long_gap":
        gap = 10
    elif family == "interval_coded_cue":
        cue_a, cue_b = 0, 1
        gap = 9 if label else 3
    elif family == "same_multiset_reversed_order":
        gap = 5
    elif family == "distractor_resistant_cue":
        gap = 7
    elif family == CONTROL_FAMILY:
        cue_a, cue_b = (10, 2) if label else (11, 2)
        gap = rng.choice((2, 5, 9))

    reversal = family == "controlled_rule_reversal" and split == "evaluation" and index >= 120
    if reversal:
        label = 1 - label

    start = rng.randint(1, 3)
    events = [Event(cue_a, start)]
    distractor_count = rng.randint(1, 4) if family == "distractor_resistant_cue" else rng.randint(0, 2)
    occupied = set()
    for _ in range(distractor_count):
        if gap <= 2:
            break
        offset = rng.randint(1, gap - 1)
        if offset not in occupied:
            occupied.add(offset)
            events.append(Event(rng.randint(2, 9), start + offset))
    events.append(Event(cue_b, start + gap))
    events.sort(key=lambda event: (event.time, event.symbol))

    if generator_family == "heldout_symbol_permutation":
        mapping = _permutation(seed + 17_003)
        events = [Event(mapping[event.symbol], event.time) for event in events]
    elif generator_family == "heldout_interval_offset":
        events = [Event(event.symbol, event.time + 3) for event in events]

    identity = f"{split}:{generator_family}:{seed}:{family}:{index}"
    return Episode(identity, family, tuple(events), label, reversal)


def generate_stream(
    *, seeds: Sequence[int], count_per_seed: int, generator_family: str, split: str
) -> List[Episode]:
    families = PRIMARY_FAMILIES + (CONTROL_FAMILY,)
    return [
        generate_episode(
            seed=seed,
            index=index,
            family=families[index % len(families)],
            generator_family=generator_family,
            split=split,
        )
        for seed in seeds
        for index in range(count_per_seed)
    ]


def destroy_timing(episode: Episode) -> Episode:
    events = tuple(Event(event.symbol, index + 1) for index, event in enumerate(episode.events))
    return Episode(episode.identity, episode.family, events, episode.label, episode.reversal)


class BoundedLearner:
    def predict(self, episode: Episode) -> Tuple[int, int]:
        raise NotImplementedError

    def learn(self, episode: Episode, outcome: int) -> int:
        raise NotImplementedError

    def state_bytes(self) -> int:
        return _deep_size(self.__dict__)


class SpikingLocalLearner(BoundedLearner):
    """Fixed coincidence topology with local eligibility and outcome modulation."""

    def __init__(self, *, plastic: bool = True, destroy_event_timing: bool = False) -> None:
        self.plastic = plastic
        self.destroy_event_timing = destroy_event_timing
        self.units = [Neuron(index, num_branches=2) for index in range(24)]
        self.weights: Dict[int, float] = {}
        self.max_readout_entries = 96
        self.manager = ThreeFactorLearningManager(
            lr=0.02,
            trace_decay=0.9,
            baseline_decay=0.99,
            use_rpe=False,
            max_traces=256,
            max_trace_age=64.0,
        )
        self._active_features: Tuple[int, ...] = ()
        self._episode_end_time = 0.0

    @staticmethod
    def _feature_id(left: int, right: int, delta: int) -> int:
        time_bin = 0 if delta <= 3 else (1 if delta <= 7 else 2)
        return ((left * 13 + right) * 3 + time_bin) % 24

    def _encode(self, episode: Episode) -> Tuple[Tuple[int, ...], int]:
        current = destroy_timing(episode) if self.destroy_event_timing else episode
        self.manager.reset()
        features: List[int] = []
        work = len(current.events)
        for right_index, right in enumerate(current.events):
            for left in current.events[max(0, right_index - 8):right_index]:
                delta = right.time - left.time
                if delta <= 0 or delta > 12:
                    continue
                feature_id = self._feature_id(left.symbol, right.symbol, delta)
                unit = self.units[feature_id]
                decay = math.exp(-delta / 8.0)
                unit.add_input_to_branch(0, 0.8)
                unit.add_input_to_branch(0, 0.8 * decay)
                work += 4
                if unit.step():
                    features.append(feature_id)
                    self.manager.update_trace(feature_id, 0, decay, float(right.time))
                    work += self.manager.last_update_event_cost
        self._active_features = tuple(dict.fromkeys(features))
        self._episode_end_time = float(current.events[-1].time if current.events else 0)
        return self._active_features, work

    def predict(self, episode: Episode) -> Tuple[int, int]:
        features, work = self._encode(episode)
        score = sum(self.weights.get(feature, 0.0) for feature in features)
        return (1 if score > 0.0 else 0), work + len(features)

    def learn(self, episode: Episode, outcome: int) -> int:
        if not self.plastic:
            return 0
        signed_outcome = 1.0 if outcome else -1.0
        updates = self.manager.apply_reward(signed_outcome, time=self._episode_end_time)
        for (feature_id, _), delta in updates.items():
            if feature_id not in self.weights and len(self.weights) >= self.max_readout_entries:
                victim = min(self.weights, key=lambda key: (abs(self.weights[key]), key))
                del self.weights[victim]
            self.weights[feature_id] = max(-2.0, min(2.0, self.weights.get(feature_id, 0.0) + delta))
        return self.manager.last_reward_event_cost + len(updates)


class ExactSparseSpikingLearner(BoundedLearner):
    """Exact fixed coincidence addresses evaluated through one sparse active unit."""

    def __init__(self, *, plastic: bool = True, destroy_event_timing: bool = False) -> None:
        self.plastic = plastic
        self.destroy_event_timing = destroy_event_timing
        self.active_unit = Neuron(0, num_branches=2)
        self.weights: Dict[int, float] = {}
        self.max_readout_entries = 96
        self.manager = ThreeFactorLearningManager(
            lr=0.02,
            trace_decay=0.9,
            baseline_decay=0.99,
            use_rpe=False,
            max_traces=256,
            max_trace_age=64.0,
        )
        self._active_feature: Optional[int] = None
        self._episode_end_time = 0.0

    @staticmethod
    def _feature_id(left: int, right: int, delta: int) -> int:
        time_bin = 0 if delta <= 3 else (1 if delta <= 7 else 2)
        return (left * 12 + right) * 3 + time_bin

    def _reset_active_unit(self, feature_id: int) -> None:
        self.active_unit.id = feature_id
        self.active_unit.v = 0.0
        self.active_unit.spike = False
        self.active_unit.refractory_time = 0
        self.active_unit.active_branches.clear()
        for branch in self.active_unit.branches:
            branch.current_input = 0.0
            branch.is_active = False

    def predict(self, episode: Episode) -> Tuple[int, int]:
        current = destroy_timing(episode) if self.destroy_event_timing else episode
        self.manager.reset()
        self._active_feature = None
        if len(current.events) < 2:
            return 0, len(current.events)
        left, right = current.events[0], current.events[-1]
        delta = right.time - left.time
        if delta <= 0 or delta > 12:
            return 0, len(current.events) + 1
        feature_id = self._feature_id(left.symbol, right.symbol, delta)
        self._reset_active_unit(feature_id)
        decay = math.exp(-delta / 8.0)
        self.active_unit.add_input_to_branch(0, 0.8)
        self.active_unit.add_input_to_branch(0, 0.8 * decay)
        work = len(current.events) + 5
        if self.active_unit.step():
            self._active_feature = feature_id
            self.manager.update_trace(feature_id, 0, decay, float(right.time))
            work += self.manager.last_update_event_cost
        self._episode_end_time = float(right.time)
        score = self.weights.get(feature_id, 0.0)
        return (1 if score > 0.0 else 0), work + 1

    def learn(self, episode: Episode, outcome: int) -> int:
        if not self.plastic or self._active_feature is None:
            return 0
        signed_outcome = 1.0 if outcome else -1.0
        updates = self.manager.apply_reward(signed_outcome, time=self._episode_end_time)
        for (feature_id, _), delta in updates.items():
            if feature_id not in self.weights and len(self.weights) >= self.max_readout_entries:
                victim = min(self.weights, key=lambda key: (abs(self.weights[key]), key))
                del self.weights[victim]
            self.weights[feature_id] = max(-2.0, min(2.0, self.weights.get(feature_id, 0.0) + delta))
        return self.manager.last_reward_event_cost + len(updates)


class TransitionMemoryLearner(BoundedLearner):
    def __init__(self) -> None:
        self.counts: Dict[Tuple[int, int], List[int]] = {}
        self._key: Optional[Tuple[int, int]] = None

    def predict(self, episode: Episode) -> Tuple[int, int]:
        symbols = [event.symbol for event in episode.events]
        self._key = (symbols[0], symbols[-1]) if symbols else (0, 0)
        counts = self.counts.get(self._key, [0, 0])
        return (1 if counts[1] > counts[0] else 0), len(symbols) + 1

    def learn(self, episode: Episode, outcome: int) -> int:
        assert self._key is not None
        if self._key not in self.counts and len(self.counts) >= 48:
            victim = min(self.counts, key=lambda key: (sum(self.counts[key]), key))
            del self.counts[victim]
        self.counts.setdefault(self._key, [0, 0])[outcome] += 1
        return 1


class NonspikingTemporalLearner(BoundedLearner):
    """Scalar temporal features with local readout updates and equal raw history."""

    def __init__(self) -> None:
        self.weights: Dict[int, float] = {}
        self._features: Tuple[int, ...] = ()

    @staticmethod
    def _features_for(episode: Episode) -> Tuple[int, ...]:
        if not episode.events:
            return ()
        first, last = episode.events[0], episode.events[-1]
        duration_bin = 0 if last.time - first.time <= 3 else (1 if last.time - first.time <= 7 else 2)
        values = [first.symbol, 12 + last.symbol, 24 + duration_bin]
        values.extend(27 + event.symbol for event in episode.events[-3:])
        return tuple(dict.fromkeys(values))

    def predict(self, episode: Episode) -> Tuple[int, int]:
        self._features = self._features_for(episode)
        score = sum(self.weights.get(feature, 0.0) for feature in self._features)
        return (1 if score > 0.0 else 0), len(episode.events) + len(self._features)

    def learn(self, episode: Episode, outcome: int) -> int:
        signed_outcome = 1.0 if outcome else -1.0
        for feature in self._features:
            if feature not in self.weights and len(self.weights) >= 96:
                victim = min(self.weights, key=lambda key: (abs(self.weights[key]), key))
                del self.weights[victim]
            self.weights[feature] = max(-2.0, min(2.0, self.weights.get(feature, 0.0) + 0.05 * signed_outcome))
        return len(self._features)


def _deep_size(value: Any, seen: Optional[set[int]] = None) -> int:
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_deep_size(key, seen) + _deep_size(item, seen) for key, item in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(_deep_size(item, seen) for item in value)
    elif hasattr(value, "__dict__"):
        size += _deep_size(value.__dict__, seen)
    return size


def _arm_factory(name: str, *, candidate_version: int = 1) -> BoundedLearner:
    spiking_type = ExactSparseSpikingLearner if candidate_version == 2 else SpikingLocalLearner
    if name == "intact_spiking_local":
        return spiking_type()
    if name == "frozen_spiking":
        return spiking_type(plastic=False)
    if name == "shuffled_outcome_feedback":
        return spiking_type()
    if name == "timing_destroyed_spiking":
        return spiking_type(destroy_event_timing=True)
    if name == "bounded_transition_memory":
        return TransitionMemoryLearner()
    if name == "nonspiking_temporal_state":
        return NonspikingTemporalLearner()
    raise ValueError(f"Unknown arm: {name}")


def run_arm(
    name: str,
    training: Sequence[Episode],
    evaluation: Sequence[Episode],
    budgets: Mapping[str, float],
    *,
    feedback_seed: int,
    candidate_version: int = 1,
) -> Dict[str, Any]:
    learner = _arm_factory(name, candidate_version=candidate_version)
    shuffled_training = [episode.label for episode in training]
    shuffled_evaluation = [episode.label for episode in evaluation]
    feedback_rng = random.Random(feedback_seed)
    feedback_rng.shuffle(shuffled_training)
    feedback_rng.shuffle(shuffled_evaluation)

    for index, episode in enumerate(training):
        learner.predict(episode)
        outcome = shuffled_training[index] if name == "shuffled_outcome_feedback" else episode.label
        learner.learn(episode, outcome)

    records = []
    max_state_bytes = learner.state_bytes()
    max_event_work = 0
    max_cpu_ms = 0.0
    for index, episode in enumerate(evaluation):
        started = time.perf_counter_ns()
        prediction, prediction_work = learner.predict(episode)
        outcome = shuffled_evaluation[index] if name == "shuffled_outcome_feedback" else episode.label
        learning_work = learner.learn(episode, outcome)
        cpu_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        event_work = prediction_work + learning_work
        state_bytes = learner.state_bytes()
        max_state_bytes = max(max_state_bytes, state_bytes)
        max_event_work = max(max_event_work, event_work)
        max_cpu_ms = max(max_cpu_ms, cpu_ms)
        records.append({
            "identity": episode.identity,
            "family": episode.family,
            "label": episode.label,
            "prediction": prediction,
            "correct": int(prediction == episode.label),
            "reversal": episode.reversal,
            "event_work": event_work,
        })

    primary = [record for record in records if record["family"] != CONTROL_FAMILY]
    control = [record for record in records if record["family"] == CONTROL_FAMILY]
    post_reversal = [record for record in records if record["reversal"]]
    correct = sum(record["correct"] for record in records)
    resource_pass = (
        max_state_bytes <= budgets["max_state_bytes"]
        and max_event_work <= budgets["max_event_work_per_episode"]
        and max_cpu_ms <= budgets["max_cpu_ms_per_episode"]
    )
    return {
        "arm": name,
        "records": records,
        "metrics": {
            "primary_temporal_accuracy": _mean(record["correct"] for record in primary),
            "timing_irrelevant_accuracy": _mean(record["correct"] for record in control),
            "correction_latency_episodes": _correction_latency(records),
            "old_rule_interference_rate": 1.0 - _mean(record["correct"] for record in post_reversal),
            "answered_coverage": 1.0,
            "answered_error_rate": 1.0 - _mean(record["correct"] for record in records),
            "work_per_correct_prediction": sum(record["event_work"] for record in records) / max(1, correct),
            "resource_contract_pass_rate": 1.0 if resource_pass else 0.0,
        },
        "resources": {
            "max_state_bytes": max_state_bytes,
            "max_event_work_per_episode": max_event_work,
            "max_cpu_ms_per_episode": max_cpu_ms,
            "contracts_passed": resource_pass,
        },
    }


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _correction_latency(records: Sequence[Mapping[str, Any]]) -> float:
    groups: Dict[str, List[int]] = {}
    for record in records:
        if record["family"] != "controlled_rule_reversal":
            continue
        parts = str(record["identity"]).split(":")
        group = ":".join(parts[:3])
        if record["reversal"]:
            groups.setdefault(group, []).append(int(record["correct"]))
    latencies = []
    for values in groups.values():
        latency = len(values) + 1
        for index in range(19, len(values)):
            if sum(values[index - 19:index + 1]) / 20.0 >= 0.75:
                latency = index + 1
                break
        latencies.append(latency)
    return _mean(latencies)


def paired_bootstrap_interval(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> Dict[str, float]:
    differences = [
        left["correct"] - right["correct"]
        for left, right in zip(candidate, baseline)
        if left["family"] != CONTROL_FAMILY and right["family"] != CONTROL_FAMILY
    ]
    if not differences:
        return {"gain": 0.0, "lower": 0.0, "upper": 0.0, "one_sided_p": 1.0}
    counts = {value: differences.count(value) for value in (-1, 0, 1)}
    population = (-1, 0, 1)
    weights = tuple(counts[value] for value in population)
    rng = random.Random(seed)
    means = []
    nonpositive = 0
    for _ in range(resamples):
        total = sum(rng.choices(population, weights=weights, k=len(differences)))
        mean = total / len(differences)
        means.append(mean)
        if mean <= 0.0:
            nonpositive += 1
    means.sort()
    lower_index = int(0.025 * (resamples - 1))
    upper_index = int(0.975 * (resamples - 1))
    return {
        "gain": sum(differences) / len(differences),
        "lower": means[lower_index],
        "upper": means[upper_index],
        "one_sided_p": (nonpositive + 1) / (resamples + 1),
    }


def holm_adjust(comparisons: Mapping[str, Mapping[str, float]]) -> Dict[str, Dict[str, Any]]:
    ordered = sorted(comparisons, key=lambda name: comparisons[name]["one_sided_p"])
    adjusted: Dict[str, Dict[str, Any]] = {}
    running = 0.0
    count = len(ordered)
    continue_rejecting = True
    for rank, name in enumerate(ordered):
        raw = float(comparisons[name]["one_sided_p"])
        running = max(running, min(1.0, raw * (count - rank)))
        threshold = 0.05 / (count - rank)
        rejected = continue_rejecting and raw <= threshold
        if not rejected:
            continue_rejecting = False
        adjusted[name] = {
            "raw_one_sided_p": raw,
            "holm_adjusted_p": running,
            "rejected_at_0_05": rejected,
        }
    return adjusted


def run_benchmark(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    counts = manifest["episode_counts"]
    training = generate_stream(
        seeds=manifest["training_seeds"],
        count_per_seed=counts["training_per_seed"],
        generator_family="development_base",
        split="training",
    )
    evaluation = []
    for family in manifest["heldout_generator_families"]:
        evaluation.extend(generate_stream(
            seeds=manifest["evaluation_seeds"],
            count_per_seed=counts["evaluation_per_seed_per_heldout_family"],
            generator_family=family,
            split="evaluation",
        ))

    candidate_version = 2 if str(manifest["experiment_id"]).endswith("hypothesis-2") else 1
    arms = {
        name: run_arm(
            name,
            training,
            evaluation,
            manifest["budgets"],
            feedback_seed=manifest["statistics"]["bootstrap_seed"] + index,
            candidate_version=candidate_version,
        )
        for index, name in enumerate(manifest["arms"])
    }
    candidate_records = arms["intact_spiking_local"]["records"]
    strongest_nonspiking = max(
        ("bounded_transition_memory", "nonspiking_temporal_state"),
        key=lambda name: arms[name]["metrics"]["primary_temporal_accuracy"],
    )
    comparison_names = {
        "frozen_spiking": "frozen_spiking",
        "shuffled_outcome_feedback": "shuffled_outcome_feedback",
        "strongest_nonspiking": strongest_nonspiking,
    }
    comparisons = {
        label: paired_bootstrap_interval(
            candidate_records,
            arms[name]["records"],
            seed=manifest["statistics"]["bootstrap_seed"] + index,
            resamples=manifest["statistics"]["bootstrap_resamples"],
        )
        for index, (label, name) in enumerate(comparison_names.items())
    }
    holm = holm_adjust(comparisons)
    threshold = manifest["decision_threshold"]["minimum_gain_percentage_points"] / 100.0
    timing_loss = (
        arms["intact_spiking_local"]["metrics"]["primary_temporal_accuracy"]
        - arms["timing_destroyed_spiking"]["metrics"]["primary_temporal_accuracy"]
    )
    control_loss = (
        arms["intact_spiking_local"]["metrics"]["timing_irrelevant_accuracy"]
        - arms["timing_destroyed_spiking"]["metrics"]["timing_irrelevant_accuracy"]
    )
    comparison_pass = all(
        comparisons[name]["gain"] >= threshold
        and comparisons[name]["lower"] > 0.0
        and holm[name]["rejected_at_0_05"]
        for name in comparisons
    )
    resource_pass = all(arm["resources"]["contracts_passed"] for arm in arms.values())
    decision = (
        comparison_pass
        and timing_loss >= threshold
        and control_loss < threshold
        and resource_pass
    )
    return {
        "schema": "sara-r1-causal-temporal-learning-result-v1",
        "experiment_id": manifest["experiment_id"],
        "protocol_fingerprint": manifest["protocol_fingerprint"],
        "evaluation_run": 1,
        "training_episodes": len(training),
        "independent_evaluation_episodes": len(evaluation),
        "strongest_nonspiking_arm": strongest_nonspiking,
        "arms": {name: {key: value for key, value in result.items() if key != "records"} for name, result in arms.items()},
        "comparisons": comparisons,
        "holm_multiplicity": holm,
        "timing_ablation": {"primary_loss": timing_loss, "control_loss": control_loss},
        "decision": {
            f"r1_hypothesis_{candidate_version}_passed": decision,
            "comparison_thresholds_passed": comparison_pass,
            "timing_causality_passed": timing_loss >= threshold and control_loss < threshold,
            "resource_contracts_passed": resource_pass,
            "status": "pass" if decision else "negative_result_retained",
        },
    }


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = [
    "CONTROL_FAMILY", "Episode", "Event", "PRIMARY_FAMILIES",
    "ExactSparseSpikingLearner", "SpikingLocalLearner", "destroy_timing",
    "generate_episode", "generate_stream", "holm_adjust", "load_manifest", "paired_bootstrap_interval",
    "run_arm", "run_benchmark",
]
