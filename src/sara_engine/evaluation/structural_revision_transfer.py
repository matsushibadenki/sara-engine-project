"""Frozen held-out-composition benchmark for revision-specific local gain."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Iterable, Optional

from sara_engine.evaluation.local_outcome_snn_integration import interval_bin, route_id
from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.learning.observable_revision import BoundedObservableRevisionReadout, ObservableRevisionConfig
from sara_engine.learning.revision_gain import BoundedRevisionGainReadout, RevisionGainConfig
from sara_engine.neuro.neuron import Neuron
from sara_engine.utils.project_paths import processed_data_path


PROTOCOL_PATH = processed_data_path("benchmark_fixtures", "structural_revision_transfer_v1.json")
PROTOCOL_SHA256 = "d0767227fb327a2f31d202fee29c8aeebf0cffda88ef074be520987d29cfe522"


def load_protocol() -> dict:
    raw = Path(PROTOCOL_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen structural revision transfer protocol changed")
    return json.loads(raw)


@dataclass(frozen=True)
class TransferEpisode:
    identity: str
    split: str
    left: int
    right: int
    interval: int
    label: int
    revision: int
    revision_notice: bool = False


def shared_routes(episode: TransferEpisode, symbols: int) -> tuple[int, int, int]:
    return episode.left, symbols + episode.right, 2 * symbols + interval_bin(episode.interval)


def _addresses(protocol: dict) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    seen, held_out = [], []
    for left in range(protocol["symbols"]):
        for right in range(protocol["symbols"]):
            for interval_index, gap in enumerate(protocol["intervals"]):
                target = held_out if (left + 2 * right + interval_index) % 4 == 0 else seen
                target.append((left, right, gap))
    if len(seen) + len(held_out) != protocol["address_space"]:
        raise ValueError("Address space does not match frozen protocol")
    return seen, held_out


def _balanced_signs(size: int, rng: random.Random) -> list[int]:
    signs = [-1 if index < size // 2 else 1 for index in range(size)]
    rng.shuffle(signs)
    return signs


def make_streams(protocol: dict, seed: int) -> tuple[list[TransferEpisode], list[TransferEpisode], list[TransferEpisode], list[TransferEpisode]]:
    seen, held_out = _addresses(protocol)
    rng = random.Random(seed)
    left_sign = _balanced_signs(protocol["symbols"], rng)
    right_sign = _balanced_signs(protocol["symbols"], rng)
    interval_sign = _balanced_signs(len(protocol["intervals"]), rng)

    def label(address: tuple[int, int, int], revision: int) -> int:
        left, right, gap = address
        total = left_sign[left] + right_sign[right] + interval_sign[interval_bin(gap)]
        return int((total if revision == 1 else -total) > 0)

    def build(split: str, addresses: list[tuple[int, int, int]], count: int, offset: int, revision: int, notice: bool = False) -> list[TransferEpisode]:
        order_rng = random.Random(seed + offset)
        result = []
        rounds, remainder = divmod(count, len(addresses))
        for round_index in range(rounds + int(remainder > 0)):
            epoch = list(addresses)
            order_rng.shuffle(epoch)
            chosen = epoch[:remainder] if round_index == rounds else epoch
            result.extend(chosen)
        return [TransferEpisode(
            f"{split}:{seed}:{index}", split, *address, label(address, revision), revision,
            notice and index == 0,
        ) for index, address in enumerate(result)]

    return (
        build("training", seen, protocol["training_episodes"], 71_003, 1),
        build("seen_pre", seen, protocol["seen_pre_revision_evaluation_episodes"], 73_019, 1),
        build("held_out_pre", held_out, protocol["held_out_pre_revision_evaluation_episodes"], 79_007, 1),
        build("held_out_post", held_out, protocol["held_out_post_revision_evaluation_episodes"], 83_017, 2, True),
    )


class SharedFeatureEncoder:
    """Fixed sparse feature encoder backed by explicit event-driven neurons."""

    def __init__(self, protocol: dict) -> None:
        config = protocol["configuration"]
        self.symbols = protocol["symbols"]
        self.branch_inputs = tuple(float(value) for value in config["branch_inputs"])
        self.units = [Neuron(index, num_branches=config["branches_per_neuron"])
                      for index in range(protocol["feature_route_count"])]

    @staticmethod
    def _reset(unit: Neuron) -> None:
        unit.v = 0.0
        unit.spike = False
        unit.refractory_time = 0
        unit.active_branches.clear()
        for branch in unit.branches:
            branch.current_input = 0.0
            branch.is_active = False

    def encode(self, episode: TransferEpisode) -> tuple[tuple[int, float], ...]:
        active = []
        for route in shared_routes(episode, self.symbols):
            unit = self.units[route]
            self._reset(unit)
            for value in self.branch_inputs:
                unit.add_input_to_branch(0, value)
            if not unit.step():
                raise RuntimeError("Frozen shared-feature input did not produce a spike")
            active.append((route, 1.0))
        return tuple(active)


class StaticSparseReadout:
    def __init__(self, protocol: dict, rule: str) -> None:
        if rule not in ("three_factor", "residual"):
            raise ValueError("Unknown static rule")
        config = protocol["configuration"]
        self.rule = rule
        self.learning_rate = config["stable_learning_rate"]
        self.trace_decay = config["trace_decay"]
        self.max_age = config["max_feedback_age"]
        self.max_delta = config["max_delta"]
        self.max_routes = config["max_shared_readout_routes"]
        self.weights: dict[int, float] = {}
        self.pending: Optional[tuple[tuple[tuple[int, float], ...], int, float]] = None

    def predict(self, active: tuple[tuple[int, float], ...], *, time_value: int) -> float:
        if self.pending is not None:
            raise ValueError("Resolve pending static prediction")
        additions = sum(route not in self.weights for route, _ in active)
        if len(self.weights) + additions > self.max_routes:
            raise ValueError("Static route budget exceeded")
        total = sum(value for _, value in active)
        score = sum(self.weights.get(route, 0.0) * value for route, value in active) / total
        self.pending = (active, time_value, score)
        return score

    def observe(self, target: float, *, time_value: int) -> int:
        if self.pending is None:
            raise ValueError("No pending static prediction")
        active, predicted_at, score = self.pending
        self.pending = None
        age = time_value - predicted_at
        if age > self.max_age:
            return 0
        total = sum(value for _, value in active)
        signal = target - score if self.rule == "residual" else target
        for route, eligibility in active:
            delta = max(-self.max_delta, min(self.max_delta,
                self.learning_rate * self.trace_decay ** age * eligibility / total * signal))
            self.weights[route] = max(-1.0, min(1.0, self.weights.get(route, 0.0) + delta))
        return len(active)


class TransferArm:
    def __init__(self, protocol: dict, arm: str) -> None:
        if arm not in protocol["arms"]:
            raise ValueError("Unknown transfer arm")
        self.protocol = protocol
        self.arm = arm
        self.atomic = arm == "revision_gain_atomic_snn"
        self.encoder = None if arm == "revision_gain_shared_scalar" else SharedFeatureEncoder(protocol)
        config = protocol["configuration"]
        max_routes = config["max_atomic_readout_routes"] if self.atomic else config["max_shared_readout_routes"]
        self.gain: Optional[BoundedRevisionGainReadout] = None
        self.previous: Optional[BoundedObservableRevisionReadout] = None
        self.static: Optional[StaticSparseReadout] = None
        if arm in ("revision_gain_shared_snn", "revision_gain_shared_scalar", "revision_gain_atomic_snn", "shuffled_feedback_shared_snn"):
            self.gain = BoundedRevisionGainReadout(RevisionGainConfig(
                stable_learning_rate=config["stable_learning_rate"],
                adaptive_learning_rate=config["candidate_adaptive_learning_rate"],
                trace_decay=config["trace_decay"], max_feedback_age=config["max_feedback_age"],
                adaptation_horizon=protocol["adaptation_horizon"], max_routes=max_routes,
                max_active=1 if self.atomic else config["max_active_routes"], max_delta=config["max_delta"],
            ))
        elif arm == "previous_policy_shared_snn":
            self.previous = BoundedObservableRevisionReadout(ObservableRevisionConfig(
                learning_rate=config["previous_adaptive_learning_rate"], trace_decay=config["trace_decay"],
                max_feedback_age=config["max_feedback_age"], adaptation_horizon=protocol["adaptation_horizon"],
                max_routes=max_routes, max_active=config["max_active_routes"], max_delta=config["max_delta"],
            ))
        else:
            rule = "residual" if arm == "always_residual_shared_snn" else "three_factor"
            self.static = StaticSparseReadout(protocol, rule)
        self.receipt = None

    def _active(self, episode: TransferEpisode) -> tuple[tuple[int, float], ...]:
        if self.atomic:
            return ((route_id(episode.left, episode.right, episode.interval, self.protocol["symbols"]), 1.0),)
        if self.encoder is None:
            return tuple((route, 1.0) for route in shared_routes(episode, self.protocol["symbols"]))
        return self.encoder.encode(episode)

    def notify(self, revision: int, *, time_value: int) -> int:
        if self.gain is not None:
            self.gain.notify_revision(revision, time=time_value)
        elif self.previous is not None:
            self.previous.notify_revision(revision, time=time_value)
        return 1

    def predict(self, episode: TransferEpisode, *, time_value: int) -> tuple[float, str, int]:
        active = self._active(episode)
        encoder_work = 1 if self.encoder is None else len(active) * (len(self.encoder.branch_inputs) + 2)
        if self.gain is not None:
            self.receipt = self.gain.predict(active, time=time_value)
            return self.receipt.score, self.receipt.mode, encoder_work + len(active)
        if self.previous is not None:
            self.receipt = self.previous.predict(active, time=time_value)
            return self.receipt.score, self.receipt.mode, encoder_work + len(active)
        assert self.static is not None
        return self.static.predict(active, time_value=time_value), self.static.rule, encoder_work + len(active)

    def observe(self, target: float, *, time_value: int) -> int:
        if self.gain is not None:
            result = self.gain.observe(self.receipt, target, time=time_value)
            self.receipt = None
            return len(result.deltas)
        if self.previous is not None:
            result = self.previous.observe(self.receipt, target, time=time_value)
            self.receipt = None
            return len(result.deltas)
        assert self.static is not None
        return self.static.observe(target, time_value=time_value)


def _metrics(records: list[dict]) -> dict:
    return {
        "accuracy": sum(row["correct"] for row in records) / len(records),
        "brier": sum(row["brier"] for row in records) / len(records),
        "count": len(records),
    }


def _latency(records: list[dict]) -> int:
    window = 96
    for index in range(window - 1, len(records)):
        if sum(row["correct"] for row in records[index - window + 1:index + 1]) / window >= 0.90:
            return index + 1
    return len(records) + 1


def run_arm(protocol: dict, arm: str, streams: tuple[list[TransferEpisode], ...], seed: int) -> dict:
    learner = TransferArm(protocol, arm)
    episodes = [episode for stream in streams for episode in stream]
    feedback = [episode.label for episode in episodes]
    random.Random(seed + 89_009).shuffle(feedback)
    records: dict[str, list[dict]] = {name: [] for name in ("seen_pre", "held_out_pre", "held_out_post")}
    max_work = 0
    max_ms = 0.0
    for index, episode in enumerate(episodes):
        logical_time = index * 100
        started = time.perf_counter_ns()
        notice_work = learner.notify(episode.revision, time_value=logical_time) if episode.revision_notice else 0
        score, mode, prediction_work = learner.predict(episode, time_value=logical_time)
        target_label = feedback[index] if arm == "shuffled_feedback_shared_snn" else episode.label
        update_work = learner.observe(float(2 * target_label - 1), time_value=logical_time + protocol["feedback_delay"])
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        max_work = max(max_work, notice_work + prediction_work + update_work)
        max_ms = max(max_ms, elapsed_ms)
        if episode.split != "training":
            prediction = int(score > 0.0)
            probability = (max(-1.0, min(1.0, score)) + 1.0) / 2.0
            records[episode.split].append({
                "identity": episode.identity, "mode": mode, "score": score,
                "prediction": prediction, "correct": int(prediction == episode.label),
                "brier": (probability - episode.label) ** 2,
            })
    post = records["held_out_post"]
    early_count = protocol["early_post_revision_episodes"]
    digest = hashlib.sha256(json.dumps(
        [(row["identity"], row["mode"], row["score"], row["prediction"])
         for name in ("seen_pre", "held_out_pre", "held_out_post") for row in records[name]],
        separators=(",", ":"),
    ).encode()).hexdigest()
    state_bytes = _deep_size(learner)
    budgets = protocol["budgets"]
    units = len(learner.encoder.units) if learner.encoder is not None else 0
    contracts = (units <= budgets["max_units"] and state_bytes <= budgets["max_state_bytes"]
                 and max_work <= budgets["max_event_work_per_episode"]
                 and max_ms <= budgets["max_cpu_ms_per_episode"])
    return {
        "seen_pre_revision": _metrics(records["seen_pre"]),
        "held_out_pre_revision": _metrics(records["held_out_pre"]),
        "early_held_out_post_revision": _metrics(post[:early_count]),
        "late_held_out_post_revision": _metrics(post[early_count:]),
        "full_held_out_post_revision": _metrics(post),
        "adaptation_latency_episodes": _latency(post),
        "prediction_trace_sha256": digest,
        "resources": {"units": units, "max_state_bytes": state_bytes,
                      "max_event_work_per_episode": max_work, "max_cpu_ms_per_episode": max_ms,
                      "contracts_passed": contracts},
    }


def decide(protocol: dict, rows: list[dict]) -> dict:
    threshold = protocol["acceptance"]
    mean = lambda values: sum(values) / len(values)
    arm = lambda row, name: row["arms"][name]
    candidate = "revision_gain_shared_snn"
    held_pre = [arm(row, candidate)["held_out_pre_revision"]["accuracy"] for row in rows]
    atomic_gains = [arm(row, candidate)["held_out_pre_revision"]["accuracy"] - arm(row, "revision_gain_atomic_snn")["held_out_pre_revision"]["accuracy"] for row in rows]
    early_gains = [arm(row, candidate)["early_held_out_post_revision"]["accuracy"] - arm(row, "always_residual_shared_snn")["early_held_out_post_revision"]["accuracy"] for row in rows]
    residual_brier = [arm(row, "always_residual_shared_snn")["full_held_out_post_revision"]["brier"] - arm(row, candidate)["full_held_out_post_revision"]["brier"] for row in rows]
    previous_brier = [arm(row, "previous_policy_shared_snn")["full_held_out_post_revision"]["brier"] - arm(row, candidate)["full_held_out_post_revision"]["brier"] for row in rows]
    drops = [arm(row, candidate)["seen_pre_revision"]["accuracy"] - arm(row, candidate)["held_out_pre_revision"]["accuracy"] for row in rows]
    latency = max(arm(row, candidate)["adaptation_latency_episodes"] for row in rows)
    scalar_equal = all(arm(row, candidate)["prediction_trace_sha256"] == arm(row, "revision_gain_shared_scalar")["prediction_trace_sha256"] for row in rows)
    resources = all(result["resources"]["contracts_passed"] for row in rows for result in row["arms"].values())
    checks = {
        "held_out_pre_accuracy": mean(held_pre) >= threshold["minimum_candidate_held_out_pre_accuracy"],
        "held_out_pre_gain_over_atomic": mean(atomic_gains) >= threshold["minimum_held_out_pre_accuracy_gain_over_atomic"],
        "early_post_gain_over_residual": mean(early_gains) >= threshold["minimum_early_post_accuracy_gain_over_always_residual"],
        "full_post_brier_gain_over_residual": mean(residual_brier) >= threshold["minimum_full_post_brier_gain_over_always_residual"],
        "full_post_brier_gain_over_previous": mean(previous_brier) >= threshold["minimum_full_post_brier_gain_over_previous_policy"],
        "seen_to_held_out_drop": max(drops) <= threshold["maximum_seen_to_held_out_pre_accuracy_drop"],
        "adaptation_latency": latency <= threshold["maximum_adaptation_latency_episodes"],
        "all_seed_early_gains_positive": all(value > 0.0 for value in early_gains),
        "exact_scalar_prediction_equivalence": scalar_equal,
        "all_resource_contracts_pass": resources,
    }
    passed = all(checks.values())
    return {
        "checks": checks, "passed": passed,
        "mean_held_out_pre_accuracy": mean(held_pre),
        "mean_held_out_pre_accuracy_gain_over_atomic": mean(atomic_gains),
        "mean_early_post_accuracy_gain_over_always_residual": mean(early_gains),
        "mean_full_post_brier_gain_over_always_residual": mean(residual_brier),
        "mean_full_post_brier_gain_over_previous_policy": mean(previous_brier),
        "maximum_seen_to_held_out_pre_accuracy_drop": max(drops),
        "maximum_adaptation_latency_episodes": latency,
        "status": "component_pass" if passed else "negative_result_retained",
        "production_promotion": False, "r2_passed": False,
        "autonomous_change_detection": False, "snn_superiority": False,
    }


def run_benchmark(protocol: dict) -> dict:
    rows = []
    for seed in protocol["seeds"]:
        streams = make_streams(protocol, seed)
        rows.append({"seed": seed, "arms": {
            arm: run_arm(protocol, arm, streams, seed) for arm in protocol["arms"]
        }})
    return {"schema": "sara-structural-revision-transfer-result-v1",
            "experiment_id": protocol["experiment_id"], "scope": protocol["scope"],
            "rows": rows, "decision": decide(protocol, rows)}


__all__ = ["TransferEpisode", "TransferArm", "decide", "load_protocol", "make_streams", "run_arm", "run_benchmark", "shared_routes"]
