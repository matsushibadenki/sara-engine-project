"""Frozen benchmark for an explicit revision-triggered local learning policy."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Optional

from sara_engine.evaluation.local_outcome_snn_integration import FixedCoincidenceEncoder, IntegrationEpisode, route_id
from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.learning.observable_revision import BoundedObservableRevisionReadout, ObservableRevisionConfig, RevisionPrediction
from sara_engine.utils.project_paths import processed_data_path


PROTOCOL_PATH = processed_data_path("benchmark_fixtures", "observable_revision_policy_v1.json")
PROTOCOL_SHA256 = "74aeea09837d97304c12c2d777b97f275914b4cd905c5136d9fe4a12dc193182"


def load_protocol() -> dict:
    raw = Path(PROTOCOL_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen observable revision protocol changed")
    return json.loads(raw)


@dataclass(frozen=True)
class RevisionEpisode:
    identity: str
    left: int
    right: int
    interval: int
    label: int
    revision: int
    revision_notice: bool

    def encoder_episode(self) -> IntegrationEpisode:
        return IntegrationEpisode(self.identity, self.left, self.right, self.interval, self.label)


def make_streams(protocol: dict, seed: int, scenario: str) -> tuple[list[RevisionEpisode], list[RevisionEpisode]]:
    if scenario not in protocol["scenarios"]:
        raise ValueError("Unknown revision scenario")
    addresses = [(left, right, gap) for left in range(protocol["symbols"]) for right in range(protocol["symbols"]) for gap in protocol["intervals"]]
    rank = list(range(len(addresses)))
    random.Random(seed).shuffle(rank)
    initial = {address: rank[index] % 2 for index, address in enumerate(addresses)}

    def repeated(count: int, offset: int) -> list[tuple[int, int, int]]:
        rng = random.Random(seed + offset)
        result = []
        rounds, remainder = divmod(count, len(addresses))
        for round_index in range(rounds + int(remainder > 0)):
            epoch = list(addresses)
            rng.shuffle(epoch)
            result.extend(epoch[:remainder] if round_index == rounds else epoch)
        return result

    training = [RevisionEpisode(f"training:{seed}:{index}", *address, initial[address], 1, False)
                for index, address in enumerate(repeated(protocol["training_episodes"], 41_003))]
    pre_count = protocol["pre_revision_evaluation_episodes"]
    total = pre_count + protocol["post_revision_evaluation_episodes"]
    evaluation = []
    for index, address in enumerate(repeated(total, 51_011)):
        after = index >= pre_count
        label = initial[address] ^ int(after and scenario == "true_revision")
        evaluation.append(RevisionEpisode(
            f"evaluation:{scenario}:{seed}:{index}", *address, label,
            2 if after else 1, index == pre_count,
        ))
    return training, evaluation


class StaticRuleReadout:
    """Matched scalar state for fixed-rule experimental controls."""

    def __init__(self, protocol: dict, rule: str) -> None:
        if rule not in ("three_factor", "residual"):
            raise ValueError("Unknown fixed rule")
        config = protocol["configuration"]
        self.rule = rule
        self.learning_rate = config["learning_rate"]
        self.trace_decay = config["trace_decay"]
        self.max_age = config["max_feedback_age"]
        self.max_delta = config["max_delta"]
        self.max_routes = config["max_readout_routes"]
        self.weights: dict[int, float] = {}
        self.pending: Optional[tuple[int, int, float]] = None

    def predict(self, route: int, *, time_value: int) -> float:
        if self.pending is not None:
            raise ValueError("Resolve pending fixed-rule prediction")
        if route not in self.weights and len(self.weights) >= self.max_routes:
            raise ValueError("Fixed-rule route budget exceeded")
        score = self.weights.get(route, 0.0)
        self.pending = (route, time_value, score)
        return score

    def observe(self, target: float, *, time_value: int) -> int:
        if self.pending is None:
            raise ValueError("No pending fixed-rule prediction")
        route, predicted_at, score = self.pending
        self.pending = None
        age = time_value - predicted_at
        if age > self.max_age:
            return 0
        signal = target - score if self.rule == "residual" else target
        delta = max(-self.max_delta, min(self.max_delta, self.learning_rate * self.trace_decay ** age * signal))
        self.weights[route] = max(-1.0, min(1.0, score + delta))
        return 1


class RevisionPolicyArm:
    def __init__(self, protocol: dict, arm: str) -> None:
        if arm not in protocol["arms"]:
            raise ValueError("Unknown revision arm")
        self.arm = arm
        self.symbols = protocol["symbols"]
        self.encoder = None if arm == "observable_policy_scalar" else FixedCoincidenceEncoder(protocol)
        self.policy: Optional[BoundedObservableRevisionReadout] = None
        self.static: Optional[StaticRuleReadout] = None
        if arm in ("observable_policy_snn", "observable_policy_scalar", "shuffled_feedback_snn"):
            config = protocol["configuration"]
            self.policy = BoundedObservableRevisionReadout(ObservableRevisionConfig(
                learning_rate=config["learning_rate"], trace_decay=config["trace_decay"],
                max_feedback_age=config["max_feedback_age"], adaptation_horizon=protocol["adaptation_horizon"],
                max_routes=config["max_readout_routes"], max_active=config["max_active_routes"],
                max_delta=config["max_delta"],
            ))
        else:
            rule = "residual" if arm == "always_residual_snn" else "three_factor"
            self.static = StaticRuleReadout(protocol, rule)
        self.receipt: Optional[RevisionPrediction] = None

    def notify(self, revision: int, *, time_value: int) -> int:
        if self.policy is not None:
            self.policy.notify_revision(revision, time=time_value)
        # Fixed controls account for receiving or explicitly ignoring one event.
        return 1

    def predict(self, episode: RevisionEpisode, *, time_value: int) -> tuple[float, str, int]:
        if self.encoder is None:
            route = route_id(episode.left, episode.right, episode.interval, self.symbols)
            work = 1
        else:
            route, work = self.encoder.encode(episode.encoder_episode())
        if self.policy is not None:
            self.receipt = self.policy.predict(((route, 1.0),), time=time_value)
            return self.receipt.score, self.receipt.mode, work + 1
        assert self.static is not None
        return self.static.predict(route, time_value=time_value), self.static.rule, work + 1

    def observe(self, target: float, *, time_value: int) -> int:
        if self.policy is not None:
            assert self.receipt is not None
            result = self.policy.observe(self.receipt, target, time=time_value)
            self.receipt = None
            return len(result.deltas)
        assert self.static is not None
        return self.static.observe(target, time_value=time_value)


def _segment_metrics(records: list[dict]) -> dict:
    if not records:
        return {"accuracy": 0.0, "brier": 0.0, "count": 0}
    return {
        "accuracy": sum(item["correct"] for item in records) / len(records),
        "brier": sum(item["brier"] for item in records) / len(records),
        "count": len(records),
    }


def _adaptation_latency(post: list[dict]) -> int:
    window = 192
    for index in range(window - 1, len(post)):
        if sum(item["correct"] for item in post[index - window + 1:index + 1]) / window >= 0.90:
            return index + 1
    return len(post) + 1


def run_arm(protocol: dict, arm: str, training: list[RevisionEpisode], evaluation: list[RevisionEpisode], seed: int) -> dict:
    learner = RevisionPolicyArm(protocol, arm)
    shuffled = [item.label for item in training + evaluation]
    random.Random(seed + 61_001).shuffle(shuffled)
    records = []
    max_work = 0
    max_ms = 0.0
    feedback_delay = protocol["feedback_delay"]
    all_episodes = training + evaluation
    for index, episode in enumerate(all_episodes):
        logical_time = index * 100
        started = time.perf_counter_ns()
        signal_work = learner.notify(episode.revision, time_value=logical_time) if episode.revision_notice else 0
        score, mode, prediction_work = learner.predict(episode, time_value=logical_time)
        label = shuffled[index] if arm == "shuffled_feedback_snn" else episode.label
        update_work = learner.observe(float(2 * label - 1), time_value=logical_time + feedback_delay)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        max_work = max(max_work, signal_work + prediction_work + update_work)
        max_ms = max(max_ms, elapsed_ms)
        if index >= len(training):
            prediction = int(score > 0.0)
            probability = (max(-1.0, min(1.0, score)) + 1.0) / 2.0
            records.append({
                "identity": episode.identity, "revision": episode.revision,
                "revision_notice": episode.revision_notice, "mode": mode,
                "score": score, "prediction": prediction,
                "correct": int(prediction == episode.label), "brier": (probability - episode.label) ** 2,
            })
    state_bytes = _deep_size(learner)
    pre_count = protocol["pre_revision_evaluation_episodes"]
    early_count = protocol["early_post_revision_episodes"]
    pre = records[:pre_count]
    post = records[pre_count:]
    early = post[:early_count]
    late = post[early_count:]
    trace_digest = hashlib.sha256(json.dumps(
        [(item["identity"], item["mode"], item["score"], item["prediction"]) for item in records],
        separators=(",", ":"),
    ).encode()).hexdigest()
    budgets = protocol["budgets"]
    units = len(learner.encoder.units) if learner.encoder is not None else 0
    contracts = units <= budgets["max_units"] and state_bytes <= budgets["max_state_bytes"] and max_work <= budgets["max_event_work_per_episode"] and max_ms <= budgets["max_cpu_ms_per_episode"]
    return {
        "pre_revision": _segment_metrics(pre), "early_post_revision": _segment_metrics(early),
        "late_post_revision": _segment_metrics(late), "full_post_revision": _segment_metrics(post),
        "adaptation_latency_episodes": _adaptation_latency(post), "prediction_trace_sha256": trace_digest,
        "resources": {"units": units, "max_state_bytes": state_bytes,
                      "max_event_work_per_episode": max_work, "max_cpu_ms_per_episode": max_ms,
                      "contracts_passed": contracts},
    }


def decide(protocol: dict, rows: list[dict]) -> dict:
    threshold = protocol["acceptance"]
    true_rows = [row for row in rows if row["scenario"] == "true_revision"]
    false_rows = [row for row in rows if row["scenario"] == "false_revision"]
    mean = lambda values: sum(values) / len(values)
    early_gains = [row["arms"]["observable_policy_snn"]["early_post_revision"]["accuracy"] - row["arms"]["always_three_factor_snn"]["early_post_revision"]["accuracy"] for row in true_rows]
    post_brier_gains = []
    for row in true_rows:
        candidate = row["arms"]["observable_policy_snn"]["full_post_revision"]["brier"]
        best_static = min(row["arms"][name]["full_post_revision"]["brier"] for name in ("always_three_factor_snn", "always_residual_snn"))
        post_brier_gains.append(best_static - candidate)
    pre_regressions = [row["arms"]["observable_policy_snn"]["pre_revision"]["brier"] - row["arms"]["always_three_factor_snn"]["pre_revision"]["brier"] for row in rows]
    false_harms = [row["arms"]["observable_policy_snn"]["full_post_revision"]["brier"] - row["arms"]["always_three_factor_snn"]["full_post_revision"]["brier"] for row in false_rows]
    scalar_equal = all(row["arms"]["observable_policy_snn"]["prediction_trace_sha256"] == row["arms"]["observable_policy_scalar"]["prediction_trace_sha256"] for row in rows)
    resources = all(result["resources"]["contracts_passed"] for row in rows for result in row["arms"].values())
    max_latency = max(row["arms"]["observable_policy_snn"]["adaptation_latency_episodes"] for row in true_rows)
    checks = {
        "early_accuracy_gain": mean(early_gains) >= threshold["minimum_early_accuracy_gain_over_three_factor"],
        "full_post_brier_gain": mean(post_brier_gains) >= threshold["minimum_full_post_brier_gain_over_best_static"],
        "pre_revision_tolerance": max(pre_regressions) <= threshold["maximum_pre_revision_brier_regression"],
        "false_switch_tolerance": max(false_harms) <= threshold["maximum_false_switch_brier_harm"],
        "adaptation_latency": max_latency <= threshold["maximum_adaptation_latency_episodes"],
        "all_seed_early_gains_positive": all(value > 0.0 for value in early_gains),
        "exact_scalar_prediction_equivalence": scalar_equal,
        "all_resource_contracts_pass": resources,
    }
    return {
        "checks": checks, "passed": all(checks.values()),
        "mean_early_accuracy_gain_over_three_factor": mean(early_gains),
        "mean_full_post_brier_gain_over_best_static": mean(post_brier_gains),
        "maximum_pre_revision_brier_regression": max(pre_regressions),
        "maximum_false_switch_brier_harm": max(false_harms),
        "maximum_adaptation_latency_episodes": max_latency,
        "status": "component_pass" if all(checks.values()) else "negative_result_retained",
        "production_promotion": False, "r2_passed": False,
        "autonomous_change_detection": False, "snn_superiority": False,
    }


def run_benchmark(protocol: dict) -> dict:
    rows = []
    for scenario in protocol["scenarios"]:
        for seed in protocol["seeds"]:
            training, evaluation = make_streams(protocol, seed, scenario)
            rows.append({"scenario": scenario, "seed": seed, "arms": {
                arm: run_arm(protocol, arm, training, evaluation, seed) for arm in protocol["arms"]
            }})
    return {"schema": "sara-observable-revision-policy-result-v1", "experiment_id": protocol["experiment_id"],
            "scope": protocol["scope"], "rows": rows, "decision": decide(protocol, rows)}


__all__ = ["RevisionEpisode", "RevisionPolicyArm", "decide", "load_protocol", "make_streams", "run_arm", "run_benchmark"]
