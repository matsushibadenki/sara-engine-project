"""Frozen fixed-topology SNN readout integration benchmark.

The encoder is deliberately fixed and resets its sensory unit after each
episode. The benchmark tests wiring and bounded local adaptation, not learned
representation, SNN superiority, production readiness, or physical energy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Optional

from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.learning.local_outcome import BoundedLocalOutcomeReadout, LocalOutcomeConfig, LocalPrediction
from sara_engine.learning.three_factor_learning import ThreeFactorLearningManager
from sara_engine.neuro.neuron import Neuron
from sara_engine.utils.project_paths import processed_data_path


PROTOCOL_PATH = processed_data_path("benchmark_fixtures", "local_outcome_snn_integration_v1.json")
PROTOCOL_SHA256 = "2bcc62e0de66683299e71df749e0845771db6259bad272160d4ed0ee2a65708b"


def load_protocol() -> dict:
    raw = Path(PROTOCOL_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen SNN integration protocol changed")
    return json.loads(raw)


@dataclass(frozen=True)
class IntegrationEpisode:
    identity: str
    left: int
    right: int
    interval: int
    label: int


def interval_bin(interval: int) -> int:
    if interval <= 0:
        raise ValueError("Interval must be positive")
    return 0 if interval <= 3 else 1 if interval <= 7 else 2


def route_id(left: int, right: int, interval: int, symbols: int) -> int:
    if type(left) is not int or type(right) is not int or not 0 <= left < symbols or not 0 <= right < symbols:
        raise ValueError("Symbol is outside the fixed vocabulary")
    return (left * symbols + right) * 3 + interval_bin(interval)


def make_streams(protocol: dict, seed: int) -> tuple[list[IntegrationEpisode], list[IntegrationEpisode]]:
    symbols = protocol["symbols"]
    addresses = [(left, right, gap) for left in range(symbols) for right in range(symbols) for gap in protocol["intervals"]]
    label_order = list(range(len(addresses)))
    random.Random(seed).shuffle(label_order)
    labels = {address: label_order[index] % 2 for index, address in enumerate(addresses)}

    def build(split: str, count: int, offset: int) -> list[IntegrationEpisode]:
        rng = random.Random(seed + offset)
        repetitions, remainder = divmod(count, len(addresses))
        ordered = []
        for repetition in range(repetitions + int(remainder > 0)):
            epoch = list(addresses)
            rng.shuffle(epoch)
            ordered.extend(epoch[:remainder] if repetition == repetitions else epoch)
        return [
            IntegrationEpisode(f"{split}:{seed}:{index}", left, right, gap, labels[(left, right, gap)])
            for index, (left, right, gap) in enumerate(ordered)
        ]

    return (
        build("training", protocol["training_episodes"], 10_001),
        build("evaluation", protocol["evaluation_episodes"], 20_003),
    )


class FixedCoincidenceEncoder:
    """Fixed one-hot address space backed by explicit dendritic neurons."""

    def __init__(self, protocol: dict, *, destroy_timing: bool = False) -> None:
        config = protocol["configuration"]
        self.symbols = protocol["symbols"]
        self.destroy_timing = destroy_timing
        self.branch_inputs = tuple(float(value) for value in config["branch_inputs"])
        self.units = [Neuron(index, num_branches=config["branches_per_neuron"]) for index in range(protocol["address_space"])]

    @staticmethod
    def _reset(unit: Neuron) -> None:
        unit.v = 0.0
        unit.spike = False
        unit.refractory_time = 0
        unit.active_branches.clear()
        for branch in unit.branches:
            branch.current_input = 0.0
            branch.is_active = False

    def encode(self, episode: IntegrationEpisode) -> tuple[int, int]:
        gap = 2 if self.destroy_timing else episode.interval
        address = route_id(episode.left, episode.right, gap, self.symbols)
        unit = self.units[address]
        self._reset(unit)
        for value in self.branch_inputs:
            unit.add_input_to_branch(0, value)
        if not unit.step():
            raise RuntimeError("Frozen coincidence input did not produce a spike")
        # Two deliveries, address calculation, and one active-neuron step.
        return address, len(self.branch_inputs) + 2


class IntegrationLearner:
    def __init__(self, protocol: dict, arm: str) -> None:
        if arm not in protocol["arms"]:
            raise ValueError("Unknown integration arm")
        config = protocol["configuration"]
        self.arm = arm
        self.symbols = protocol["symbols"]
        self.encoder: Optional[FixedCoincidenceEncoder] = None
        if arm != "scalar_residual":
            self.encoder = FixedCoincidenceEncoder(protocol, destroy_timing=arm == "snn_timing_destroyed_residual")
        self.weights: dict[int, float] = {}
        self.pending_route: Optional[int] = None
        self.pending_score = 0.0
        self.manager: Optional[ThreeFactorLearningManager] = None
        self.readout: Optional[BoundedLocalOutcomeReadout] = None
        if arm == "snn_three_factor":
            self.manager = ThreeFactorLearningManager(
                lr=config["learning_rate"], trace_decay=config["trace_decay"], use_rpe=False,
                max_traces=config["max_active_routes"], max_trace_age=config["max_feedback_age"],
            )
        else:
            self.readout = BoundedLocalOutcomeReadout(LocalOutcomeConfig(
                learning_rate=0.0 if arm == "snn_frozen" else config["learning_rate"],
                trace_decay=config["trace_decay"], max_age=config["max_feedback_age"],
                max_routes=config["max_readout_routes"], max_active=config["max_active_routes"],
                max_delta=config["max_delta"], integral_rate=config["integral_rate"],
            ))
        self.pending_receipt: Optional[LocalPrediction] = None

    def predict(self, episode: IntegrationEpisode, *, time_value: int) -> tuple[float, int]:
        if self.encoder is None:
            address = route_id(episode.left, episode.right, episode.interval, self.symbols)
            encoder_work = 1
        else:
            address, encoder_work = self.encoder.encode(episode)
        if self.manager is not None:
            self.manager.reset()
            self.manager.update_trace(address, 0, 1.0, time_value)
            self.pending_route = address
            self.pending_score = self.weights.get(address, 0.0)
            return self.pending_score, encoder_work + 1
        assert self.readout is not None
        self.pending_receipt = self.readout.predict(((address, 1.0),), time=time_value)
        return self.pending_receipt.score, encoder_work + 1

    def observe(self, target: float, *, time_value: int) -> int:
        if self.manager is not None:
            if self.pending_route is None:
                raise ValueError("No pending prediction")
            updates = self.manager.apply_reward(target, time=time_value)
            delta = updates.get((self.pending_route, 0), 0.0)
            self.weights[self.pending_route] = max(-1.0, min(1.0, self.pending_score + delta))
            work = self.manager.last_reward_event_cost
            self.pending_route = None
            return work
        assert self.readout is not None and self.pending_receipt is not None
        update = self.readout.observe(self.pending_receipt, target, time=time_value)
        self.pending_receipt = None
        return update.route_visits


def run_arm(protocol: dict, arm: str, training: list[IntegrationEpisode], evaluation: list[IntegrationEpisode], seed: int) -> dict:
    learner = IntegrationLearner(protocol, arm)
    feedback_rng = random.Random(seed + 30_011)
    training_feedback = [episode.label for episode in training]
    evaluation_feedback = [episode.label for episode in evaluation]
    feedback_rng.shuffle(training_feedback)
    feedback_rng.shuffle(evaluation_feedback)
    feedback_delay = protocol["feedback_delay"]

    def process(episodes: list[IntegrationEpisode], shuffled: list[int], scored: bool):
        correct = 0
        brier = 0.0
        records = []
        max_work = 0
        max_ms = 0.0
        for index, episode in enumerate(episodes):
            logical_time = index * 100 if not scored else (len(training) + index) * 100
            start = time.perf_counter_ns()
            score, prediction_work = learner.predict(episode, time_value=logical_time)
            feedback = shuffled[index] if arm == "snn_shuffled_feedback" else episode.label
            update_work = learner.observe(float(2 * feedback - 1), time_value=logical_time + feedback_delay)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            work = prediction_work + update_work
            max_work = max(max_work, work)
            max_ms = max(max_ms, elapsed_ms)
            if scored:
                prediction = int(score > 0.0)
                correct += int(prediction == episode.label)
                brier += ((max(-1.0, min(1.0, score)) + 1.0) / 2.0 - episode.label) ** 2
                records.append((episode.identity, score, prediction))
        return correct, brier, records, max_work, max_ms

    process(training, training_feedback, False)
    state_after_training = _deep_size(learner)
    correct, brier, records, max_work, max_ms = process(evaluation, evaluation_feedback, True)
    state_bytes = max(state_after_training, _deep_size(learner))
    digest = hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()
    budgets = protocol["budgets"]
    contracts = (
        len(learner.encoder.units) if learner.encoder is not None else 0
    ) <= budgets["max_units"] and state_bytes <= budgets["max_state_bytes"] and max_work <= budgets["max_event_work_per_episode"] and max_ms <= budgets["max_cpu_ms_per_episode"]
    return {
        "accuracy": correct / len(evaluation), "brier": brier / len(evaluation),
        "prediction_trace_sha256": digest,
        "resources": {"units": len(learner.encoder.units) if learner.encoder is not None else 0,
                      "max_state_bytes": state_bytes, "max_event_work_per_episode": max_work,
                      "max_cpu_ms_per_episode": max_ms, "contracts_passed": contracts},
    }


def decide(protocol: dict, rows: list[dict]) -> dict:
    acceptance = protocol["acceptance"]
    brier_gains = [row["arms"]["snn_three_factor"]["brier"] - row["arms"]["snn_residual"]["brier"] for row in rows]
    accuracy_regressions = [row["arms"]["snn_three_factor"]["accuracy"] - row["arms"]["snn_residual"]["accuracy"] for row in rows]
    shuffled_gains = [row["arms"]["snn_residual"]["accuracy"] - row["arms"]["snn_shuffled_feedback"]["accuracy"] for row in rows]
    timing_losses = [row["arms"]["snn_residual"]["accuracy"] - row["arms"]["snn_timing_destroyed_residual"]["accuracy"] for row in rows]
    scalar_equal = all(row["arms"]["snn_residual"]["prediction_trace_sha256"] == row["arms"]["scalar_residual"]["prediction_trace_sha256"] for row in rows)
    resources = all(result["resources"]["contracts_passed"] for row in rows for result in row["arms"].values())
    mean = lambda values: sum(values) / len(values)
    checks = {
        "mean_brier_gain": mean(brier_gains) >= acceptance["minimum_brier_gain_over_three_factor"],
        "accuracy_tolerance": max(accuracy_regressions) <= acceptance["maximum_accuracy_regression_vs_three_factor"],
        "mean_shuffled_gain": mean(shuffled_gains) >= acceptance["minimum_accuracy_gain_over_shuffled"],
        "mean_timing_loss": mean(timing_losses) >= acceptance["minimum_timing_loss"],
        "all_seed_brier_gains_positive": all(value > 0.0 for value in brier_gains),
        "exact_scalar_prediction_equivalence": scalar_equal,
        "all_resource_contracts_pass": resources,
    }
    return {
        "checks": checks, "passed": all(checks.values()),
        "mean_brier_gain_over_three_factor": mean(brier_gains),
        "maximum_accuracy_regression_vs_three_factor": max(accuracy_regressions),
        "mean_accuracy_gain_over_shuffled": mean(shuffled_gains),
        "mean_timing_loss": mean(timing_losses),
        "next_status": "integration_component_pass" if all(checks.values()) else "negative_result_retained",
        "production_promotion": False, "r2_passed": False, "snn_superiority": False,
    }


def run_benchmark(protocol: dict) -> dict:
    rows = []
    for seed in protocol["seeds"]:
        training, evaluation = make_streams(protocol, seed)
        rows.append({"seed": seed, "arms": {
            arm: run_arm(protocol, arm, training, evaluation, seed) for arm in protocol["arms"]
        }})
    return {"schema": "sara-local-outcome-snn-integration-result-v1", "experiment_id": protocol["experiment_id"],
            "scope": protocol["scope"], "rows": rows, "decision": decide(protocol, rows)}


__all__ = ["FixedCoincidenceEncoder", "IntegrationEpisode", "IntegrationLearner", "decide", "interval_bin", "load_protocol", "make_streams", "route_id", "run_arm", "run_benchmark"]
