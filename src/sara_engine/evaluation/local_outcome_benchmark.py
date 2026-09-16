"""Fixed synthetic comparison of bounded local readout update rules.

All arms see identical supplied route IDs. This component probe cannot establish
spike timing, representation learning, deep credit assignment or energy savings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time

from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.learning.local_outcome import BoundedLocalOutcomeReadout, LocalOutcomeConfig
from sara_engine.learning.three_factor_learning import ThreeFactorLearningManager
from sara_engine.utils.project_paths import processed_data_path


PROTOCOL_PATH = processed_data_path("benchmark_fixtures", "local_outcome_rule_v1.json")
PROTOCOL_SHA256 = "021c32b811580693b8805fca6105b10b9ba796edb9b1f21a609daefcb4f7d6e2"


def load_protocol() -> dict:
    raw = Path(PROTOCOL_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen local outcome protocol changed")
    return json.loads(raw)


@dataclass(frozen=True)
class Trial:
    route: int
    label: int
    start: int
    delay: int
    scored: bool


def make_stream(protocol: dict, seed: int, scenario: str) -> list[Trial]:
    if scenario not in protocol["scenarios"]:
        raise ValueError("Unknown scenario")
    rng = random.Random(seed)
    targets = [rng.randrange(2) for _ in range(protocol["routes"])]
    result = []
    for index in range(protocol["warmup"] + protocol["evaluation"]):
        route = rng.randrange(protocol["routes"])
        delay = rng.choice((1, 8))
        label = targets[route]
        scored_index = index - protocol["warmup"]
        if scenario == "reversal" and scored_index >= 0:
            label ^= (scored_index // 256) % 2
        noise = rng.random()
        if scenario == "noisy" and noise < 0.2:
            label ^= 1
        result.append(Trial(route, label, index * 100, 80 if scenario == "expired_feedback" else delay, scored_index >= 0))
    return result


class ReferenceReadout:
    """Actual existing three-factor manager or a matched non-spiking EMA."""

    def __init__(self, protocol: dict, arm: str) -> None:
        self.arm = arm
        self.learning_rate = protocol["learning_rate"]
        self.trace_decay = protocol["trace_decay"]
        self.max_age = protocol["max_age"]
        self.max_delta = protocol["max_delta"]
        self.max_routes = protocol["max_routes"]
        self.weights: dict[int, float] = {}
        self.manager = ThreeFactorLearningManager(
            lr=self.learning_rate, trace_decay=self.trace_decay, use_rpe=False,
            max_traces=protocol["max_active"], max_trace_age=self.max_age,
        ) if arm == "three_factor" else None

    def predict(self, trial: Trial) -> float:
        if trial.route not in self.weights and len(self.weights) >= self.max_routes:
            raise ValueError("Reference route budget exceeded")
        if self.manager:
            self.manager.reset()
            self.manager.update_trace(trial.route, 0, 1.0, trial.start)
        return self.weights.get(trial.route, 0.0)

    def observe(self, trial: Trial, score: float, target: float) -> bool:
        if self.manager:
            deltas = self.manager.apply_reward(target, time=trial.start + trial.delay)
            if not deltas:
                return False
            delta = deltas[(trial.route, 0)]
        else:
            if trial.delay > self.max_age:
                return False
            delta = self.learning_rate * self.trace_decay ** trial.delay * (target - score)
        delta = max(-self.max_delta, min(self.max_delta, delta))
        self.weights[trial.route] = max(-1.0, min(1.0, score + delta))
        return True


def make_arm(protocol: dict, arm: str):
    if arm in ("three_factor", "scalar_ema"):
        return ReferenceReadout(protocol, arm)
    if arm not in ("residual", "integrated", "frozen", "shuffled"):
        raise ValueError("Unknown arm")
    return BoundedLocalOutcomeReadout(LocalOutcomeConfig(
        learning_rate=0.0 if arm == "frozen" else protocol["learning_rate"],
        trace_decay=protocol["trace_decay"], max_age=protocol["max_age"],
        max_routes=protocol["max_routes"], max_active=protocol["max_active"],
        max_delta=protocol["max_delta"],
        integral_rate=protocol["integral_rate"] if arm == "integrated" else 0.0,
        integral_decay=protocol["integral_decay"], integral_cap=protocol["integral_cap"],
        integral_max_age=protocol["integral_max_age"],
    ))


def run_arm(protocol: dict, arm: str, stream: list[Trial], seed: int) -> dict:
    learner = make_arm(protocol, arm)
    feedback_rng = random.Random(seed + 77_777_777)
    correct = 0
    brier = 0.0
    scored = 0
    updates = 0
    peak_bytes = 0
    max_ms = 0.0
    total_ms = 0.0
    digest = hashlib.sha256()
    input_digest = hashlib.sha256()
    for trial in stream:
        begin = time.perf_counter_ns()
        if isinstance(learner, ReferenceReadout):
            receipt = None
            score = learner.predict(trial)
        else:
            receipt = learner.predict(((trial.route, 1.0),), time=trial.start)
            score = receipt.score
        predict_ns = time.perf_counter_ns() - begin
        # Traverse the entire maintained object, including pending feedback,
        # outside the timed section. Measurement overhead is not runtime work.
        peak_bytes = max(peak_bytes, _deep_size(learner))
        observed_label = feedback_rng.randrange(2) if arm == "shuffled" else trial.label
        target = float(2 * observed_label - 1)
        begin = time.perf_counter_ns()
        if receipt is None:
            changed = learner.observe(trial, score, target)
        else:
            outcome = learner.observe(receipt, target, time=trial.start + trial.delay)
            changed = outcome.decision == "updated"
        elapsed_ms = (predict_ns + time.perf_counter_ns() - begin) / 1e6
        total_ms += elapsed_ms
        max_ms = max(max_ms, elapsed_ms)
        peak_bytes = max(peak_bytes, _deep_size(learner))
        updates += int(changed)
        if trial.scored:
            correct += int(int(score > 0.0) == trial.label)
            brier += ((score + 1.0) / 2.0 - trial.label) ** 2
            scored += 1
        input_digest.update(json.dumps(asdict(trial), sort_keys=True).encode())
        digest.update(json.dumps([asdict(trial), score, observed_label, changed], sort_keys=True).encode())
    return {
        "accuracy": correct / scored, "brier": brier / scored,
        "scored": scored, "updates": updates,
        "prediction_trace_sha256": digest.hexdigest(),
        "input_sha256": input_digest.hexdigest(),
        "max_state_bytes": peak_bytes,
        "max_episode_ms": max_ms, "mean_episode_ms": total_ms / len(stream),
    }


def select_rule(protocol: dict, rows: list[dict]) -> dict:
    thresholds = protocol["selection"]
    primary = [row for row in rows if row["scenario"] != "expired_feedback"]
    if not primary:
        raise ValueError("Selection requires scored scenarios")
    resources_ok = all(
        item["max_state_bytes"] <= thresholds["max_state_bytes"]
        and item["max_episode_ms"] <= thresholds["max_episode_ms"]
        for row in rows for item in row["arms"].values()
    )
    expiry_ok = all(item["updates"] == 0 for row in rows if row["scenario"] == "expired_feedback" for item in row["arms"].values())
    inputs_ok = all(len({item["input_sha256"] for item in row["arms"].values()}) == 1 for row in rows)
    scalar_parity = all(
        row["arms"]["residual"]["prediction_trace_sha256"] == row["arms"]["scalar_ema"]["prediction_trace_sha256"]
        for row in rows
    )
    def mean_gain(candidate: str, baseline: str, subset: list[dict]) -> float:
        return sum(row["arms"][baseline]["brier"] - row["arms"][candidate]["brier"] for row in subset) / len(subset)

    residual_gain = mean_gain("residual", "three_factor", primary)
    integral_gain = mean_gain("integrated", "residual", primary)
    residual_ok = residual_gain >= thresholds["residual_mean_brier_gain_over_three_factor"]
    integral_ok = integral_gain >= thresholds["integrated_mean_brier_gain_over_residual"]
    for scenario in {row["scenario"] for row in primary}:
        subset = [row for row in primary if row["scenario"] == scenario]
        for candidate in ("residual", "integrated"):
            accuracy_gain = sum(row["arms"][candidate]["accuracy"] - row["arms"]["three_factor"]["accuracy"] for row in subset) / len(subset)
            if accuracy_gain < -thresholds["max_scenario_accuracy_regression"]:
                if candidate == "residual":
                    residual_ok = False
                else:
                    integral_ok = False
        if mean_gain("integrated", "residual", subset) < -thresholds["integrated_max_scenario_brier_regression"]:
            integral_ok = False
    if thresholds["all_seed_mean_gains_positive"]:
        for seed in {row["seed"] for row in primary}:
            subset = [row for row in primary if row["seed"] == seed]
            residual_ok &= mean_gain("residual", "three_factor", subset) > 0.0
            integral_ok &= mean_gain("integrated", "residual", subset) > 0.0
    valid = resources_ok and expiry_ok and inputs_ok and scalar_parity
    selected = "integrated" if valid and residual_ok and integral_ok else "residual" if valid and residual_ok else "three_factor"
    return {
        "selected_component_rule": selected,
        "residual_mean_brier_gain": residual_gain,
        "integrated_mean_brier_gain": integral_gain,
        "residual_quality_gate": bool(residual_ok), "integrated_quality_gate": bool(integral_ok),
        "resource_gate": resources_ok, "expiry_gate": expiry_ok,
        "same_input_gate": inputs_ok, "scalar_equivalence_observed": scalar_parity,
        "production_promotion": False, "r2_passed": False,
    }


def run_benchmark(protocol: dict) -> dict:
    rows = []
    for scenario in protocol["scenarios"]:
        for seed in protocol["seeds"]:
            stream = make_stream(protocol, seed, scenario)
            rows.append({"seed": seed, "scenario": scenario, "arms": {
                arm: run_arm(protocol, arm, stream, seed) for arm in protocol["arms"]
            }})
    return {"experiment_id": protocol["experiment_id"], "scope": protocol["scope"],
            "rows": rows, "selection": select_rule(protocol, rows)}
