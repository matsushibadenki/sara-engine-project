"""Frozen CPU-only count learning with bounded, prediction-bound delayed credit."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from time import process_time

from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.neuro.neuron import Neuron
from sara_engine.utils.project_paths import processed_data_path

PROTOCOL_SHA256 = "fbb5422d235d31764d00f37fae5f1ba0f05d2c41374854ea6ca092dc06058188"


def load_protocol():
    raw = Path(processed_data_path("benchmark_fixtures", "appliances_learning_v1.json")).read_bytes()
    if sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen appliance comparison protocol changed")
    return json.loads(raw)


def project(history, cutoff, arm, protocol, *, scalar_lif=False):
    """Input-only fixed projection; labels and outcome times are not arguments."""
    if arm not in protocol["arms"] or len(history) != protocol["history_samples"]:
        raise ValueError("Invalid arm or bounded history length")
    if any(not math.isfinite(value) or value < 0 for value in history):
        raise ValueError("Invalid historical energy")
    bins = tuple(min(protocol["maximum_energy_bin"], int(value // protocol["energy_bin_wh"]))
                 for value in history)
    states = [1.0] * len(history)
    spike_count = 0
    steps = 0
    if arm in ("joint", "joint_shuffled", "analog"):
        config = protocol["lif"]
        unit = Neuron(0, num_branches=1)
        unit.leak, unit.threshold = config["leak"], config["threshold"]
        unit.branches[0].threshold = config["branch_threshold"]
        unit.branches[0].gain = config["branch_gain"]
        membrane, refractory = 0.0, 0
        for index, value in enumerate(history):
            current = min(config["maximum_input"], value / config["input_scale_wh"])
            branch = current * config["branch_gain"] / (1 + math.exp(-5 * (current - config["branch_threshold"])))
            if arm == "analog":
                membrane = config["leak"] * membrane + branch
                states[index] = 1 + min(1.0, membrane)
            else:
                if scalar_lif:
                    if refractory:
                        refractory -= 1
                        membrane, spike = 0.0, False
                    elif branch == 0 and abs(membrane) < 0.01:
                        membrane, spike = 0.0, False
                    else:
                        membrane = membrane * config["leak"] + branch
                        spike = membrane > config["threshold"]
                        if spike:
                            membrane, refractory = 0.0, config["refractory_steps"]
                else:
                    unit.add_input_to_branch(0, current)
                    spike = unit.step()
                    if spike and unit.refractory_time != config["refractory_steps"]:
                        raise ValueError("Neuron refractory contract changed")
                states[index] = protocol["joint_spike_strength"] if spike else 1.0
                spike_count += int(spike)
            steps += 1
    if arm == "snapshot":
        values = (bins[-1],) * len(bins)
    elif arm == "histogram":
        values = tuple(sorted(bins))
    elif arm == "transition_gap":
        # Exact change positions reconstruct the common quantized ordinal tape.
        changes = {0: bins[0]}
        changes.update((i, value) for i, value in enumerate(bins[1:], 1) if value != bins[i - 1])
        restored, current = [], bins[0]
        for index in range(len(bins)):
            current = changes.get(index, current)
            restored.append(current)
        values = tuple(restored)
    else:
        values = bins
    tokens = (*values, 0, bins[-1], cutoff.hour, cutoff.weekday())
    strengths = (*states, 1.0, 1.0, 1.0, 1.0)
    active = tuple((index * protocol["route_stride"] + token, strength)
                   for index, (token, strength) in enumerate(zip(tokens, strengths)))
    if len(active) != protocol["active_routes"] or len({r for r, _ in active}) != len(active):
        raise ValueError("Actual route count mismatch")
    return active, {"history_visits": len(history), "neuron_or_analog_steps": steps,
                    "spikes": spike_count, "clipped_values": sum(value >= 1280 for value in history)}


@dataclass(frozen=True)
class Receipt:
    serial: int
    cutoff: datetime
    due: datetime
    active: tuple
    probability: float


class DelayedCounts:
    """One real class-counter increment per active route per observed outcome."""

    def __init__(self, protocol):
        self.protocol = protocol
        self.counts = {}
        self.pending = {}
        self.serial = 0
        self.last_time = None
        self.counter_writes = self.feedback_events = self.changed_counters = 0
        self.peak_pending = 0
        self.prediction_route_visits = 0

    def _time(self, stamp):
        if self.last_time is not None and stamp < self.last_time:
            raise ValueError("Clock must be monotonic")

    def predict(self, active, cutoff):
        self._time(cutoff)
        if len(self.pending) >= self.protocol["budgets"]["maximum_pending"]:
            raise ValueError("Pending credit capacity exceeded")
        if len(active) != self.protocol["active_routes"] or len({r for r, _ in active}) != len(active):
            raise ValueError("Wrong actual active-route count")
        if any(type(r) is not int or not 0 <= r < self.protocol["budgets"]["maximum_routes"]
               or not math.isfinite(s) or not 1 <= s <= 2 for r, s in active):
            raise ValueError("Invalid route or strength")
        pseudocount = self.protocol["posterior_pseudocount_per_class"]
        total, weighted = 0.0, 0.0
        for route, strength in active:
            negative, positive = self.counts.get(route, (0, 0))
            weighted += strength * (positive + pseudocount) / (negative + positive + 2 * pseudocount)
            total += strength
        self.serial += 1
        receipt = Receipt(self.serial, cutoff, cutoff + timedelta(minutes=self.protocol["feedback_delay_minutes"]),
                          tuple(active), weighted / total)
        self.pending[receipt.serial] = receipt
        self.peak_pending = max(self.peak_pending, len(self.pending))
        self.prediction_route_visits += len(active)
        self.last_time = cutoff
        return receipt

    def resolve(self, receipt, now, *, label=None, training=False):
        self._time(now)
        if self.pending.get(receipt.serial) is not receipt:
            raise ValueError("Receipt is forged, duplicated or cross-owned")
        if now != receipt.due:
            raise ValueError("Outcome must arrive at exactly the registered delay")
        if training and (type(label) is not int or label not in (0, 1)):
            raise ValueError("Invalid binary observed outcome")
        if training:
            for route, _ in receipt.active:
                counts = self.counts.setdefault(route, [0, 0])
                old = counts[label]
                counts[label] += 1
                self.changed_counters += int(counts[label] != old)
                self.counter_writes += 1
            self.feedback_events += 1
        del self.pending[receipt.serial]
        self.last_time = now

    def state_bound(self):
        # Conservative object/entry/scratch allowances; excludes source/evaluator tape.
        return 65536 + len(self.counts) * 512 + self.protocol["budgets"]["maximum_pending"] * (
            1024 + self.protocol["active_routes"] * 256)

    def checkpoint(self):
        return {str(route): list(counts) for route, counts in sorted(self.counts.items())}

    def restore(self, checkpoint):
        if self.counts or self.pending or self.serial:
            raise ValueError("Restore requires a fresh readout")
        restored = {}
        for route, counts in checkpoint.items():
            if not route.isdecimal() or str(int(route)) != route or not 0 <= int(route) < self.protocol["budgets"]["maximum_routes"]:
                raise ValueError("Invalid saved route")
            if len(counts) != 2 or any(type(n) is not int or not 0 <= n <= 100000 for n in counts):
                raise ValueError("Invalid saved counts")
            restored[int(route)] = list(counts)
        self.counts = restored


def deliver(model, receipt, now, outcome_supplier, *, training):
    """Validate time/ownership before accessing an evaluator's outcome cell."""
    model._time(now)
    if model.pending.get(receipt.serial) is not receipt or now != receipt.due:
        raise ValueError("Outcome access requires an owned receipt at its due time")
    label = outcome_supplier()
    if type(label) is not int or label not in (0, 1):
        raise ValueError("Invalid observed outcome")
    model.resolve(receipt, now, label=label, training=training)
    return label


def shuffled_targets(training, protocol):
    """Evaluator-only label permutation; no outcome is passed into projection."""
    days = defaultdict(list)
    for index, example in enumerate(training):
        days[example.cutoff.date()].append(index)
    targets = {}
    for day, indices in sorted(days.items()):
        order = sorted(indices, key=lambda i: sha256(f'{protocol["shuffle_seed"]}:{day}:{i}'.encode()).digest())
        for destination, source in zip(indices, order):
            targets[training[destination].cutoff] = training[source].label
    return targets


def run_phase(model, examples, arm, protocol, *, training, targets=None):
    """Chronological prediction/feedback loop; at most three owned pending receipts."""
    pending = deque()
    days = defaultdict(lambda: [0.0, 0])
    trace = sha256()
    work = dict.fromkeys(("history_visits", "neuron_or_analog_steps", "spikes", "clipped_values"), 0)
    true_positive = true_negative = positives = negatives = 0
    start = process_time()
    before = sha256(json.dumps(model.checkpoint(), sort_keys=True).encode()).hexdigest()
    writes_before = model.counter_writes
    reference_checked = 0

    def release(receipt, example):
        # This is the only label access for ordinary feedback/scoring, after due time.
        if receipt.due != example.outcome_time:
            raise ValueError("Evaluator outcome and receipt delay disagree")
        label = deliver(model, receipt, receipt.due,
                        lambda: targets[example.cutoff] if targets is not None else example.label,
                        training=training)
        if not training:
            probability = receipt.probability
            day = days[str(example.cutoff.date())]
            day[0] += (probability - label) ** 2
            day[1] += 1
            trace.update(json.dumps([example.cutoff.isoformat(), label, probability], separators=(",", ":")).encode())
            return label, int(probability >= 0.5)
        return None

    def account(outcome):
        nonlocal positives, negatives, true_positive, true_negative
        if outcome is not None:
            label, predicted = outcome
            positives += label
            negatives += 1 - label
            true_positive += int(label == predicted == 1)
            true_negative += int(label == predicted == 0)

    for example in examples:
        if example.split != ("train" if training else "development"):
            raise ValueError("Wrong learner split")
        while pending and pending[0][0].due <= example.cutoff:
            account(release(*pending.popleft()))
        active, costs = project(example.history, example.cutoff, arm, protocol)
        if arm in ("joint", "joint_shuffled"):
            scalar_active, _ = project(example.history, example.cutoff, arm, protocol, scalar_lif=True)
            if active != scalar_active:
                raise ValueError("Neuron and independent scalar LIF reference disagree")
            reference_checked += 1
        for key in work:
            work[key] += costs[key]
        receipt = model.predict(active, example.cutoff)
        pending.append((receipt, example))
    while pending:
        account(release(*pending.popleft()))
    after = sha256(json.dumps(model.checkpoint(), sort_keys=True).encode()).hexdigest()
    actual_writes = model.counter_writes - writes_before
    expected_writes = len(examples) * protocol["active_routes"] if training else 0
    if actual_writes != expected_writes or (not training and before != after) or model.pending:
        raise ValueError("Actual update or frozen-state integrity failed")
    if model.state_bound() > protocol["budgets"]["maximum_state_bytes_per_arm"]:
        raise ValueError("Readout state bound exceeded")
    measured_bytes = _deep_size(model)
    if measured_bytes > model.state_bound():
        raise ValueError("Measured readout state exceeds conservative bound")
    day_brier = {day: loss / count for day, (loss, count) in sorted(days.items())}
    return {
        "examples": len(examples), "actual_counter_writes": actual_writes,
        "changed_counters_total": model.changed_counters, "peak_pending": model.peak_pending,
        "pending_after_phase": len(model.pending), "state_bound_bytes": model.state_bound(),
        "measured_readout_object_bytes_at_end": measured_bytes,
        "route_count": len(model.counts), "work": work,
        "scalar_lif_reference_predictions_checked": reference_checked,
        "scalar_reference_extra_history_visits": reference_checked * protocol["history_samples"],
        "cpu_seconds_including_reference_and_instrumentation": process_time() - start,
        "prediction_route_visits": len(examples) * protocol["active_routes"],
        "frozen": not training, "prediction_sha256": None if training else trace.hexdigest(),
        "day_brier": day_brier, "equal_day_brier": None if training else sum(day_brier.values()) / len(day_brier),
        "pooled_brier": None if training else sum(loss for loss, _ in days.values()) / len(examples),
        "balanced_accuracy": None if training else (true_positive / positives + true_negative / negatives) / 2,
    }


def decide(arms, protocol):
    metrics = {arm: result["development"] for arm, result in arms.items()}
    reference = min(protocol["references"], key=lambda arm: metrics[arm]["equal_day_brier"])
    joint = metrics["joint"]
    days = sorted(joint["day_brier"])
    if any(sorted(result["day_brier"]) != days for result in metrics.values()):
        raise ValueError("Paired day identities differ")
    differences = [metrics[reference]["day_brier"][day] - joint["day_brier"][day] for day in days]
    rng = random.Random(protocol["bootstrap_seed"])
    bootstrap = sorted(sum(rng.choice(differences) for _ in days) / len(days)
                       for _ in range(protocol["bootstrap_replicates"]))
    lower = bootstrap[100]
    gate = protocol["gates"]
    checks = {
        "quality": joint["balanced_accuracy"] >= gate["minimum_balanced_accuracy"] and joint["equal_day_brier"] <= gate["maximum_day_brier"],
        "snn_increment": sum(differences) / len(days) >= gate["minimum_reference_brier_gain"] and lower > gate["minimum_bootstrap_lower_gain"],
        "shuffled_outcome": metrics["joint_shuffled"]["equal_day_brier"] - joint["equal_day_brier"] >= gate["minimum_shuffle_brier_gain"],
        "actual_updates_equal": len({result["training"]["actual_counter_writes"] for result in arms.values()}) == 1,
        "all_writes_changed": all(result["training"]["changed_counters_total"] == result["training"]["actual_counter_writes"] for result in arms.values()),
        "ordinal_gap_equivalence": metrics["ordinal"]["prediction_sha256"] == metrics["transition_gap"]["prediction_sha256"],
        "state_and_credit": all(result["training"]["peak_pending"] <= protocol["budgets"]["maximum_pending"] and result["development"]["frozen"] and result["development"]["actual_counter_writes"] == 0 for result in arms.values()),
        "checkpoint_replay": all(result["checkpoint_replay_exact"] for result in arms.values()),
    }
    return {"best_nonspiking_reference": reference, "reference_brier_gain": sum(differences) / len(days),
            "paired_day_bootstrap_lower_95_gain": lower, "days": len(days), "gates": checks,
            "passed": all(checks.values()), "test_unsealing_authorized": False}
