"""Bounded, supervised discovery of unnamed predictive routes.

This isolated research component selects from explicit one/two-channel keys.
It does not learn encoders, invent factors, or implement spiking neurons.
"""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations, product
import json
from math import log2
import random
from time import process_time

from sara_engine.evaluation.sparse_predictive_memory import deep_size


PROTOCOL_SHA256 = "7de44a4e078bdb6fe370df3a9839ca32d1464bfd066e643201bcdd025d4ed78f"
ARMS = ("joint_single", "learned_no_inhibition", "learned_output_inhibition",
        "learned_full_inhibition", "provided_routes")
SPLITS = ("in_distribution", "nuisance_shift", "novel_combination", "intervention_pairs")
ROUTES = tuple((key, output) for width in (1, 2)
               for key in combinations(range(5), width) for output in range(3))


def bits(value, width):
    if (not isinstance(value, (tuple, list)) or len(value) != width
            or any(type(bit) is not int or bit not in (0, 1) for bit in value)):
        raise ValueError("Expected a bounded binary event packet")
    return tuple(value)


def scalar_slots(value):
    """Conservative logical leaves, counting shared references repeatedly."""
    if isinstance(value, dict):
        return sum(scalar_slots(k) + scalar_slots(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return sum(scalar_slots(v) for v in value)
    return 1


@dataclass(frozen=True)
class Sample:
    events: tuple
    outcome: tuple


class Task:
    """Private generator structure is passed only to the provided-route arm."""

    def __init__(self, seed, name):
        if name not in ("independent", "interacting"):
            raise ValueError("Unknown task")
        self.seed, self.name = seed, name
        rng = random.Random(seed)
        self.inputs, self.outputs = list(range(5)), list(range(3))
        rng.shuffle(self.inputs)
        rng.shuffle(self.outputs)

    def encode(self, latent, nuisance):
        a, b, c = latent
        raw = (a, b, c, *nuisance)
        target = (a, b ^ c if self.name == "interacting" else b, c)
        return Sample(tuple(raw[i] for i in self.inputs), tuple(target[i] for i in self.outputs))

    def provided_routes(self):
        dependencies = ((0,), (1, 2) if self.name == "interacting" else (1,), (2,))
        return tuple((tuple(sorted(self.inputs.index(i) for i in dependencies[target])), slot)
                     for slot, target in enumerate(self.outputs))

    def samples(self, split, count):
        if (split not in ("train", *SPLITS) or type(count) is not int
                or not 2 <= count <= 4096 or count % 2):
            raise ValueError("Invalid split or sample bound")
        rng = random.Random(self.seed if split == "train" else self.seed + 100000 + SPLITS.index(split))
        train_states = list(product((0, 1), repeat=3))[:-1]
        rows = []
        while len(rows) < count:
            latent = (1, 1, 1) if split == "novel_combination" else rng.choice(train_states)
            a, b, _ = latent
            if split in ("train", "in_distribution"):
                nuisance = (a if rng.random() < 0.9 else 1 - a,
                            b if rng.random() < 0.9 else 1 - b)
            elif split == "nuisance_shift":
                nuisance = (1 - a, 1 - b)
            else:
                nuisance = (rng.randrange(2), rng.randrange(2))
            rows.append(self.encode(latent, nuisance))
            if split == "intervention_pairs":
                changed = list(latent)
                axis = rng.randrange(3)
                changed[axis] ^= 1
                rows.append(self.encode(tuple(changed), nuisance))
        return rows


def synthesize(packets):
    """Missing or conflicting component packets abstain; order has no effect."""
    if not isinstance(packets, (tuple, list)) or len(packets) > 3:
        raise ValueError("Factor packet limit exceeded")
    grouped = [[], [], []]
    for slot, probability in packets:
        if type(slot) is not int or not 0 <= slot < 3:
            raise ValueError("Invalid output address")
        if probability is not None:
            if type(probability) not in (float, int) or not 0 <= probability <= 1:
                raise ValueError("Invalid factor probability")
            grouped[slot].append(probability)
    output = []
    for values in grouped:
        if not values or len({p >= 0.5 for p in values}) > 1:
            output.append(None)
        else:
            output.append(sum(sorted(values)) / len(values))
    return tuple(output)


class LocalRouteMemory:
    """Fixed candidate bank with bounded delayed local error and count updates."""

    def __init__(self, arm, provided=None):
        if arm not in ARMS or (arm != "provided_routes" and provided is not None):
            raise ValueError("Invalid arm or unauthorized route hints")
        if arm == "provided_routes":
            if (not isinstance(provided, tuple) or len(provided) != 3
                    or any(route not in ROUTES for route in provided)):
                raise ValueError("Expected three diagnostic routes")
            routes = provided
        elif arm == "joint_single":
            routes = tuple((tuple(range(5)), slot) for slot in range(3))
        else:
            routes = ROUTES
        self.arm, self.routes = arm, routes
        self.tables = [{} for _ in routes]
        self.errors = [0.25 for _ in routes]
        self.observations = [0 for _ in routes]
        self.pending = None
        self.clock = -1
        self.updates = 0

    def select(self):
        if self.arm in ("joint_single", "provided_routes"):
            return (0, 1, 2), 0, 0
        available = [i for i, n in enumerate(self.observations) if n >= 16]
        selected, visits, checks = [], 0, 0
        for _ in range(3):
            scores = []
            for i in available:
                if i in selected:
                    continue
                visits += 1
                inputs, slot = self.routes[i]
                score = self.errors[i] + 0.005 * len(inputs)
                for j in selected:
                    if self.arm != "learned_no_inhibition":
                        checks += 1
                        other_inputs, other_slot = self.routes[j]
                        score += 0.25 * (slot == other_slot)
                        if self.arm == "learned_full_inhibition":
                            score += 0.02 * bool(set(inputs) & set(other_inputs))
                scores.append((score, i))
            if not scores:
                break
            selected.append(min(scores)[1])
        return tuple(selected), visits, checks

    def probability(self, index, events):
        inputs, _ = self.routes[index]
        counts = self.tables[index].get(tuple(events[i] for i in inputs))
        if counts is None:
            return None
        return (counts[1] + 1) / (sum(counts) + 2)

    def predict(self, events, selected=None):
        events = bits(events, 5)
        if selected is None:
            selected, visits, checks = self.select()
        else:
            if (not isinstance(selected, tuple) or len(selected) > 3 or len(set(selected)) != len(selected)
                    or any(type(i) is not int or not 0 <= i < len(self.routes) for i in selected)):
                raise ValueError("Invalid factor selection")
            visits = checks = 0
        packets = tuple((self.routes[i][1], self.probability(i, events)) for i in selected)
        return synthesize(packets), {"selected": selected, "packets": packets,
                                    "score_visits": visits, "overlap_checks": checks}

    def begin(self, events, time):
        events = bits(events, 5)
        if self.pending is not None or type(time) is not int or time <= self.clock:
            raise ValueError("Pending receipt or nonmonotonic time")
        prediction, trace = self.predict(events)
        probabilities = tuple(self.probability(i, events) for i in range(len(self.routes)))
        self.pending = (time, events, probabilities)
        self.clock = time
        return prediction, trace

    def observe(self, outcome, time):
        outcome = bits(outcome, 3)
        if self.pending is None or type(time) is not int or time != self.pending[0] + 1:
            raise ValueError("Outcome must follow its pending prediction by one tick")
        _, events, probabilities = self.pending
        for i, (inputs, slot) in enumerate(self.routes):
            key = tuple(events[j] for j in inputs)
            table = self.tables[i]
            if key not in table:
                # Binary keys bound each table independently of stream duration.
                if len(table) >= 2 ** len(inputs):
                    raise AssertionError("Binary key capacity violated")
                table[key] = [0, 0]
            probability = probabilities[i] if probabilities[i] is not None else 0.5
            residual = (probability - outcome[slot]) ** 2
            self.errors[i] += (residual - self.errors[i]) / 32
            table[key][outcome[slot]] += 1
            self.observations[i] += 1
        self.updates += len(self.routes)
        self.pending = None


def reference_predict(model, events, selected):
    """Independent scalar readout of frozen count tables, including abstention."""
    values = [[], [], []]
    for i in selected:
        inputs, slot = model.routes[i]
        key = tuple(events[j] for j in inputs)
        for stored_key, counts in model.tables[i].items():
            if key == stored_key:
                values[slot].append((1 + counts[1]) / (2 + counts[0] + counts[1]))
                break
    result = []
    for group in values:
        if not group or (min(group) < 0.5 <= max(group)):
            result.append(None)
        else:
            result.append(sum(sorted(group)) / len(group))
    return tuple(result)


def route_overlap(model, selected):
    pairs = list(combinations(selected, 2))
    return (sum(bool(set(model.routes[a][0]) & set(model.routes[b][0])) for a, b in pairs)
            / len(pairs)) if pairs else 0.0


def evaluate(model, samples, selected=None):
    before = deepcopy(model.__dict__)
    digest = sha256()
    exact = components = covered = squared = visits = checks = 0
    active = overlap = 0.0
    usage = [0, 0, 0]
    reference_equal = permutation_equal = True
    for row in samples:
        probabilities, trace = model.predict(row.events, selected)
        predicted = tuple(None if p is None else int(p >= 0.5) for p in probabilities)
        exact += predicted == row.outcome
        components += sum(a == b for a, b in zip(predicted, row.outcome))
        covered += sum(p is not None for p in probabilities)
        squared += sum(((p if p is not None else 0.5) - y) ** 2
                       for p, y in zip(probabilities, row.outcome))
        visits += trace["score_visits"]
        checks += trace["overlap_checks"]
        active += sum(p is not None for _, p in trace["packets"])
        overlap += route_overlap(model, trace["selected"])
        for slot, probability in trace["packets"]:
            usage[slot] += probability is not None
        reference_equal &= probabilities == reference_predict(model, row.events, trace["selected"])
        permutation_equal &= probabilities == synthesize(tuple(reversed(trace["packets"])))
        digest.update(json.dumps([row.events, row.outcome, probabilities, trace["selected"]]).encode() + b"\n")
    if model.__dict__ != before:
        raise AssertionError("Frozen evaluation changed state")
    count, total = len(samples), sum(usage)
    entropy = -sum((n / total) * log2(n / total) for n in usage if n) if total else 0.0
    return {"samples": count, "exact_accuracy": exact / count,
            "component_accuracy": components / (3 * count), "brier": squared / (3 * count),
            "component_coverage": covered / (3 * count), "active_factors_mean": active / count,
            "input_overlap_rate": overlap / count, "output_usage_entropy": entropy,
            "candidate_score_visits": visits, "overlap_pair_checks": checks,
            "scalar_reference_equal": reference_equal, "packet_permutation_equal": permutation_equal,
            "trace_sha256": digest.hexdigest()}


def train(model, rows, config):
    peak_bytes = peak_slots = peak_updates = 0
    correct, curve, digest = 0, [], sha256()
    start = process_time()
    for t, row in enumerate(rows):
        prediction, trace = model.begin(row.events, t)
        correct += tuple(None if p is None else int(p >= 0.5) for p in prediction) == row.outcome
        digest.update(json.dumps([prediction, trace["selected"]]).encode() + b"\n")
        peak_bytes = max(peak_bytes, deep_size(model.__dict__))
        peak_slots = max(peak_slots, scalar_slots(model.__dict__))
        previous = model.updates
        model.observe(row.outcome, t + 1)
        peak_updates = max(peak_updates, model.updates - previous)
        peak_bytes = max(peak_bytes, deep_size(model.__dict__))
        peak_slots = max(peak_slots, scalar_slots(model.__dict__))
        if (peak_bytes > config["max_predictor_state_bytes"]
                or peak_slots > config["max_logical_scalar_slots"]
                or peak_updates > config["max_outcome_updates_per_tick"]):
            raise AssertionError("Registered resource ceiling exceeded")
        if (t + 1) % 64 == 0:
            chosen = model.select()[0]
            curve.append({"observations": t + 1, "prequential_exact": correct / (t + 1),
                          "selected_routes": chosen, "input_overlap_rate": route_overlap(model, chosen)})
    return {"cpu_seconds_including_instrumentation": process_time() - start,
            "peak_predictor_bytes": peak_bytes, "peak_logical_slots": peak_slots,
            "peak_outcome_updates_per_tick": peak_updates, "outcome_updates": model.updates,
            "training_trace_sha256": digest.hexdigest(), "learning_curve": curve}


def run_case(seed, task_name, config):
    task = Task(seed, task_name)
    training = task.samples("train", config["training_steps"])
    evaluation = {name: task.samples(name, config["evaluation_steps_per_split"]) for name in SPLITS}
    models, arms = {}, {}
    for arm in ARMS:
        model = LocalRouteMemory(arm, task.provided_routes() if arm == "provided_routes" else None)
        resources = train(model, training, config)
        results = {split: evaluate(model, rows) for split, rows in evaluation.items()}
        chosen = model.select()[0]
        arms[arm] = {"training": resources, "splits": results,
                     "selected_routes": [model.routes[i] for i in chosen],
                     "selected_local_errors": [model.errors[i] for i in chosen],
                     "replay_equal": results == {split: evaluate(deepcopy(model), rows)
                                                 for split, rows in evaluation.items()}}
        models[arm] = model
    shuffled_rows = [row.outcome for row in training]
    random.Random(seed + 200000).shuffle(shuffled_rows)
    shuffled = LocalRouteMemory("learned_full_inhibition")
    train(shuffled, [Sample(row.events, outcome) for row, outcome in zip(training, shuffled_rows)], config)
    reset = deepcopy(models["learned_full_inhibition"])
    selected = reset.select()[0]
    if selected:
        reset.tables[selected[-1]].clear()
    controls = {
        "untrained": evaluate(LocalRouteMemory("learned_full_inhibition"), evaluation["nuisance_shift"]),
        "shuffled_training_outcomes": evaluate(shuffled, evaluation["nuisance_shift"]),
        "reset_selected_factor": evaluate(reset, evaluation["nuisance_shift"], selected),
    }
    full = arms["learned_full_inhibition"]["splits"]
    plain = arms["learned_no_inhibition"]["splits"]
    output = arms["learned_output_inhibition"]["splits"]
    joint = arms["joint_single"]["splits"]
    limits = config["gates"]
    reference = all(m["scalar_reference_equal"] and m["packet_permutation_equal"]
                    for arm in arms.values() for m in arm["splits"].values())
    integrity = {"frozen_replay": all(arm["replay_equal"] for arm in arms.values()),
                 "scalar_and_packet_equivalence": reference,
                 "untrained_abstention": controls["untrained"]["component_coverage"] == 0,
                 "resource_ceilings": all(
                     arm["training"]["peak_predictor_bytes"] <= config["max_predictor_state_bytes"]
                     and arm["training"]["peak_logical_slots"] <= config["max_logical_scalar_slots"]
                     and arm["training"]["peak_outcome_updates_per_tick"] <= config["max_outcome_updates_per_tick"]
                     for arm in arms.values())}
    gates = {
        "novel_accuracy": full["novel_combination"]["exact_accuracy"] >= limits["novel_exact_min"],
        "shift_accuracy": full["nuisance_shift"]["exact_accuracy"] >= limits["shift_exact_min"],
        "intervention_accuracy": full["intervention_pairs"]["exact_accuracy"] >= limits["intervention_exact_min"],
        "novel_gain_over_joint": full["novel_combination"]["exact_accuracy"] - joint["novel_combination"]["exact_accuracy"] >= limits["novel_gain_over_joint_min"],
        "gain_over_no_inhibition": full["nuisance_shift"]["exact_accuracy"] - plain["nuisance_shift"]["exact_accuracy"] >= limits["shift_gain_over_no_inhibition_min"],
        "input_inhibition_increment": full["nuisance_shift"]["exact_accuracy"] - output["nuisance_shift"]["exact_accuracy"] >= limits["full_gain_over_output_only_min"],
        "input_overlap_reduction": plain["nuisance_shift"]["input_overlap_rate"] - full["nuisance_shift"]["input_overlap_rate"] >= limits["input_overlap_reduction_min"],
        "reset_effect": full["nuisance_shift"]["exact_accuracy"] - controls["reset_selected_factor"]["exact_accuracy"] >= limits["reset_exact_drop_min"],
        "outcome_alignment": full["nuisance_shift"]["exact_accuracy"] - controls["shuffled_training_outcomes"]["exact_accuracy"] >= limits["shuffle_exact_drop_min"],
    }
    return {"seed": seed, "task": task_name, "arms": arms, "controls": controls,
            "integrity": integrity, "specialization_gates": gates,
            "specialization_passed": all(integrity.values()) and all(gates.values())}
