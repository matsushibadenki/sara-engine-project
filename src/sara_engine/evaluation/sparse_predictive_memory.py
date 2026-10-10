"""Isolated Memory Mosaics-inspired sparse associative-memory experiment.

Input slots supply factor identity. This tests compositional memory reuse, not
learned factor discovery, inhibition, or a spike-specific learning advantage.
Only integer event IDs, bounded dictionaries and local outcome writes are used.
"""
from copy import deepcopy
from hashlib import sha256
import json
import random
import sys
from time import process_time


ARMS = ("joint", "provided_factors", "redundant_factors")
PROTOCOL_SHA256 = "767a13e47a906844b03b70b5ab5bf221ab46c54af8d844d6cc2a274e8607532f"


def packet(value):
    """Validate before lookup or mutation; never consume an unbounded iterable."""
    if (not isinstance(value, (tuple, list)) or len(value) != 3
            or any(type(item) is not int or not 0 <= item < 256 for item in value)):
        raise ValueError("Expected three integer event IDs in [0, 256)")
    return tuple(value)


def deep_size(value, seen=None):
    """Interpreter-dependent retained bytes, not physical energy or peak RSS."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(deep_size(item, seen) for item in value)
    return size


class PredictiveMemory:
    """One pending causal receipt; FIFO retention with a common scalar budget."""

    def __init__(self, arm, budget=132):
        if arm not in ARMS or type(budget) is not int or not 12 <= budget <= 4096:
            raise ValueError("Invalid arm or memory budget")
        self.arm, self.budget = arm, budget
        self.tables = [{} for _ in range(1 if arm == "joint" else 3)]
        self.record_scalars = {"joint": 6, "provided_factors": 2,
                               "redundant_factors": 4}[arm]
        self.capacity = budget // (len(self.tables) * self.record_scalars)
        self.pending = None
        self.clock = -1
        self.writes = self.lookups = self.evictions = 0

    def predict(self, events):
        events = packet(events)
        self.lookups += len(self.tables)
        if self.arm == "joint":
            return self.tables[0].get(events, (None, None, None))
        return tuple(table.get((events[i],) if self.arm == "provided_factors"
                               else events, (None,))[0]
                     for i, table in enumerate(self.tables))

    def begin(self, events, time):
        events = packet(events)
        if self.pending is not None or type(time) is not int or time <= self.clock:
            raise ValueError("Pending receipt or nonmonotonic time")
        prediction = self.predict(events)
        self.pending = (time, events)
        self.clock = time
        return prediction

    def observe(self, outcome, time):
        outcome = packet(outcome)
        if (self.pending is None or type(time) is not int
                or time != self.pending[0] + 1):
            raise ValueError("Outcome must arrive exactly one tick after prediction")
        events = self.pending[1]
        for i, table in enumerate(self.tables):
            key = (events[i],) if self.arm == "provided_factors" else events
            value = outcome if self.arm == "joint" else (outcome[i],)
            if key not in table and len(table) == self.capacity:
                del table[next(iter(table))]
                self.evictions += 1
            table[key] = value
        self.pending = None
        self.writes += 3

    def retained_scalars(self):
        return self.record_scalars * sum(map(len, self.tables))


def stream(seed, periods, count):
    """Evaluator-owned latent periods; the predictor sees only three event IDs."""
    if (len(periods) != 3 or any(type(p) is not int or not 2 <= p <= 64 for p in periods)
            or sum(periods) > 256 or not 1 <= count <= 10000):
        raise ValueError("Invalid synthetic stream bounds")
    rng = random.Random(seed)
    codes = list(range(sum(periods)))
    rng.shuffle(codes)
    channels, offset = [], 0
    for period in periods:
        channels.append(codes[offset:offset + period])
        offset += period
    phases = [rng.randrange(p) for p in periods]
    return [tuple(channels[i][(t + phases[i]) % p] for i, p in enumerate(periods))
            for t in range(count)]


def evaluate(model, rows, seen):
    before = deepcopy(model.tables)
    digest = sha256()
    exact = components = covered = novel = novel_exact = 0
    for current, outcome in zip(rows, rows[1:]):
        predicted = model.predict(current)
        hit = predicted == outcome
        exact += hit
        components += sum(a == b for a, b in zip(predicted, outcome))
        covered += sum(value is not None for value in predicted)
        if current not in seen:
            novel += 1
            novel_exact += hit
        digest.update(json.dumps([current, predicted, outcome]).encode() + b"\n")
    if model.tables != before:
        raise AssertionError("Frozen evaluation mutated memory")
    count = len(rows) - 1
    return {"steps": count, "exact_accuracy": exact / count,
            "component_accuracy": components / (3 * count),
            "component_coverage": covered / (3 * count),
            "novel_contexts": novel, "novel_exact_accuracy": novel_exact / novel if novel else None,
            "trace_sha256": digest.hexdigest()}


def run_seed(seed, protocol):
    steps = protocol["training_steps"]
    rows = stream(seed, protocol["periods"], steps + protocol["evaluation_steps"] + 1)
    seen = set(rows[:steps])
    evaluation_rows = rows[steps:]
    results, models = {}, {}
    for arm in ARMS:
        model = PredictiveMemory(arm, protocol["memory_scalar_budget"])
        start = process_time()
        peak = 0
        curve = []
        correct = 0
        first_correct = None
        for t in range(steps):
            prediction = model.begin(rows[t], t)
            peak = max(peak, deep_size(model.__dict__))
            correct += prediction == rows[t + 1]
            if first_correct is None and prediction == rows[t + 1]:
                first_correct = t + 1
            # The target becomes available only after the prediction receipt.
            model.observe(rows[t + 1], t + 1)
            peak = max(peak, deep_size(model.__dict__))
            curve.append({"observations": t + 1, "prequential_exact": correct / (t + 1)})
        training_cpu = process_time() - start
        train_lookups = model.lookups
        start = process_time()
        metrics = evaluate(model, evaluation_rows, seen)
        evaluation_cpu = process_time() - start
        evaluation_lookups = model.lookups - train_lookups
        replay = evaluate(deepcopy(model), evaluation_rows, seen)
        results[arm] = {**metrics, "training_cpu_seconds": training_cpu,
                        "evaluation_cpu_seconds": evaluation_cpu,
                        "peak_predictor_state_bytes": peak,
                        "scalar_budget": model.budget, "retained_scalars": model.retained_scalars(),
                        "outcome_scalar_writes": model.writes, "evictions": model.evictions,
                        "training_table_lookups": train_lookups,
                        "evaluation_table_lookups": evaluation_lookups,
                        "first_correct_prequential_observation": first_correct,
                        "learning_curve": curve, "replay_exact": replay == metrics}
        models[arm] = model

    reset = deepcopy(models["provided_factors"])
    reset.tables[1].clear()
    reset_result = evaluate(reset, evaluation_rows, seen)
    permutation = (2, 0, 1)
    permuted = PredictiveMemory("provided_factors", protocol["memory_scalar_budget"])
    permuted_rows = [tuple(row[i] for i in permutation) for row in rows]
    for t in range(steps):
        permuted.begin(permuted_rows[t], t)
        permuted.observe(permuted_rows[t + 1], t + 1)
    permutation_equal = all(
        tuple(permuted.predict(tuple(row[i] for i in permutation))[permutation.index(i)]
              for i in range(3)) == models["provided_factors"].predict(row)
        for row in evaluation_rows[:-1])
    shuffled = PredictiveMemory("provided_factors", protocol["memory_scalar_budget"])
    outcomes = list(rows[1:steps + 1])
    random.Random(seed + 1000000).shuffle(outcomes)
    for t in range(steps):
        shuffled.begin(rows[t], t)
        shuffled.observe(outcomes[t], t + 1)
    shuffled_result = evaluate(shuffled, evaluation_rows, seen)
    copy_accuracy = sum(a == b for a, b in zip(evaluation_rows, evaluation_rows[1:])) / (len(evaluation_rows) - 1)
    factor = results["provided_factors"]
    gates = protocol["gates"]
    checks = {
        "novel_composition": factor["novel_exact_accuracy"] >= gates["factor_novel_exact_min"],
        "novel_gain": factor["novel_exact_accuracy"] - results["joint"]["novel_exact_accuracy"] >= gates["factor_vs_joint_novel_gain_min"],
        "reset_effect": factor["exact_accuracy"] - reset_result["exact_accuracy"] >= gates["reset_exact_drop_min"],
        "outcome_alignment": factor["exact_accuracy"] - shuffled_result["exact_accuracy"] >= gates["shuffled_exact_drop_min"],
        "permutation_equivalence": permutation_equal,
        "replay": all(row["replay_exact"] for row in results.values()),
        "equal_write_budget": all(row["outcome_scalar_writes"] == steps * protocol["outcome_scalar_writes_per_step"] for row in results.values()),
        "bounded_state": all(row["retained_scalars"] <= protocol["memory_scalar_budget"] and row["peak_predictor_state_bytes"] <= gates["predictor_state_bytes_max"] for row in results.values()),
    }
    return {"seed": seed, "arms": results, "gates": checks,
            "controls": {"reset_middle_factor": reset_result,
                         "consistent_factor_permutation": permutation_equal,
                         "shuffled_training_outcomes": shuffled_result,
                         "copy_current_state_exact_accuracy": copy_accuracy}}
