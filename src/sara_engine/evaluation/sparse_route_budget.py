"""Matched actual-update study of bounded predictive candidate pruning."""
from copy import deepcopy
from hashlib import sha256
import json
import random
from time import process_time

from sara_engine.evaluation.sparse_factor_routing import (
    LocalRouteMemory, ROUTES, SPLITS, Sample, Task, bits, evaluate, scalar_slots,
)
from sara_engine.evaluation.sparse_predictive_memory import deep_size


PROTOCOL_SHA256 = "c8c99b539917e5e1f3847d5ab3e4f419ea80529d0be05fd4aa7d8cb0086eb7af"
ARMS = ("round_robin", "random_prune", "residual_prune", "overlap_prune")
WARMUP = 180
UPDATES = 9


class BudgetRouteMemory(LocalRouteMemory):
    """Nine causal count/error/observation updates, regardless of selection arm.

Inactive tables remain available for audit but cannot predict or learn.
Input sharing is permitted by residual pruning; overlap pruning is a control.
"""

    def __init__(self, strategy, seed):
        if strategy not in ARMS or type(seed) is not int:
            raise ValueError("Invalid strategy or seed")
        super().__init__("learned_output_inhibition")
        self.strategy, self.seed = strategy, seed
        self.pool = tuple(range(45))
        self.groups = []
        rng = random.Random(seed + 300000)
        for slot in range(3):
            group = [i for i, (_, output) in enumerate(ROUTES) if output == slot]
            rng.shuffle(group)
            self.groups.append(tuple(group))
        self.steps = 0
        self.pruned = False
        self.pruning_visits = self.pruning_overlap_checks = 0

    def cost(self, index):
        return self.errors[index] + 0.005 * len(self.routes[index][0])

    def narrow(self):
        if self.pruned or self.steps < WARMUP or self.strategy == "round_robin":
            return
        chosen = []
        if self.strategy == "random_prune":
            chosen = [i for group in self.groups for i in group[:3]]
        elif self.strategy == "residual_prune":
            for group in self.groups:
                self.pruning_visits += len(group)
                chosen.extend(sorted(group, key=lambda i: (self.cost(i), i))[:3])
        else:
            for _ in range(3):
                for group in self.groups:
                    scored = []
                    for i in group:
                        if i in chosen:
                            continue
                        self.pruning_visits += 1
                        penalty = 0.0
                        for j in chosen:
                            self.pruning_overlap_checks += 1
                            penalty += 0.25 * bool(set(self.routes[i][0]) & set(self.routes[j][0]))
                        scored.append((self.cost(i) + penalty, i))
                    chosen.append(min(scored)[1])
        self.pool = tuple(sorted(chosen))
        self.pruned = True

    def select(self):
        selected = []
        for slot in range(3):
            candidates = [i for i in self.pool if self.routes[i][1] == slot and self.observations[i] >= 16]
            if candidates:
                selected.append(min(candidates, key=lambda i: (self.cost(i), i)))
        return tuple(selected), len(self.pool), 0

    def schedule(self):
        if self.pruned:
            return self.pool
        offset = (self.steps * 3) % 15
        return tuple(group[(offset + j) % 15] for group in self.groups for j in range(3))

    def begin(self, events, time):
        events = bits(events, 5)
        if self.pending is not None or type(time) is not int or time <= self.clock:
            raise ValueError("Pending receipt or nonmonotonic time")
        self.narrow()
        prediction, trace = self.predict(events)
        active = self.schedule()
        if len(active) != UPDATES or len(set(active)) != UPDATES:
            raise AssertionError("Actual update budget violated")
        probabilities = tuple(self.probability(i, events) for i in active)
        self.pending = (time, events, active, probabilities)
        self.clock = time
        return prediction, {**trace, "updated_routes": active}

    def observe(self, outcome, time):
        outcome = bits(outcome, 3)
        if self.pending is None or type(time) is not int or time != self.pending[0] + 1:
            raise ValueError("Outcome must follow the pending receipt by one tick")
        _, events, active, probabilities = self.pending
        for i, probability in zip(active, probabilities):
            inputs, slot = self.routes[i]
            key = tuple(events[j] for j in inputs)
            counts = self.tables[i].setdefault(key, [0, 0])
            p = 0.5 if probability is None else probability
            self.errors[i] += ((p - outcome[slot]) ** 2 - self.errors[i]) / 32
            counts[outcome[slot]] += 1
            self.observations[i] += 1
        self.updates += len(active)
        self.steps += 1
        self.pending = None


def train_budget(model, rows, config):
    peak_bytes = peak_slots = correct = lookups = score_visits = 0
    curve, ledger = [], sha256()
    start = process_time()
    for t, row in enumerate(rows):
        prediction, trace = model.begin(row.events, t)
        before = tuple(model.observations)
        selected = trace["updated_routes"]
        lookups += len(selected) + len(trace["selected"])
        score_visits += trace["score_visits"]
        peak_bytes = max(peak_bytes, deep_size(model.__dict__))
        peak_slots = max(peak_slots, scalar_slots(model.__dict__))
        model.observe(row.outcome, t + 1)
        deltas = tuple(after - prior for after, prior in zip(model.observations, before))
        if deltas != tuple(int(i in selected) for i in range(45)):
            raise AssertionError("Unexpected active/inactive route update")
        ledger.update(json.dumps([t, selected, deltas]).encode() + b"\n")
        correct += tuple(None if p is None else int(p >= 0.5) for p in prediction) == row.outcome
        peak_bytes = max(peak_bytes, deep_size(model.__dict__))
        peak_slots = max(peak_slots, scalar_slots(model.__dict__))
        if peak_bytes > config["max_predictor_state_bytes"] or peak_slots > config["max_logical_scalar_slots"]:
            raise AssertionError("State ceiling exceeded")
        if (t + 1) % 64 == 0 or t + 1 == WARMUP:
            curve.append({"observations": t + 1, "exact_accuracy": correct / (t + 1),
                          "active_candidates": len(model.pool), "route_updates": model.updates})
    counts = sum(sum(counts) for table in model.tables for counts in table.values())
    actual_observations = sum(model.observations)
    if counts != len(rows) * UPDATES or actual_observations != counts or model.updates != counts:
        raise AssertionError("Actual outcome-update ledger mismatch")
    return {"cpu_seconds_including_instrumentation": process_time() - start,
            "outcome_route_updates": counts, "observation_increments": actual_observations,
            "count_error_observation_assignments": counts * 3,
            "per_tick_updates_verified": UPDATES, "update_ledger_sha256": ledger.hexdigest(),
            "training_probability_lookups": lookups, "training_score_visits": score_visits,
            "pruning_score_visits": model.pruning_visits,
            "pruning_overlap_checks": model.pruning_overlap_checks,
            "peak_predictor_bytes": peak_bytes, "peak_logical_slots": peak_slots,
            "active_candidates": len(model.pool), "learning_curve": curve}


def run_case(seed, name, config):
    task = Task(seed, name)
    training = task.samples("train", config["training_steps"])
    splits = {split: task.samples(split, config["evaluation_steps_per_split"]) for split in SPLITS}
    arms, models = {}, {}
    for strategy in ARMS:
        model = BudgetRouteMemory(strategy, seed)
        resources = train_budget(model, training, config)
        results = {split: evaluate(model, rows) for split, rows in splits.items()}
        arms[strategy] = {"training": resources, "splits": results,
                          "active_routes": [model.routes[i] for i in model.pool],
                          "selected_routes": [model.routes[i] for i in model.select()[0]],
                          "replay_equal": results == {split: evaluate(deepcopy(model), rows) for split, rows in splits.items()}}
        models[strategy] = model
    shuffled = [row.outcome for row in training]
    random.Random(seed + 400000).shuffle(shuffled)
    shuffled_model = BudgetRouteMemory("residual_prune", seed)
    shuffled_cost = train_budget(shuffled_model, [Sample(row.events, y) for row, y in zip(training, shuffled)], config)
    controls = {"shuffled_outcomes": evaluate(shuffled_model, splits["nuisance_shift"]),
                "shuffled_outcome_route_updates": shuffled_cost["outcome_route_updates"]}
    candidate = models["residual_prune"]
    # Ground-truth dependency inspection is evaluator-only, never a learner input.
    dependencies = task.provided_routes()
    shared = {i for i in range(5) if sum(i in key for key, _ in dependencies) > 1}
    controls["shared_input_channels"] = sorted(shared)
    controls["true_dependency_routes_retained"] = all(route in arms["residual_prune"]["active_routes"] for route in dependencies)
    if shared:
        masked = [Sample(tuple(0 if i in shared else bit for i, bit in enumerate(row.events)), row.outcome)
                  for row in splits["intervention_pairs"]]
        controls["shared_input_mask"] = evaluate(candidate, masked)
    residual, cyclic = arms["residual_prune"]["splits"], arms["round_robin"]["splits"]
    limits = config["per_case_gates"]
    integrity = {
        "equal_actual_updates": all(arm["training"]["outcome_route_updates"] == config["training_steps"] * config["updates_per_tick"] for arm in arms.values()),
        "frozen_replay": all(arm["replay_equal"] for arm in arms.values()),
        "scalar_and_packet_equivalence": all(m["scalar_reference_equal"] and m["packet_permutation_equal"] for arm in arms.values() for m in arm["splits"].values()),
        "state_ceilings": all(arm["training"]["peak_predictor_bytes"] <= config["max_predictor_state_bytes"] and arm["training"]["peak_logical_slots"] <= config["max_logical_scalar_slots"] for arm in arms.values()),
    }
    gates = {
        "shift_accuracy": residual["nuisance_shift"]["exact_accuracy"] >= limits["residual_shift_exact_min"],
        "novel_accuracy": residual["novel_combination"]["exact_accuracy"] >= limits["residual_novel_exact_min"],
        "intervention_accuracy": residual["intervention_pairs"]["exact_accuracy"] >= limits["residual_intervention_exact_min"],
        "brier_noninferiority": residual["nuisance_shift"]["brier"] - cyclic["nuisance_shift"]["brier"] <= limits["shift_brier_regression_max"],
        "pool_reduction": len(candidate.pool) == 9 and len(models["round_robin"].pool) == 45,
        "shared_dependencies_retained": not shared or controls["true_dependency_routes_retained"],
        "shared_input_mask_effect": not shared or residual["intervention_pairs"]["exact_accuracy"] - controls["shared_input_mask"]["exact_accuracy"] >= limits["shared_input_mask_accuracy_drop_min"],
        "outcome_alignment": residual["nuisance_shift"]["exact_accuracy"] - controls["shuffled_outcomes"]["exact_accuracy"] >= limits["shuffle_accuracy_drop_min"],
    }
    return {"seed": seed, "task": name, "arms": arms, "controls": controls,
            "integrity": integrity, "component_gates": gates}


def summarize(runs, config):
    count = len(runs)
    cyclic = sum(r["arms"]["round_robin"]["splits"]["nuisance_shift"]["brier"] for r in runs) / count
    residual = sum(r["arms"]["residual_prune"]["splits"]["nuisance_shift"]["brier"] for r in runs) / count
    gain = sum(r["arms"]["residual_prune"]["splits"]["nuisance_shift"]["exact_accuracy"] - r["arms"]["random_prune"]["splits"]["nuisance_shift"]["exact_accuracy"] for r in runs) / count
    reduction = (cyclic - residual) / cyclic if cyclic else 0.0
    gates = {"brier_improvement": reduction >= config["aggregate_gates"]["mean_shift_brier_reduction_fraction_min"],
             "gain_over_random": gain >= config["aggregate_gates"]["mean_shift_gain_over_random_min"]}
    integrity = all(all(run["integrity"].values()) for run in runs)
    return {"execution_integrity_passed": integrity,
            "component_passed": integrity and all(all(run["component_gates"].values()) for run in runs) and all(gates.values()),
            "aggregate_gates": gates, "mean_shift_brier_round_robin": cyclic,
            "mean_shift_brier_residual": residual, "brier_reduction_fraction": reduction,
            "mean_shift_accuracy_gain_over_random": gain}
