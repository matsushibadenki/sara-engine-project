"""Development-only evaluation of the inherited confidence hybrid on Sepsis Cases."""

from __future__ import annotations
from bisect import bisect_left
from collections import Counter, defaultdict
import hashlib, json, random, resource, time
from pathlib import Path

from sara_engine.evaluation.bpi2012_data import gap_bucket
from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.evaluation.r2_bpi2012_development import _metrics
from sara_engine.evaluation.sepsis_data import Trace, load_traces, split_name
from sara_engine.learning.confidence_router import ConfidenceRouteConfig, route_prediction
from sara_engine.learning.sparse_multiclass import BoundedSparseMulticlassReadout, SparseMulticlassConfig
from sara_engine.neuro.neuron import Neuron
from sara_engine.utils.project_paths import processed_data_path

PROTOCOL_PATH = processed_data_path("benchmark_fixtures", "r2_sepsis_hybrid_v1.json")
PROTOCOL_SHA256 = "f6cb79e10ca590c062f4f0f49e4ff3a26cd5dfc1173de55c44134fe68b0b5f77"

def load_protocol() -> dict:
    raw = Path(PROTOCOL_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen Sepsis hybrid protocol changed")
    return json.loads(raw)

class SepsisRouteEncoder:
    def __init__(self, *, spiking: bool, constant_gap: bool = False) -> None:
        self.spiking = spiking; self.constant_gap = constant_gap; self.routes = {}; self.units = []

    def _route(self, key: tuple) -> int:
        if key not in self.routes:
            if len(self.routes) >= 4096: raise ValueError("Route vocabulary exceeded")
            route = len(self.routes); self.routes[key] = route
            if self.spiking: self.units.append(Neuron(route, num_branches=1))
        return self.routes[key]

    def encode(self, trace: Trace, index: int) -> tuple[int, ...]:
        event = trace.events[index]
        previous = trace.events[index - 1].activity if index else "<BOS>"
        previous2 = trace.events[index - 2].activity if index > 1 else "<BOS2>"
        gap = 0.0 if index == 0 else (event.timestamp - trace.events[index - 1].timestamp).total_seconds()
        bucket = 1 if self.constant_gap and gap > 0 else gap_bucket(gap)
        prefix = bisect_left((1, 2, 4, 8, 16, 32, 64, 128), index + 1)
        keys = (("a", event.activity), ("aa", previous, event.activity), ("aaa", previous2, previous, event.activity),
            ("al", event.activity, event.lifecycle), ("alg", event.activity, event.lifecycle, bucket),
            ("aag", previous, event.activity, bucket), ("p", prefix))
        result = []
        for key in keys:
            route = self._route(key)
            if self.spiking:
                unit = self.units[route]; unit.v = 0.0; unit.spike = False; unit.refractory_time = 0; unit.active_branches.clear()
                unit.add_input_to_branch(0, 1.6)
                if not unit.step(): raise RuntimeError("Sepsis feature neuron did not spike")
            result.append(route)
        return tuple(result)

def _base(table: Counter, labels: tuple[str, ...]):
    total = sum(table.values()) + len(labels)
    probabilities = {label: (table[label] + 1.0) / total for label in labels}
    return max(labels, key=lambda label: (probabilities[label], -labels.index(label))), probabilities

def run_arm(protocol: dict, arm: str) -> dict:
    traces = load_traces(); labels = tuple(sorted({e.activity for t in traces if split_name(t) != "frozen_test" for e in t.events}))
    scalar = arm == "scalar_hybrid"; encoder = SepsisRouteEncoder(spiking=not scalar, constant_gap=arm == "constant_gap_hybrid")
    inherited = protocol["inherited_without_tuning"]
    learner = BoundedSparseMulticlassReadout(labels, SparseMulticlassConfig(inherited["learning_rate"], inherited["weight_cap"], 4096, 7, 16))
    config = ConfidenceRouteConfig(inherited["base_probability_cap"], inherited["local_score_margin"], inherited["minimum_base_support"])
    tables = defaultdict(Counter)
    targets = [t.events[i + 1].activity for t in traces if split_name(t) != "frozen_test" for i in range(len(t.events) - 1)]
    feedback = list(targets)
    if arm == "shuffled_outcome_hybrid": random.Random(protocol["shuffled_outcome_seed"]).shuffle(feedback)
    feedback_index = 0; rows = []; base_rows = []; overrides = 0; latencies = []; max_work = 0
    for trace in traces:
        split = split_name(trace)
        if split == "frozen_test": continue
        for index in range(len(trace.events) - 1):
            started = time.perf_counter_ns(); event = trace.events[index]
            previous = trace.events[index - 1].activity if index else "<BOS>"; table = tables[(previous, event.activity)]
            base_predicted, base_probabilities = _base(table, labels); active = encoder.encode(trace, index); receipt = learner.predict(active)
            decision = route_prediction(labels=labels, base_predicted=base_predicted, base_probabilities=base_probabilities,
                base_support=sum(table.values()), local_predicted=receipt.predicted, local_scores=dict(receipt.scores), config=config)
            target = trace.events[index + 1].activity
            update = learner.observe(receipt, feedback[feedback_index] if arm == "shuffled_outcome_hybrid" else target)
            feedback_index += 1; table[target] += 1
            latencies.append((time.perf_counter_ns() - started) / 1e6)
            max_work = max(max_work, len(active) * (5 if not scalar else 1) + len(labels) * len(active) + update.updates + 16)
            if split == "development":
                probabilities = dict(decision.probabilities)
                rows.append({"target": target, "predicted": decision.predicted, "brier": sum((probabilities[l] - int(l == target)) ** 2 for l in labels)})
                base_rows.append({"target": target, "predicted": base_predicted, "brier": sum((base_probabilities[l] - int(l == target)) ** 2 for l in labels)})
                overrides += int(decision.overridden)
    ordered = sorted(latencies); percentile = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]
    state = _deep_size((encoder, learner, tables)); rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; budget = protocol["resource_budgets"]
    resources = {"routes": len(encoder.routes), "neurons": len(encoder.units), "weight_entries": len(learner.snapshot()["weights"]),
        "state_bytes": state, "max_event_work": max_work, "latency_p50_ms": percentile(.5), "latency_p95_ms": percentile(.95),
        "latency_p99_ms": percentile(.99), "latency_watchdog_ms": max(ordered), "peak_rss_bytes": rss}
    resources["contracts_passed"] = (resources["routes"] <= budget["max_routes"] and resources["neurons"] <= budget["max_neurons"]
        and resources["weight_entries"] <= budget["max_weight_entries"] and state <= budget["max_state_bytes"] and max_work <= budget["max_event_work"]
        and resources["latency_p99_ms"] <= budget["cpu_latency_p99_ms"] and resources["latency_watchdog_ms"] <= budget["cpu_latency_watchdog_ms"]
        and rss <= budget["max_peak_rss_bytes"])
    metrics = _metrics(rows, labels, set()); metrics.update({"overrides": overrides, "override_rate": overrides / len(rows),
        "prediction_trace_sha256": hashlib.sha256(json.dumps([(r["target"], r["predicted"]) for r in rows], separators=(",", ":")).encode()).hexdigest()})
    return {"development": metrics, "base_development": _metrics(base_rows, labels, set()), "resources": resources}

def run_development(protocol: dict) -> dict:
    arms = {arm: run_arm(protocol, arm) for arm in protocol["arms"] if arm != "online_second_order_transition"}
    candidate = arms["snn_hybrid"]["development"]; base = arms["snn_hybrid"]["base_development"]; gate = protocol["development_acceptance"]
    checks = {"accuracy": candidate["top1_accuracy"] - base["top1_accuracy"] >= gate["minimum_top1_gain_over_base"],
        "macro_f1": candidate["macro_f1"] - base["macro_f1"] >= gate["minimum_macro_f1_gain_over_base"],
        "brier": candidate["multiclass_brier"] - base["multiclass_brier"] <= gate["maximum_brier_increase_over_base"],
        "timing": candidate["macro_f1"] - arms["constant_gap_hybrid"]["development"]["macro_f1"] >= gate["minimum_constant_gap_macro_f1_drop"],
        "shuffled": candidate["macro_f1"] - arms["shuffled_outcome_hybrid"]["development"]["macro_f1"] >= gate["minimum_shuffled_outcome_macro_f1_drop"],
        "scalar_equivalence": candidate["prediction_trace_sha256"] == arms["scalar_hybrid"]["development"]["prediction_trace_sha256"],
        "resources": all(a["resources"]["contracts_passed"] for a in arms.values())}
    return {"schema": "sara-r2-sepsis-confidence-hybrid-development-result-v1", "experiment_id": protocol["experiment_id"], "arms": arms,
        "decision": {"checks": checks, "passed": all(checks.values()), "status": "development_pass" if all(checks.values()) else "development_negative_result",
            "frozen_test_authorized": all(checks.values()), "production_authorized": False}}

__all__ = ["SepsisRouteEncoder", "load_protocol", "run_development"]
