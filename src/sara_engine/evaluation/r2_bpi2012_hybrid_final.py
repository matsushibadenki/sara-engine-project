"""Final frozen-test evaluation for the fixed BPI 2012 confidence hybrid."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import resource
import time

from sara_engine.evaluation.bpi2012_data import load_traces, split_name
from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.evaluation.r2_bpi2012_development import EventRouteEncoder, _metrics
from sara_engine.learning.confidence_router import ConfidenceRouteConfig, route_prediction
from sara_engine.learning.sparse_multiclass import BoundedSparseMulticlassReadout, SparseMulticlassConfig
from sara_engine.utils.project_paths import processed_data_path


PROTOCOL_PATH = processed_data_path("benchmark_fixtures", "r2_bpi2012_hybrid_final_v1.json")
PROTOCOL_SHA256 = "5005b87c100290f97f2c457827e8635ebe3a22abd5dead89da1fac1c36fa3dcf"


def load_protocol() -> dict:
    raw = Path(PROTOCOL_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen BPI 2012 final hybrid protocol changed")
    return json.loads(raw)


def _base(table: Counter, labels: tuple[str, ...]) -> tuple[str, dict[str, float]]:
    total = sum(table.values()) + len(labels)
    probabilities = {label: (table[label] + 1.0) / total for label in labels}
    return max(labels, key=lambda label: (probabilities[label], -labels.index(label))), probabilities


def _run_arm(protocol: dict, arm: str) -> dict:
    traces = load_traces()
    labels = tuple(sorted({event.activity for trace in traces if split_name(trace) != "frozen_test" for event in trace.events}))
    parent = json.loads(Path(processed_data_path("benchmark_fixtures", "r2_bpi2012_model_v1.json")).read_text())
    scalar = arm == "scalar_hybrid"
    encoder = EventRouteEncoder(parent, spiking=not scalar, constant_gap=arm == "constant_gap_hybrid")
    settings = parent["learning"]
    learner = BoundedSparseMulticlassReadout(labels, SparseMulticlassConfig(
        settings["learning_rate"], settings["weight_cap"], settings["maximum_route_vocabulary"],
        settings["maximum_active_routes"], settings["maximum_class_count"]))
    selected = protocol["selected_router"]
    config = ConfidenceRouteConfig(selected["base_probability_cap"], selected["local_score_margin"], selected["minimum_base_support"])
    tables = defaultdict(Counter)
    targets = [trace.events[index + 1].activity for trace in traces for index in range(len(trace.events) - 1)]
    feedback = list(targets)
    if arm == "shuffled_outcome_hybrid":
        random.Random(protocol["shuffled_outcome_seed"]).shuffle(feedback)
    feedback_index = 0
    rows = []
    base_rows = []
    overrides = 0
    latencies = []
    max_work = 0
    for trace in traces:
        evaluate = split_name(trace) == "frozen_test"
        for index in range(len(trace.events) - 1):
            started = time.perf_counter_ns()
            current = trace.events[index].activity
            previous = trace.events[index - 1].activity if index else "<BOS>"
            table = tables[(previous, current)]
            base_predicted, base_probabilities = _base(table, labels)
            active = encoder.encode(trace, index)
            receipt = learner.predict(active)
            decision = route_prediction(labels=labels, base_predicted=base_predicted,
                base_probabilities=base_probabilities, base_support=sum(table.values()),
                local_predicted=receipt.predicted, local_scores=dict(receipt.scores), config=config)
            target = trace.events[index + 1].activity
            update = learner.observe(receipt, feedback[feedback_index] if arm == "shuffled_outcome_hybrid" else target)
            feedback_index += 1
            table[target] += 1
            elapsed = (time.perf_counter_ns() - started) / 1e6
            latencies.append(elapsed)
            max_work = max(max_work, len(active) * (5 if not scalar else 1) + len(labels) * len(active) + update.updates + 16)
            if evaluate:
                probabilities = dict(decision.probabilities)
                rows.append({"target": target, "predicted": decision.predicted,
                    "brier": sum((probabilities[label] - int(label == target)) ** 2 for label in labels)})
                base_rows.append({"target": target, "predicted": base_predicted,
                    "brier": sum((base_probabilities[label] - int(label == target)) ** 2 for label in labels)})
                overrides += int(decision.overridden)
    ordered = sorted(latencies)
    percentile = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]
    state_bytes = _deep_size((encoder, learner, tables))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gate = protocol["final_acceptance"]
    resources = {"routes": len(encoder.routes), "neurons": len(encoder.units),
        "weight_entries": len(learner.snapshot()["weights"]), "state_bytes": state_bytes,
        "max_event_work": max_work, "latency_p50_ms": percentile(.5), "latency_p95_ms": percentile(.95),
        "latency_p99_ms": percentile(.99), "latency_watchdog_ms": max(ordered), "peak_rss_bytes": rss}
    resources["contracts_passed"] = (resources["routes"] <= gate["max_routes"]
        and resources["neurons"] <= gate["max_neurons"] and resources["weight_entries"] <= gate["max_weight_entries"]
        and state_bytes <= gate["max_state_bytes"] and max_work <= gate["max_event_work"]
        and resources["latency_p99_ms"] <= gate["cpu_latency_p99_ms"]
        and resources["latency_watchdog_ms"] <= gate["cpu_latency_watchdog_ms"] and rss <= gate["max_peak_rss_bytes"])
    metrics = _metrics(rows, labels, set())
    metrics.update({"overrides": overrides, "override_rate": overrides / len(rows),
        "prediction_trace_sha256": hashlib.sha256(json.dumps([(row["target"], row["predicted"]) for row in rows], separators=(",", ":")).encode()).hexdigest()})
    return {"frozen_test": metrics, "base_frozen_test": _metrics(base_rows, labels, set()), "resources": resources}


def run_final(protocol: dict) -> dict:
    arms = {arm: _run_arm(protocol, arm) for arm in protocol["arms"] if arm != "online_second_order_transition"}
    candidate = arms["snn_hybrid"]["frozen_test"]
    base = arms["snn_hybrid"]["base_frozen_test"]
    gate = protocol["final_acceptance"]
    checks = {
        "accuracy": candidate["top1_accuracy"] - base["top1_accuracy"] >= gate["minimum_top1_gain_over_base"],
        "macro_f1": candidate["macro_f1"] - base["macro_f1"] >= gate["minimum_macro_f1_gain_over_base"],
        "brier": candidate["multiclass_brier"] - base["multiclass_brier"] <= gate["maximum_brier_increase_over_base"],
        "timing": candidate["macro_f1"] - arms["constant_gap_hybrid"]["frozen_test"]["macro_f1"] >= gate["minimum_constant_gap_macro_f1_drop"],
        "shuffled": candidate["macro_f1"] - arms["shuffled_outcome_hybrid"]["frozen_test"]["macro_f1"] >= gate["minimum_shuffled_outcome_macro_f1_drop"],
        "scalar_equivalence": candidate["prediction_trace_sha256"] == arms["scalar_hybrid"]["frozen_test"]["prediction_trace_sha256"],
        "resources": all(arm["resources"]["contracts_passed"] for arm in arms.values()),
    }
    return {"schema": "sara-r2-bpi2012-confidence-hybrid-final-result-v1", "experiment_id": protocol["experiment_id"],
        "arms": arms, "decision": {"checks": checks, "passed": all(checks.values()),
        "status": "final_pass" if all(checks.values()) else "final_negative_result", "production_authorized": False}}


__all__ = ["load_protocol", "run_final"]
