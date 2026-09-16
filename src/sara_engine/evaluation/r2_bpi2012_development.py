"""Development-only sparse multiclass evaluation on BPI Challenge 2012."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import time

from sara_engine.evaluation.r1_temporal_learning_benchmark import _deep_size
from sara_engine.learning.sparse_multiclass import BoundedSparseMulticlassReadout, SparseMulticlassConfig
from sara_engine.neuro.neuron import Neuron
from sara_engine.utils.project_paths import processed_data_path
from sara_engine.evaluation.bpi2012_data import Trace, gap_bucket, load_traces, split_name


PROTOCOL_PATH = processed_data_path("benchmark_fixtures", "r2_bpi2012_model_v1.json")
PROTOCOL_SHA256 = "3d8f87797fdf33c772720b5706cc8d9d4c8de48d8c6dc72b61ed308c7849f940"


def load_protocol() -> dict:
    raw = Path(PROTOCOL_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen BPI 2012 model protocol changed")
    return json.loads(raw)


class EventRouteEncoder:
    def __init__(self, protocol: dict, *, spiking: bool, constant_gap: bool = False) -> None:
        self.protocol = protocol; self.spiking = spiking; self.constant_gap = constant_gap
        self.routes = {}; self.units = []

    def _route(self, key: tuple) -> int:
        if key not in self.routes:
            if len(self.routes) >= self.protocol["learning"]["maximum_route_vocabulary"]:
                raise ValueError("Route vocabulary exceeded")
            route = len(self.routes); self.routes[key] = route
            if self.spiking: self.units.append(Neuron(route, num_branches=1))
        return self.routes[key]

    def encode(self, trace: Trace, index: int) -> tuple[int, ...]:
        event = trace.events[index]
        previous = trace.events[index - 1].activity if index else "<BOS>"
        previous2 = trace.events[index - 2].activity if index > 1 else "<BOS2>"
        gap = 0.0 if index == 0 else (event.timestamp - trace.events[index - 1].timestamp).total_seconds()
        bucket = 1 if self.constant_gap and gap > 0 else gap_bucket(gap)
        prefix_bucket = bisect_left(self.protocol["prefix_length_bounds"], index + 1)
        keys = (("a", event.activity), ("aa", previous, event.activity),
                ("aaa", previous2, previous, event.activity), ("al", event.activity, event.lifecycle),
                ("alg", event.activity, event.lifecycle, bucket), ("aag", previous, event.activity, bucket),
                ("hw", event.timestamp.hour // 4, event.timestamp.weekday()), ("p", prefix_bucket))
        result = []
        for key in keys:
            route = self._route(key)
            if self.spiking:
                unit = self.units[route]; unit.v = 0.0; unit.spike = False; unit.refractory_time = 0; unit.active_branches.clear()
                unit.add_input_to_branch(0, 1.6)
                if not unit.step(): raise RuntimeError("BPI feature neuron did not spike")
            result.append(route)
        return tuple(result)


def _metrics(rows: list[dict], labels: tuple[str, ...], rare: set[str]) -> dict:
    support = Counter(row["target"] for row in rows); predicted = Counter(row["predicted"] for row in rows)
    tp = Counter(row["target"] for row in rows if row["target"] == row["predicted"])
    recalls = {}; f1s = []
    for label in labels:
        recall = tp[label] / support[label] if support[label] else 0.0
        precision = tp[label] / predicted[label] if predicted[label] else 0.0
        recalls[label] = recall
        f1s.append(0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision))
    rare_values = [recalls[label] for label in rare]
    return {"count": len(rows), "top1_accuracy": sum(row["target"] == row["predicted"] for row in rows) / len(rows),
            "macro_f1": sum(f1s) / len(f1s), "multiclass_brier": sum(row["brier"] for row in rows) / len(rows),
            "per_activity_recall": recalls, "rare_activity_mean_recall": sum(rare_values) / len(rare_values) if rare_values else 0.0}


def run_arm(protocol: dict, arm: str, traces: list[Trace], labels: tuple[str, ...], rare: set[str]) -> dict:
    scalar = arm == "scalar_sparse_mistake"
    encoder = EventRouteEncoder(protocol, spiking=not scalar, constant_gap=arm == "snn_constant_gap")
    rate = 0.0 if arm == "snn_frozen" else protocol["learning"]["learning_rate"]
    learner = BoundedSparseMulticlassReadout(labels, SparseMulticlassConfig(rate, protocol["learning"]["weight_cap"],
        protocol["learning"]["maximum_route_vocabulary"], protocol["learning"]["maximum_active_routes"],
        protocol["learning"]["maximum_class_count"]))
    all_targets = [trace.events[index + 1].activity for trace in traces if split_name(trace) != "frozen_test"
                   for index in range(len(trace.events) - 1)]
    shuffled = list(all_targets); random.Random(protocol["shuffled_outcome_seed"]).shuffle(shuffled); feedback_index = 0
    rows = []; latencies = []; max_work = 0
    for trace in traces:
        split = split_name(trace)
        if split == "frozen_test": continue
        for index in range(len(trace.events) - 1):
            started = time.perf_counter_ns(); active = encoder.encode(trace, index); receipt = learner.predict(active)
            probabilities = learner.probabilities(receipt); true_target = trace.events[index + 1].activity
            target = shuffled[feedback_index] if arm == "snn_shuffled_outcomes" else true_target; feedback_index += 1
            update = learner.observe(receipt, target)
            elapsed = (time.perf_counter_ns() - started) / 1e6; latencies.append(elapsed)
            max_work = max(max_work, len(active) * (5 if encoder.spiking else 1) + len(labels) * len(active) + update.updates)
            if split == "development":
                rows.append({"target": true_target, "predicted": receipt.predicted,
                             "brier": sum((probabilities[label] - int(label == true_target)) ** 2 for label in labels)})
    ordered = sorted(latencies); percentile = lambda q: ordered[min(len(ordered)-1, int(q*(len(ordered)-1)))]
    state = _deep_size((encoder, learner)); rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    budget = protocol["resource_budgets"]
    contracts = (len(encoder.units) <= budget["max_neurons"] and len(learner.snapshot()["weights"]) <= budget["max_weight_entries"]
                 and state <= budget["max_state_bytes"] and max_work <= budget["max_event_work_per_prediction"]
                 and percentile(.99) <= budget["cpu_latency_p99_ms"] and max(ordered) <= budget["cpu_latency_watchdog_ms"]
                 and rss <= budget["max_peak_rss_bytes"])
    digest = hashlib.sha256(json.dumps([(row["target"],row["predicted"]) for row in rows],separators=(",",":")).encode()).hexdigest()
    return {"development": _metrics(rows, labels, rare), "prediction_trace_sha256": digest,
            "resources": {"neurons": len(encoder.units), "routes": len(encoder.routes), "weight_entries": len(learner.snapshot()["weights"]),
                          "state_bytes": state, "max_event_work": max_work, "latency_p50_ms": percentile(.5),
                          "latency_p95_ms": percentile(.95), "latency_p99_ms": percentile(.99), "latency_watchdog_ms": max(ordered),
                          "peak_rss_bytes": rss, "contracts_passed": contracts}}


def run_second_order(traces: list[Trace], labels: tuple[str, ...], rare: set[str]) -> dict:
    counts = defaultdict(Counter); rows=[]
    for trace in traces:
        split=split_name(trace)
        if split=="frozen_test": continue
        for index in range(len(trace.events)-1):
            current=trace.events[index].activity; previous=trace.events[index-1].activity if index else "<BOS>"; target=trace.events[index+1].activity
            table=counts[(previous,current)]; predicted=max(labels,key=lambda label:(table[label],-labels.index(label)))
            total=sum(table.values())+len(labels); probs={label:(table[label]+1)/total for label in labels}
            if split=="development": rows.append({"target":target,"predicted":predicted,"brier":sum((probs[label]-int(label==target))**2 for label in labels)})
            table[target]+=1
    return {"development":_metrics(rows,labels,rare),"resources":{"contracts_passed":True}}


def decide(protocol: dict, arms: dict) -> dict:
    gate=protocol["development_acceptance"]; c=arms["snn_sparse_mistake"]["development"]; b=arms["online_second_order_transition"]["development"]
    checks={"candidate_accuracy":c["top1_accuracy"]>=gate["minimum_candidate_top1_accuracy"],
        "accuracy_gain":c["top1_accuracy"]-b["top1_accuracy"]>=gate["minimum_top1_gain_over_second_order"],
        "candidate_macro_f1":c["macro_f1"]>=gate["minimum_candidate_macro_f1"],
        "macro_f1_gain":c["macro_f1"]-b["macro_f1"]>=gate["minimum_macro_f1_gain_over_second_order"],
        "timing_ablation":c["top1_accuracy"]-arms["snn_constant_gap"]["development"]["top1_accuracy"]>=gate["minimum_constant_gap_accuracy_drop"],
        "shuffled_gap":c["top1_accuracy"]-arms["snn_shuffled_outcomes"]["development"]["top1_accuracy"]>=gate["minimum_shuffled_outcome_accuracy_gap"],
        "rare_recall":c["rare_activity_mean_recall"]>=b["rare_activity_mean_recall"]*gate["minimum_rare_activity_recall_ratio_to_second_order"],
        "scalar_equivalence":arms["snn_sparse_mistake"]["prediction_trace_sha256"]==arms["scalar_sparse_mistake"]["prediction_trace_sha256"],
        "resources":all(result["resources"]["contracts_passed"] for result in arms.values())}
    return {"checks":checks,"passed":all(checks.values()),"frozen_test_authorized":all(checks.values()),"status":"development_pass" if all(checks.values()) else "development_negative_result"}


def run_development(protocol: dict) -> dict:
    traces=load_traces(); labels=tuple(sorted({event.activity for trace in traces if split_name(trace)!="frozen_test" for event in trace.events}))
    support=Counter(trace.events[index+1].activity for trace in traces if split_name(trace)=="development" for index in range(len(trace.events)-1))
    total=sum(support.values()); rare={label for label,count in support.items() if count/total<.01}
    arms={arm:run_arm(protocol,arm,traces,labels,rare) for arm in protocol["arms"] if arm!="online_second_order_transition"}
    arms["online_second_order_transition"]=run_second_order(traces,labels,rare)
    return {"schema":"sara-r2-bpi2012-development-result-v1","experiment_id":protocol["experiment_id"],"rare_activities":sorted(rare),"arms":arms,"decision":decide(protocol,arms)}


__all__=["decide","load_protocol","run_development"]
