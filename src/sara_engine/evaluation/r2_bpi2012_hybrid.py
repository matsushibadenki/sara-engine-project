"""Training-only confidence-hybrid diagnostic for BPI Challenge 2012."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from sara_engine.evaluation.bpi2012_data import load_traces, split_name
from sara_engine.evaluation.r2_bpi2012_development import EventRouteEncoder, _metrics
from sara_engine.learning.confidence_router import ConfidenceRouteConfig, route_prediction
from sara_engine.learning.sparse_multiclass import BoundedSparseMulticlassReadout, SparseMulticlassConfig
from sara_engine.utils.project_paths import processed_data_path


PROTOCOL_PATH = processed_data_path("benchmark_fixtures", "r2_bpi2012_hybrid_v1.json")
PROTOCOL_SHA256 = "82a5a7a411b1932b1b3de9523466938753ae9ae0a3b6fb70f4a1c1b7e26257a5"


def load_protocol() -> dict:
    raw = Path(PROTOCOL_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen BPI 2012 hybrid protocol changed")
    return json.loads(raw)


def _base_prediction(table: Counter, labels: tuple[str, ...]) -> tuple[str, dict[str, float]]:
    total = sum(table.values()) + len(labels)
    probabilities = {label: (table[label] + 1.0) / total for label in labels}
    predicted = max(labels, key=lambda label: (probabilities[label], -labels.index(label)))
    return predicted, probabilities


def _collect(protocol: dict, *, spiking: bool, constant_gap: bool = False, shuffled: bool = False) -> dict:
    traces = [trace for trace in load_traces() if split_name(trace) == "training"]
    labels = tuple(sorted({event.activity for trace in traces for event in trace.events}))
    first_end = int(len(traces) * protocol["chronological_case_folds"]["warmup_fraction"])
    second_end = first_end + int(len(traces) * protocol["chronological_case_folds"]["selection_fraction"])
    fold_by_case = {trace.case_id: "warmup" if index < first_end else "selection" if index < second_end else "confirmation"
                    for index, trace in enumerate(traces)}
    parent = json.loads(Path(processed_data_path("benchmark_fixtures", "r2_bpi2012_model_v1.json")).read_text())
    encoder = EventRouteEncoder(parent, spiking=spiking, constant_gap=constant_gap)
    settings = parent["learning"]
    learner = BoundedSparseMulticlassReadout(labels, SparseMulticlassConfig(
        settings["learning_rate"], settings["weight_cap"], settings["maximum_route_vocabulary"],
        settings["maximum_active_routes"], settings["maximum_class_count"]))
    tables = defaultdict(Counter)
    all_targets = [trace.events[index + 1].activity for trace in traces for index in range(len(trace.events) - 1)]
    feedback = list(all_targets)
    if shuffled:
        random.Random(916902).shuffle(feedback)
    feedback_index = 0
    records = {"selection": [], "confirmation": []}
    for trace in traces:
        fold = fold_by_case[trace.case_id]
        for index in range(len(trace.events) - 1):
            current = trace.events[index].activity
            previous = trace.events[index - 1].activity if index else "<BOS>"
            table = tables[(previous, current)]
            base_predicted, base_probabilities = _base_prediction(table, labels)
            receipt = learner.predict(encoder.encode(trace, index))
            true_target = trace.events[index + 1].activity
            if fold != "warmup":
                records[fold].append({"target": true_target, "base_predicted": base_predicted,
                    "base_probabilities": base_probabilities, "base_support": sum(table.values()),
                    "local_predicted": receipt.predicted, "local_scores": dict(receipt.scores)})
            learner.observe(receipt, feedback[feedback_index] if shuffled else true_target)
            feedback_index += 1
            table[true_target] += 1
    snapshot = learner.snapshot()
    return {"labels": labels, "records": records, "routes": len(encoder.routes),
            "neurons": len(encoder.units), "weight_entries": len(snapshot["weights"])}


def _evaluate(records: list[dict], labels: tuple[str, ...], config: ConfidenceRouteConfig | None) -> dict:
    rows = []
    overrides = 0
    for record in records:
        if config is None:
            predicted = record["base_predicted"]
            probabilities = record["base_probabilities"]
        else:
            decision = route_prediction(labels=labels, base_predicted=record["base_predicted"],
                base_probabilities=record["base_probabilities"], base_support=record["base_support"],
                local_predicted=record["local_predicted"], local_scores=record["local_scores"], config=config)
            predicted = decision.predicted
            probabilities = dict(decision.probabilities)
            overrides += int(decision.overridden)
        target = record["target"]
        rows.append({"target": target, "predicted": predicted,
            "brier": sum((probabilities[label] - int(label == target)) ** 2 for label in labels)})
    metrics = _metrics(rows, labels, set())
    metrics["overrides"] = overrides
    metrics["override_rate"] = overrides / len(rows)
    metrics["prediction_trace_sha256"] = hashlib.sha256(
        json.dumps([(row["target"], row["predicted"]) for row in rows], separators=(",", ":")).encode()).hexdigest()
    return metrics


def _configs(protocol: dict):
    router = protocol["router"]
    for cap in router["base_max_probability_caps"]:
        for margin in router["minimum_local_score_margins"]:
            for support in router["minimum_second_order_support"]:
                yield ConfidenceRouteConfig(cap, margin, support)


def run_diagnostic(protocol: dict) -> dict:
    arms = {
        "snn": _collect(protocol, spiking=True),
        "scalar": _collect(protocol, spiking=False),
        "constant_gap": _collect(protocol, spiking=True, constant_gap=True),
        "shuffled": _collect(protocol, spiking=True, shuffled=True),
    }
    labels = arms["snn"]["labels"]
    base_selection = _evaluate(arms["snn"]["records"]["selection"], labels, None)
    eligible = []
    candidates = []
    rule = protocol["selection_rule"]
    for config in _configs(protocol):
        metrics = _evaluate(arms["snn"]["records"]["selection"], labels, config)
        item = {"config": config, "metrics": metrics}
        candidates.append(item)
        if (metrics["top1_accuracy"] >= base_selection["top1_accuracy"] - rule["eligible_if_accuracy_loss_vs_base_at_most"]
                and metrics["multiclass_brier"] <= base_selection["multiclass_brier"] + rule["eligible_if_brier_increase_vs_base_at_most"]):
            eligible.append(item)
    chosen = min(eligible, key=lambda item: (-item["metrics"]["macro_f1"], -item["metrics"]["top1_accuracy"],
        item["metrics"]["multiclass_brier"], item["metrics"]["overrides"], item["config"])) if eligible else None
    config = chosen["config"] if chosen else None
    confirmation = {name: _evaluate(arm["records"]["confirmation"], labels, config) for name, arm in arms.items()}
    base_confirmation = _evaluate(arms["snn"]["records"]["confirmation"], labels, None)
    gate = protocol["confirmation_acceptance"]
    c = confirmation["snn"]
    checks = {
        "accuracy": c["top1_accuracy"] - base_confirmation["top1_accuracy"] >= gate["minimum_top1_gain_over_base"],
        "macro_f1": c["macro_f1"] - base_confirmation["macro_f1"] >= gate["minimum_macro_f1_gain_over_base"],
        "brier": c["multiclass_brier"] - base_confirmation["multiclass_brier"] <= gate["maximum_brier_increase_over_base"],
        "timing": c["macro_f1"] - confirmation["constant_gap"]["macro_f1"] >= gate["minimum_constant_gap_macro_f1_drop"],
        "shuffled": c["macro_f1"] - confirmation["shuffled"]["macro_f1"] >= gate["minimum_shuffled_outcome_macro_f1_drop"],
        "scalar_equivalence": c["prediction_trace_sha256"] == confirmation["scalar"]["prediction_trace_sha256"],
        "resources": all(arm["routes"] <= 4096 and arm["weight_entries"] <= 98304 for arm in arms.values()),
    }
    return {"schema": "sara-r2-bpi2012-confidence-hybrid-result-v1",
        "experiment_id": protocol["experiment_id"],
        "fold_case_counts": {"warmup": int(9160 * .70), "selection": int(9160 * .15),
                             "confirmation": 9160 - int(9160 * .70) - int(9160 * .15)},
        "selected_config": None if config is None else config.__dict__,
        "selection": {"base": base_selection, "hybrid": None if chosen is None else chosen["metrics"],
                      "eligible_candidates": len(eligible), "evaluated_candidates": len(candidates)},
        "confirmation": {"base": base_confirmation, **confirmation},
        "resources": {name: {key: arm[key] for key in ("routes", "neurons", "weight_entries")} for name, arm in arms.items()},
        "decision": {"checks": checks, "passed": all(checks.values()), "status": "diagnostic_pass" if all(checks.values()) else "diagnostic_negative_result",
                     "development_opened": False, "frozen_test_opened": False, "production_authorized": False}}


__all__ = ["load_protocol", "run_diagnostic"]
