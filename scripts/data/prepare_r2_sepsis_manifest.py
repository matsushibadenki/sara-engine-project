#!/usr/bin/env python3
"""Build the Sepsis case manifest and development-only baselines."""

from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path
from sara_engine.evaluation.bpi2012_data import gap_bucket
from sara_engine.evaluation.sepsis_data import SOURCE_SHA256, load_traces, split_name
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path

def _probabilities(counts, labels):
    total = sum(counts.values()) + len(labels)
    return {label: (counts[label] + 1.0) / total for label in labels}

def _metrics(rows, labels):
    support = Counter(row[0] for row in rows); predicted = Counter(row[1] for row in rows)
    tp = Counter(row[0] for row in rows if row[0] == row[1]); f1 = []
    for label in labels:
        precision = tp[label] / predicted[label] if predicted[label] else 0.0
        recall = tp[label] / support[label] if support[label] else 0.0
        f1.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return {"count": len(rows), "top1_accuracy": sum(target == prediction for target, prediction, _ in rows) / len(rows),
        "macro_f1": sum(f1) / len(f1),
        "multiclass_brier": sum(sum((probs[label] - int(label == target)) ** 2 for label in labels) for target, _, probs in rows) / len(rows)}

def build():
    traces = load_traces(); labels = tuple(sorted({event.activity for trace in traces if split_name(trace) != "frozen_test" for event in trace.events}))
    names = ("majority", "first_order", "second_order", "timing_aware", "constant_gap")
    global_counts = Counter(); first = defaultdict(Counter); second = defaultdict(Counter); timing = defaultdict(Counter); constant = defaultdict(Counter)
    records = {name: [] for name in names}
    for trace in traces:
        split = split_name(trace)
        if split == "frozen_test": continue
        for index in range(len(trace.events) - 1):
            event = trace.events[index]; target = trace.events[index + 1].activity
            previous = trace.events[index - 1].activity if index else "<BOS>"
            gap = 0.0 if index == 0 else (event.timestamp - trace.events[index - 1].timestamp).total_seconds()
            keys = {"majority": None, "first_order": event.activity, "second_order": (previous, event.activity),
                "timing_aware": (event.activity, event.lifecycle, gap_bucket(gap)),
                "constant_gap": (event.activity, event.lifecycle, 1 if gap > 0 else 0)}
            tables = {"majority": global_counts, "first_order": first, "second_order": second, "timing_aware": timing, "constant_gap": constant}
            if split == "development":
                for name in names:
                    counts = tables[name] if keys[name] is None else tables[name][keys[name]]
                    probabilities = _probabilities(counts, labels)
                    prediction = max(labels, key=lambda label: (probabilities[label], -labels.index(label)))
                    records[name].append((target, prediction, probabilities))
            global_counts[target] += 1; first[event.activity][target] += 1; second[(previous, event.activity)][target] += 1
            timing[(event.activity, event.lifecycle, gap_bucket(gap))][target] += 1
            constant[(event.activity, event.lifecycle, 1 if gap > 0 else 0)][target] += 1
    split_counts = Counter(split_name(trace) for trace in traces)
    prediction_counts = {name: sum(len(trace.events) - 1 for trace in traces if split_name(trace) == name) for name in split_counts}
    manifest = {"schema": "sara-r2-sepsis-manifest-v1", "source_sha256": SOURCE_SHA256, "labels": list(labels),
        "case_counts": dict(split_counts), "prediction_counts": prediction_counts, "case_overlap": 0,
        "absolute_time_features_forbidden": True, "frozen_test_metrics_computed": False,
        "table_states": {"first_order": len(first), "second_order": len(second), "timing_aware": len(timing)}}
    metrics = {name: _metrics(rows, labels) for name, rows in records.items()}
    baselines = {"schema": "sara-r2-sepsis-development-baselines-v1", "source_sha256": SOURCE_SHA256,
        "online_updates": True, "arms": metrics,
        "timing_aware_minus_constant_gap_accuracy": metrics["timing_aware"]["top1_accuracy"] - metrics["constant_gap"]["top1_accuracy"],
        "frozen_test_metrics_computed": False}
    return manifest, baselines

def main():
    manifest, baselines = build()
    mp = Path(ensure_parent_directory(processed_data_path("r2_sepsis", "source_manifest_v1.json")))
    bp = Path(ensure_parent_directory(workspace_path("evaluation", "r2_sepsis_development_baselines_v1.json")))
    mp.write_text(json.dumps(manifest, indent=2) + "\n"); bp.write_text(json.dumps(baselines, indent=2) + "\n")
    print(json.dumps({"manifest": str(mp), "baselines": str(bp), "cases": manifest["case_counts"], "predictions": manifest["prediction_counts"]}))
    return 0

if __name__ == "__main__": raise SystemExit(main())
