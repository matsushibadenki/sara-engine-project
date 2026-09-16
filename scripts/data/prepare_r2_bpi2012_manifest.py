#!/usr/bin/env python3
"""Build the BPI 2012 case manifest and train/development baselines."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, raw_data_path, workspace_path


SOURCE_SHA256 = "5cd9cc16b9bcb20bd4aae45666a5d87479ddbf47e6371618b6ad217174cecdf3"
DEV_BOUNDARY = datetime.fromisoformat("2012-01-19T10:02:23.184000+01:00")
TEST_BOUNDARY = datetime.fromisoformat("2012-02-11T10:46:32.413000+01:00")
GAP_BOUNDS_SECONDS = (0.0, 60.0, 300.0, 1800.0, 7200.0, 28800.0, 86400.0, 259200.0)


@dataclass(frozen=True)
class Event:
    activity: str
    lifecycle: str
    timestamp: datetime


@dataclass(frozen=True)
class Trace:
    case_id: str
    events: tuple[Event, ...]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _attributes(element: ET.Element) -> dict[str, str]:
    result = {}
    for child in element:
        if _local(child.tag) in ("string", "date", "int", "float", "boolean") and "key" in child.attrib:
            result[child.attrib["key"]] = child.attrib.get("value", "")
    return result


def load_traces() -> list[Trace]:
    source = Path(raw_data_path("r2_bpi2012", "BPI_Challenge_2012.xes.gz"))
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("BPI 2012 source SHA-256 mismatch")
    traces = []
    with gzip.open(source, "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if _local(element.tag) != "trace":
                continue
            trace_attrs = _attributes(element)
            events = []
            for child in element:
                if _local(child.tag) != "event":
                    continue
                attrs = _attributes(child)
                events.append(Event(attrs["concept:name"], attrs.get("lifecycle:transition", "missing"),
                                    datetime.fromisoformat(attrs["time:timestamp"])))
            traces.append(Trace(trace_attrs["concept:name"], tuple(events)))
            element.clear()
    traces.sort(key=lambda trace: (trace.events[0].timestamp, trace.case_id))
    return traces


def split_name(trace: Trace) -> str:
    start = trace.events[0].timestamp
    if start < DEV_BOUNDARY:
        return "training"
    if start < TEST_BOUNDARY:
        return "development"
    return "frozen_test"


def gap_bucket(seconds: float) -> int:
    for index, upper in enumerate(GAP_BOUNDS_SECONDS):
        if seconds <= upper:
            return index
    return len(GAP_BOUNDS_SECONDS)


def _probabilities(counts: Counter, labels: tuple[str, ...]) -> dict[str, float]:
    total = sum(counts.values()) + len(labels)
    return {label: (counts[label] + 1.0) / total for label in labels}


def _prediction(counts: Counter, labels: tuple[str, ...]) -> tuple[str, dict[str, float]]:
    probabilities = _probabilities(counts, labels)
    return max(labels, key=lambda label: (probabilities[label], -labels.index(label))), probabilities


def _metrics(rows: list[tuple[str, str, dict[str, float]]], labels: tuple[str, ...]) -> dict:
    support = Counter(target for target, _, _ in rows)
    true_positive = Counter(target for target, predicted, _ in rows if target == predicted)
    predicted_count = Counter(predicted for _, predicted, _ in rows)
    f1 = []
    recall = {}
    for label in labels:
        tp = true_positive[label]
        precision = tp / predicted_count[label] if predicted_count[label] else 0.0
        label_recall = tp / support[label] if support[label] else 0.0
        recall[label] = label_recall
        f1.append(0.0 if precision + label_recall == 0 else 2 * precision * label_recall / (precision + label_recall))
    return {
        "count": len(rows),
        "top1_accuracy": sum(target == predicted for target, predicted, _ in rows) / len(rows),
        "macro_f1": sum(f1) / len(f1),
        "multiclass_brier": sum(sum((probabilities[label] - int(label == target)) ** 2 for label in labels)
                                  for target, _, probabilities in rows) / len(rows),
        "per_activity_recall": recall,
    }


def build() -> tuple[dict, dict]:
    traces = load_traces()
    labels = tuple(sorted({event.activity for trace in traces if split_name(trace) != "frozen_test" for event in trace.events}))
    train_traces = [trace for trace in traces if split_name(trace) == "training"]
    dev_traces = [trace for trace in traces if split_name(trace) == "development"]
    test_traces = [trace for trace in traces if split_name(trace) == "frozen_test"]
    global_counts = Counter()
    first_counts = defaultdict(Counter)
    second_counts = defaultdict(Counter)
    timing_counts = defaultdict(Counter)
    constant_gap_counts = defaultdict(Counter)

    def examples(trace: Trace):
        for index in range(len(trace.events) - 1):
            current = trace.events[index]
            target = trace.events[index + 1].activity
            previous_activity = trace.events[index - 1].activity if index else "<BOS>"
            gap = 0.0 if index == 0 else (current.timestamp - trace.events[index - 1].timestamp).total_seconds()
            yield current, previous_activity, gap, target

    for trace in train_traces:
        for current, previous, gap, target in examples(trace):
            global_counts[target] += 1
            first_counts[current.activity][target] += 1
            second_counts[(previous, current.activity)][target] += 1
            timing_counts[(current.activity, current.lifecycle, gap_bucket(gap))][target] += 1
            constant_gap_counts[(current.activity, current.lifecycle, 1)][target] += 1

    online_global = global_counts.copy()
    online_first = defaultdict(Counter, {key: value.copy() for key, value in first_counts.items()})
    online_second = defaultdict(Counter, {key: value.copy() for key, value in second_counts.items()})
    online_timing = defaultdict(Counter, {key: value.copy() for key, value in timing_counts.items()})
    online_constant = defaultdict(Counter, {key: value.copy() for key, value in constant_gap_counts.items()})
    records = {name: [] for name in ("majority", "first_order", "second_order", "timing_aware", "constant_gap")}
    for trace in dev_traces:
        for current, previous, gap, target in examples(trace):
            keys = {
                "majority": None,
                "first_order": current.activity,
                "second_order": (previous, current.activity),
                "timing_aware": (current.activity, current.lifecycle, gap_bucket(gap)),
                "constant_gap": (current.activity, current.lifecycle, 1),
            }
            tables = {"majority": online_global, "first_order": online_first, "second_order": online_second,
                      "timing_aware": online_timing, "constant_gap": online_constant}
            for name, key in keys.items():
                counts = tables[name] if key is None else tables[name][key]
                predicted, probabilities = _prediction(counts, labels)
                records[name].append((target, predicted, probabilities))
            online_global[target] += 1
            online_first[current.activity][target] += 1
            online_second[(previous, current.activity)][target] += 1
            online_timing[(current.activity, current.lifecycle, gap_bucket(gap))][target] += 1
            online_constant[(current.activity, current.lifecycle, 1)][target] += 1

    manifest = {
        "schema": "sara-r2-bpi2012-processed-manifest-v1",
        "source_sha256": SOURCE_SHA256,
        "labels": list(labels),
        "case_counts": {"training": len(train_traces), "development": len(dev_traces), "frozen_test": len(test_traces)},
        "prediction_counts": {
            "training": sum(len(trace.events) - 1 for trace in train_traces),
            "development": sum(len(trace.events) - 1 for trace in dev_traces),
            "frozen_test": sum(len(trace.events) - 1 for trace in test_traces),
        },
        "gap_bounds_seconds": list(GAP_BOUNDS_SECONDS),
        "case_overlap": 0,
        "frozen_test_metrics_computed": False,
        "table_states": {"first_order": len(first_counts), "second_order": len(second_counts), "timing_aware": len(timing_counts)},
    }
    baselines = {
        "schema": "sara-r2-bpi2012-development-baselines-v1",
        "source_sha256": SOURCE_SHA256,
        "online_updates": True,
        "arms": {name: _metrics(rows, labels) for name, rows in records.items()},
        "timing_aware_minus_constant_gap_accuracy": _metrics(records["timing_aware"], labels)["top1_accuracy"] - _metrics(records["constant_gap"], labels)["top1_accuracy"],
        "frozen_test_metrics_computed": False,
    }
    return manifest, baselines


def main() -> int:
    manifest, baselines = build()
    manifest_path = Path(ensure_parent_directory(processed_data_path("r2_bpi2012", "source_manifest_v1.json")))
    baseline_path = Path(ensure_parent_directory(workspace_path("evaluation", "r2_bpi2012_development_baselines_v1.json")))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    baseline_path.write_text(json.dumps(baselines, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "baselines": str(baseline_path),
                      "cases": manifest["case_counts"], "predictions": manifest["prediction_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
