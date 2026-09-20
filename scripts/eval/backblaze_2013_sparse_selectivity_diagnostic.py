#!/usr/bin/env python3
"""Training-only sparse encoding diagnostic for the Backblaze task."""
from __future__ import annotations

from collections import Counter, deque
import csv
from datetime import date
from hashlib import sha256
import io
import json
import math
from pathlib import Path
from statistics import median
import zipfile

from sara_engine.evaluation.event_unit_causal_isolation import EventUnitV2Learner, UnitEpisode, UnitEvent
from sara_engine.utils.project_paths import processed_data_path, raw_data_path, workspace_path

SOURCE = Path(raw_data_path("r2_backblaze_2013", "data_2013.zip"))
MANIFEST = Path(processed_data_path("benchmark_fixtures", "backblaze_2013_case_manifest_v1.jsonl"))
PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "backblaze_2013_sparse_selectivity_protocol_v1.json"))
OUTPUT = Path(workspace_path("evaluation", "backblaze_2013_sparse_selectivity_diagnostic_v1.json"))
PROTOCOL_SHA256 = "b39992eec7ad5e7e2e5127680db90ad0d0acb065eb48c6d3a11c2842b924a15d"
CASE_PROTOCOL_ID = "backblaze-2013-case-manifest-v1-2026-09-20"


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_id(serial: str) -> str:
    return sha256(f"{CASE_PROTOCOL_ID}|case|{serial}".encode()).hexdigest()[:24]


def _number(value: str) -> int:
    try:
        return max(0, int(float(value))) if value else 0
    except ValueError:
        return 0


def _bucket(value: int) -> int:
    return min(15, int(math.log2(value + 1)))


def _encode(name: str, values: list[tuple[int, ...]], maximum: int) -> tuple[UnitEvent, ...]:
    events: list[UnitEvent] = []
    previous = (0,) * len(values[0])
    if name == "final_log_snapshot":
        for field, value in enumerate(values[-1]):
            events.append(UnitEvent(field * 16 + _bucket(value), len(values) - 1, field % 2))
        return tuple(events)
    for time_index, current in enumerate(values):
        for field, value in enumerate(current):
            prior = previous[field]
            if name == "v1_presence_onset_increase":
                if value > 0:
                    events.append(UnitEvent(field * 3 + (0 if prior == 0 else 1), time_index, field % 2))
                if value > prior:
                    events.append(UnitEvent(field * 3 + 2, time_index, field % 2))
            elif name == "transition_only":
                if (value == 0) != (prior == 0):
                    events.append(UnitEvent(field * 2 + int(value > 0), time_index, field % 2))
            elif name == "sparse_signed_log_delta" and value != prior:
                delta = value - prior
                route = field * 32 + (16 if delta < 0 else 0) + _bucket(abs(delta))
                events.append(UnitEvent(route, time_index, field % 2))
        previous = current
    return tuple(events[-maximum:])


def _fold(protocol_id: str, case_id: str, folds: int) -> int:
    return int(sha256(f"{protocol_id}|{case_id}".encode()).hexdigest(), 16) % folds


def _balanced(labels: list[int], predictions: list[int]) -> float:
    return sum(
        sum(predictions[index] == target for index, label in enumerate(labels) if label == target)
        / sum(label == target for label in labels)
        for target in (0, 1)
    ) / 2


def _evaluate(episodes: list[UnitEpisode], protocol_id: str, folds: int) -> dict[str, object]:
    labels: list[int] = []
    predictions: list[int] = []
    fold_scores = []
    for fold in range(folds):
        training = [row for row in episodes if _fold(protocol_id, row.identity, folds) != fold]
        evaluation = [row for row in episodes if _fold(protocol_id, row.identity, folds) == fold]
        training.sort(key=lambda row: sha256(
            f"{protocol_id}|{fold}|{row.identity}".encode()).hexdigest())
        learner = EventUnitV2Learner("B_compact_event")
        for row in training:
            receipt = learner.predict(row)
            learner.observe(receipt, row.label)
        fold_labels = []
        fold_predictions = []
        for row in evaluation:
            receipt = learner.predict(row)
            fold_labels.append(row.label)
            fold_predictions.append(receipt.predicted)
            learner.observe(receipt, receipt.predicted)
        labels.extend(fold_labels)
        predictions.extend(fold_predictions)
        fold_scores.append(_balanced(fold_labels, fold_predictions))
    return {"balanced_accuracy": _balanced(labels, predictions), "fold_balanced_accuracy": fold_scores,
            "prediction_trace_sha256": sha256(json.dumps(predictions).encode()).hexdigest()}


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("Backblaze sparse-selectivity attempt already exists")
    if _digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Backblaze sparse-selectivity protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    if _digest(MANIFEST) != protocol["case_manifest_sha256"]:
        raise ValueError("Backblaze case manifest changed")
    selected = {}
    for line in MANIFEST.read_text().splitlines():
        row = json.loads(line)
        if row["split"] == "training":
            selected[row["case_id"]] = row
    histories = {case_id: deque(maxlen=protocol["history_rows"]) for case_id in selected}
    latest = date.fromisoformat("2013-09-30")
    fields = tuple(protocol["smart_raw_fields"])
    with zipfile.ZipFile(SOURCE) as archive:
        members = sorted(name for name in archive.namelist()
                         if len(Path(name).parts) == 2 and Path(name).parts[0] == "2013"
                         and name.endswith(".csv") and date.fromisoformat(Path(name).stem) <= latest)
        for member in members:
            raw = archive.open(member)
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            try:
                reader = csv.reader(text)
                header = next(reader)
                serial_index = header.index("serial_number")
                date_index = header.index("date")
                indices = [header.index(field) for field in fields]
                for row in reader:
                    case_id = _case_id(row[serial_index])
                    target = selected.get(case_id)
                    if target is not None and row[date_index] <= target["endpoint_date"]:
                        histories[case_id].append(tuple(_number(row[index]) for index in indices))
            finally:
                text.close()
                raw.close()
    maximum = protocol["evaluation"]["maximum_events_per_case"]
    results = {}
    fold_counts = {str(fold): {"0": 0, "1": 0} for fold in range(protocol["evaluation"]["folds"])}
    for case_id, row in selected.items():
        fold_counts[str(_fold(protocol["protocol_id"], case_id, protocol["evaluation"]["folds"]))][str(row["label"])] += 1
    for name in protocol["encodings"]:
        episodes = []
        route_presence = Counter()
        event_counts = []
        signatures = set()
        for case_id, row in selected.items():
            events = _encode(name, list(histories[case_id]), maximum)
            episodes.append(UnitEpisode(case_id, name, events, row["label"]))
            routes = tuple(sorted({event.route for event in events}))
            signatures.add(routes)
            route_presence.update(routes)
            event_counts.append(len(events))
        evaluation = _evaluate(episodes, protocol["protocol_id"], protocol["evaluation"]["folds"])
        evaluation.update({
            "median_events": median(event_counts),
            "event_cap_fraction": sum(count == maximum for count in event_counts) / len(event_counts),
            "unique_route_signatures": len(signatures),
            "dominant_route_fraction": max(route_presence.values(), default=0) / len(episodes),
        })
        results[name] = evaluation
    thresholds = protocol["diagnostic_thresholds"]
    best_name = max(results, key=lambda name: results[name]["balanced_accuracy"])
    best = results[best_name]
    v1 = results["v1_presence_onset_increase"]
    checks = {
        "fold_support": all(value[label] >= thresholds["minimum_cases_per_class_per_fold"]
                            for value in fold_counts.values() for label in ("0", "1")),
        "history_complete": all(len(history) == protocol["history_rows"] for history in histories.values()),
        "minimum_quality": best["balanced_accuracy"] >= thresholds["minimum_balanced_accuracy"],
        "gain_over_v1": best["balanced_accuracy"] - v1["balanced_accuracy"] >= thresholds["minimum_gain_over_v1"],
        "route_selectivity": best["dominant_route_fraction"] <= thresholds["maximum_dominant_route_fraction"],
        "event_sparsity": best["event_cap_fraction"] <= thresholds["maximum_event_cap_fraction"],
    }
    report = {
        "schema": "sara-backblaze-2013-sparse-selectivity-diagnostic-result-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "training_cases": len(selected),
        "fold_counts": fold_counts,
        "encodings": results,
        "best_encoding": best_name,
        "checks": checks,
        "diagnostic_passed": all(checks.values()),
        "development_consumed": False,
        "heldout_consumed": False,
        "candidate_promotion_authorized": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["diagnostic_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
