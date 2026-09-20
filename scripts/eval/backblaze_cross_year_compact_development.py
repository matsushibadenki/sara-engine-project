#!/usr/bin/env python3
"""Evaluate the frozen Compact Event hypothesis across independent drive years."""
from __future__ import annotations

from collections import Counter, deque
import csv
from datetime import date
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import random
from statistics import median
import zipfile

from sara_engine.evaluation.event_unit_causal_isolation import EventUnitV2Learner, UnitEpisode, UnitEvent
from sara_engine.utils.project_paths import processed_data_path, raw_data_path, workspace_path

PROTOCOL = Path(processed_data_path("benchmark_fixtures", "backblaze_cross_year_compact_protocol_v1.json"))
TRAIN_MANIFEST = Path(processed_data_path("benchmark_fixtures", "backblaze_2013_case_manifest_v1.jsonl"))
DEV_MANIFEST = Path(processed_data_path("benchmark_fixtures", "backblaze_2014_case_manifest_v1.jsonl"))
SOURCE_2013 = Path(raw_data_path("r2_backblaze_2013", "data_2013.zip"))
SOURCE_2014 = Path(raw_data_path("r2_backblaze_2014", "data_2014.zip"))
OUTPUT = Path(workspace_path("evaluation", "backblaze_cross_year_compact_development_v1.json"))
PROTOCOL_SHA256 = "d72e7ba2b37dc8adfb5b28f8ac9486acd1d9d7b82cb4d7c2b4ddb09e970afda2"
CASE_PROTOCOLS = {
    "2013": "backblaze-2013-case-manifest-v1-2026-09-20",
    "2014": "backblaze-2014-independent-case-manifest-v1-2026-09-20",
}


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_id(year: str, serial: str) -> str:
    return sha256(f"{CASE_PROTOCOLS[year]}|case|{serial}".encode()).hexdigest()[:24]


def _number(value: str) -> int:
    try:
        return max(0, int(float(value))) if value else 0
    except ValueError:
        return 0


def _load_selected(path: Path, split: str) -> dict[str, dict]:
    return {row["case_id"]: row for row in map(json.loads, path.read_text().splitlines())
            if row["split"] == split}


def _histories(source: Path, year: str, selected: dict[str, dict], fields: tuple[str, ...],
               latest: date, width: int) -> dict[str, list[tuple[int, ...]]]:
    histories = {case_id: deque(maxlen=width) for case_id in selected}
    with zipfile.ZipFile(source) as archive:
        members = sorted(name for name in archive.namelist()
                         if len(Path(name).parts) == 2 and Path(name).parts[0] == year
                         and name.endswith(".csv") and date.fromisoformat(Path(name).stem) <= latest)
        for member in members:
            with archive.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
                reader = csv.reader(text)
                header = next(reader)
                serial_index = header.index("serial_number")
                date_index = header.index("date")
                indices = [header.index(field) for field in fields]
                for row in reader:
                    case_id = _case_id(year, row[serial_index])
                    target = selected.get(case_id)
                    if target is not None and row[date_index] <= target["endpoint_date"]:
                        histories[case_id].append(tuple(_number(row[index]) for index in indices))
    return {key: list(value) for key, value in histories.items()}


def _bucket(value: int) -> int:
    return min(15, int(math.log2(value + 1)))


def _encode(values: list[tuple[int, ...]], kind: str, maximum: int,
            suppressed: set[int] | None = None) -> tuple[UnitEvent, ...]:
    suppressed = suppressed or set()
    if kind == "final_log_snapshot":
        return tuple(UnitEvent(field * 16 + _bucket(value), 29, field % 2)
                     for field, value in enumerate(values[-1]))
    events = []
    previous = (0,) * len(values[0])
    for time_index, current in enumerate(values):
        for field, value in enumerate(current):
            prior = previous[field]
            if kind == "v1_presence_onset_increase":
                if value > 0:
                    events.append(UnitEvent(field * 3 + (0 if prior == 0 else 1), time_index, field % 2))
                if value > prior:
                    events.append(UnitEvent(field * 3 + 2, time_index, field % 2))
            elif value != prior:
                delta = value - prior
                route = field * 32 + (16 if delta < 0 else 0) + _bucket(abs(delta))
                if route not in suppressed:
                    events.append(UnitEvent(route, time_index, field % 2))
        previous = current
    return tuple(events[-maximum:])


def _episodes(selected: dict[str, dict], histories: dict[str, list[tuple[int, ...]]],
              kind: str, maximum: int, suppressed: set[int] | None = None) -> list[UnitEpisode]:
    rows = [UnitEpisode(case_id, kind, _encode(histories[case_id], kind, maximum, suppressed), row["label"])
            for case_id, row in selected.items()]
    rows.sort(key=lambda row: sha256(row.identity.encode()).hexdigest())
    return rows


def _balanced(labels: list[int], predictions: list[int]) -> float:
    return sum(sum(predictions[i] == target for i, label in enumerate(labels) if label == target)
               / sum(label == target for label in labels) for target in (0, 1)) / 2


def _run(training: list[UnitEpisode], development: list[UnitEpisode], shuffle: bool = False) -> dict:
    learner = EventUnitV2Learner("B_compact_event")
    outcomes = [row.label for row in training]
    if shuffle:
        random.Random(20_140_920).shuffle(outcomes)
    for row, target in zip(training, outcomes):
        receipt = learner.predict(row)
        learner.observe(receipt, target)
    labels, predictions = [], []
    for row in development:
        receipt = learner.predict(row)
        labels.append(row.label)
        predictions.append(receipt.predicted)
        learner.observe(receipt, receipt.predicted)
    routes = Counter()
    for row in development:
        routes.update({event.route for event in row.events})
    return {"balanced_accuracy": _balanced(labels, predictions),
            "predicted_positive": sum(predictions),
            "median_events": median(len(row.events) for row in development),
            "event_cap_fraction": sum(len(row.events) == 96 for row in development) / len(development),
            "dominant_route_fraction": max(routes.values(), default=0) / len(development),
            "state_bytes": learner.state_bytes(),
            "trace_sha256": sha256(json.dumps(predictions).encode()).hexdigest()}


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("Cross-year Compact Event attempt already exists")
    if _digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Cross-year protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    if _digest(TRAIN_MANIFEST) != protocol["training_manifest_sha256"] or _digest(DEV_MANIFEST) != protocol["development_manifest_sha256"]:
        raise ValueError("Cross-year manifest changed")
    training_selected = _load_selected(TRAIN_MANIFEST, "training")
    development_selected = _load_selected(DEV_MANIFEST, "development")
    fields = tuple(protocol["smart_raw_fields"])
    width = protocol["history_rows"]
    training_histories = _histories(SOURCE_2013, "2013", training_selected, fields, date(2013, 8, 31), width)
    development_histories = _histories(SOURCE_2014, "2014", development_selected, fields, date(2014, 10, 31), width)
    maximum = protocol["candidate"]["maximum_events_per_case"]
    raw_training = _episodes(training_selected, training_histories, "signed_delta", maximum)
    prevalence = Counter()
    for row in raw_training:
        prevalence.update({event.route for event in row.events})
    suppressed = {route for route, count in prevalence.items()
                  if count / len(raw_training) > 0.95}
    arms = {}
    for name, kind, use_suppression in (
        ("candidate", "signed_delta", True),
        ("candidate_unsuppressed", "signed_delta", False),
        ("v1", "v1_presence_onset_increase", False),
        ("final_log_snapshot", "final_log_snapshot", False),
    ):
        removed = suppressed if use_suppression else set()
        arms[name] = _run(
            _episodes(training_selected, training_histories, kind, maximum, removed),
            _episodes(development_selected, development_histories, kind, maximum, removed))
    outcome_shuffle = _run(
        _episodes(training_selected, training_histories, "signed_delta", maximum, suppressed),
        _episodes(development_selected, development_histories, "signed_delta", maximum, suppressed), True)
    acceptance = protocol["acceptance"]
    candidate = arms["candidate"]
    checks = {
        "case_support": all(sum(row["label"] == label for row in development_selected.values())
                            == acceptance["development_cases_per_class"] for label in (0, 1)),
        "history_complete": all(len(value) == width for value in training_histories.values())
                            and all(len(value) == width for value in development_histories.values()),
        "minimum_quality": candidate["balanced_accuracy"] >= acceptance["minimum_balanced_accuracy"],
        "gain_over_final_snapshot": candidate["balanced_accuracy"] - arms["final_log_snapshot"]["balanced_accuracy"]
                                    >= acceptance["minimum_gain_over_final_snapshot"],
        "outcome_shuffle_drop": candidate["balanced_accuracy"] - outcome_shuffle["balanced_accuracy"]
                                >= acceptance["minimum_outcome_shuffle_drop"],
        "route_selectivity": candidate["dominant_route_fraction"] <= acceptance["maximum_dominant_route_fraction"],
        "event_sparsity": candidate["event_cap_fraction"] <= acceptance["maximum_event_cap_fraction"],
    }
    report = {"schema": "sara-backblaze-cross-year-compact-development-result-v1",
              "protocol_sha256": PROTOCOL_SHA256, "training_cases": len(training_selected),
              "development_cases": len(development_selected), "suppressed_routes": sorted(suppressed),
              "arms": arms, "outcome_shuffle": outcome_shuffle, "checks": checks,
              "development_gate_passed": all(checks.values()), "heldout_consumed": False,
              "temporal_state_used": False, "spiking_used": False, "global_backpropagation": False}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["development_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
