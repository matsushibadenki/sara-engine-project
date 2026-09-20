#!/usr/bin/env python3
"""Describe frozen Backblaze event coverage after the failed development gate."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from sara_engine.utils.project_paths import processed_data_path, workspace_path

DATASET = Path(processed_data_path(
    "benchmark_fixtures", "backblaze_2013_event_unit_development_v1.jsonl"))
OUTPUT = Path(workspace_path("evaluation", "backblaze_2013_event_coverage_diagnostic_v1.json"))
DATASET_SHA256 = "3cd3605049965a402976eac83fe54f55766e1434f0c2a3d28621e4b5d2659dd9"


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("Backblaze coverage diagnostic already exists")
    if sha256(DATASET.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("Backblaze event dataset changed")
    rows = [json.loads(line) for line in DATASET.read_text().splitlines()]
    groups = {}
    for split in ("training", "development"):
        for label in (0, 1):
            selected = [row for row in rows if row["split"] == split and row["label"] == label]
            counts = [len(row["events"]) for row in selected]
            route_presence = Counter()
            for row in selected:
                route_presence.update({event["route"] for event in row["events"]})
            groups[f"{split}_label_{label}"] = {
                "cases": len(selected),
                "empty_cases": sum(count == 0 for count in counts),
                "median_events": median(counts),
                "maximum_events": max(counts),
                "route_presence": {str(route): route_presence[route] for route in sorted(route_presence)},
            }
    result = {
        "schema": "sara-backblaze-2013-event-coverage-diagnostic-v1",
        "dataset_sha256": DATASET_SHA256,
        "groups": groups,
        "candidate_rerun": False,
        "threshold_changed": False,
        "heldout_consumed": False,
        "interpretation_boundary": (
            "Descriptive post-hoc coverage only; it cannot rescue or reinterpret the failed gate."),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
