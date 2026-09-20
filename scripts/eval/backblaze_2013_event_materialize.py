#!/usr/bin/env python3
"""Materialize frozen train/development drive events while keeping heldout SMART sealed."""
from __future__ import annotations

from collections import deque
import csv
from datetime import date
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

from sara_engine.utils.project_paths import processed_data_path, raw_data_path, workspace_path

SOURCE = Path(raw_data_path("r2_backblaze_2013", "data_2013.zip"))
MANIFEST = Path(processed_data_path("benchmark_fixtures", "backblaze_2013_case_manifest_v1.jsonl"))
PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "backblaze_2013_event_unit_protocol_v1.json"))
OUTPUT = Path(processed_data_path("benchmark_fixtures", "backblaze_2013_event_unit_development_v1.jsonl"))
REPORT = Path(workspace_path("evaluation", "backblaze_2013_event_materialize_v1.json"))
PROTOCOL_SHA256 = "a10f1d904d36f6d9771f3b837b1e8d76092eb15a82cd59c76358c8b653213fe0"
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
    if not value:
        return 0
    try:
        return max(0, int(float(value)))
    except ValueError:
        return 0


def main() -> int:
    if OUTPUT.exists() or REPORT.exists():
        raise ValueError("Backblaze event materialization already exists")
    if _digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Backblaze event-unit protocol digest changed")
    protocol = json.loads(PROTOCOL.read_text())
    if _digest(MANIFEST) != protocol["manifest_sha256"]:
        raise ValueError("Backblaze case manifest changed")
    selected = {}
    for line in MANIFEST.read_text().splitlines():
        row = json.loads(line)
        if row["split"] in protocol["allowed_splits"]:
            selected[row["case_id"]] = row
    histories = {case_id: deque(maxlen=protocol["history_rows"]) for case_id in selected}
    fields = tuple(protocol["smart_raw_fields"])
    latest_allowed = date.fromisoformat("2013-11-30")
    with zipfile.ZipFile(SOURCE) as archive:
        members = sorted(name for name in archive.namelist()
                         if len(Path(name).parts) == 2 and Path(name).parts[0] == "2013"
                         and name.endswith(".csv") and date.fromisoformat(Path(name).stem) <= latest_allowed)
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
                    if target is None:
                        continue
                    observed = row[date_index]
                    if observed > target["endpoint_date"]:
                        continue
                    histories[case_id].append((observed, tuple(_number(row[index]) for index in indices)))
            finally:
                text.close()
                raw.close()

    rows = []
    truncated_events = 0
    for case_id, target in selected.items():
        history = list(histories[case_id])
        events = []
        previous = (0,) * len(fields)
        for time_index, (_, values) in enumerate(history):
            for field_index, value in enumerate(values):
                prior = previous[field_index]
                if value > 0:
                    kind = 0 if prior == 0 else 1
                    events.append({"route": field_index * 3 + kind, "time": time_index,
                                   "branch": field_index % 2})
                if value > prior:
                    events.append({"route": field_index * 3 + 2, "time": time_index,
                                   "branch": field_index % 2})
            previous = values
        maximum = protocol["event_encoding"]["maximum_events_per_case"]
        if len(events) > maximum:
            truncated_events += len(events) - maximum
            events = events[-maximum:]
        rows.append({"case_id": case_id, "split": target["split"], "label": target["label"],
                     "history_rows": len(history), "events": events})
    rows.sort(key=lambda row: (row["split"], row["case_id"]))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    counts = {split: sum(row["split"] == split for row in rows)
              for split in protocol["allowed_splits"]}
    checks = {
        "all_selected_cases_materialized": len(rows) == len(selected),
        "history_complete": all(row["history_rows"] == protocol["history_rows"] for row in rows),
        "event_budget": all(len(row["events"]) <= protocol["resource_budgets"]["maximum_events_per_case"]
                            for row in rows),
        "heldout_absent": all(row["split"] != "heldout" for row in rows),
    }
    report = {"schema": "sara-backblaze-2013-event-materialize-result-v1",
              "protocol_sha256": PROTOCOL_SHA256, "dataset_sha256": _digest(OUTPUT),
              "cases": len(rows), "counts": counts, "truncated_events": truncated_events,
              "checks": checks, "passed": all(checks.values()),
              "heldout_smart_values_read": False, "heldout_feature_rows_materialized": False,
              "candidate_scored": False}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
