#!/usr/bin/env python3
"""Audit Helpdesk case/time/activity metadata before defining a prediction task."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

from sara_engine.utils.project_paths import processed_data_path, raw_data_path, workspace_path

SOURCE = Path(raw_data_path("r2_helpdesk", "helpdesk_anonymized.csv"))
PROTOCOL = Path(processed_data_path("benchmark_fixtures", "helpdesk_source_audit_protocol_v1.json"))
OUTPUT = Path(workspace_path("evaluation", "helpdesk_source_audit_v1.json"))
PROTOCOL_SHA256 = "ece79c05a10e77e04c108aa699c7ab80822a985ae9b0f38855f1dc13357bc260"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("Helpdesk source audit already exists")
    if digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Helpdesk audit protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    if digest(SOURCE) != protocol["source_sha256"]:
        raise ValueError("Helpdesk source changed")
    histories: dict[str, list[tuple[datetime, int, str]]] = defaultdict(list)
    activities = Counter()
    event_count = 0
    parse_failures = 0
    with SOURCE.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        indices = {name: header.index(name) for name in protocol["allowed_audit_columns"]}
        for row in reader:
            event_count += 1
            # No non-audit column value is accessed.
            case_id = row[indices["Case ID"]]
            activity = row[indices["Activity"]]
            try:
                timestamp = datetime.strptime(row[indices["Complete Timestamp"]], "%Y/%m/%d %H:%M:%S.%f")
            except ValueError:
                parse_failures += 1
                continue
            histories[case_id].append((timestamp, event_count, activity))
            activities[activity] += 1
    first_months = Counter()
    last_months = Counter()
    terminal = Counter()
    two_times = 0
    same_time_events = 0
    for history in histories.values():
        history.sort()
        first_months[history[0][0].strftime("%Y-%m")] += 1
        last_months[history[-1][0].strftime("%Y-%m")] += 1
        terminal[history[-1][2]] += 1
        unique_times = {event[0] for event in history}
        two_times += len(unique_times) >= 2
        same_time_events += len(history) - len(unique_times)
    minimums = protocol["required_minimums"]
    checks = {
        "independent_cases": len(histories) >= minimums["independent_cases"],
        "timestamped_events": event_count - parse_failures >= minimums["timestamped_events"],
        "two_distinct_timestamps": two_times >= minimums["cases_with_two_distinct_timestamps"],
        "opening_months": len(first_months) >= minimums["opening_months"],
        "all_timestamps_parse": parse_failures == 0,
    }
    result = {
        "schema": "sara-helpdesk-source-audit-result-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "source_sha256": protocol["source_sha256"],
        "event_count": event_count,
        "case_count": len(histories),
        "activity_counts": dict(sorted(activities.items())),
        "first_month_counts": dict(sorted(first_months.items())),
        "last_month_counts": dict(sorted(last_months.items())),
        "cases_with_two_distinct_timestamps": two_times,
        "terminal_activity_counts": dict(sorted(terminal.items())),
        "timestamp_parse_failures": parse_failures,
        "same_timestamp_event_count": same_time_events,
        "checks": checks,
        "passed": all(checks.values()),
        "other_columns_read": False,
        "candidate_scored": False,
        "descriptive_features_materialized": False,
        "heldout_selected": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
