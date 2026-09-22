#!/usr/bin/env python3
"""Check whether Helpdesk closure timestamps are mechanically delayed resolution times."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

from sara_engine.utils.project_paths import processed_data_path, raw_data_path, workspace_path

SOURCE = Path(raw_data_path("r2_helpdesk", "helpdesk_anonymized.csv"))
FEASIBILITY = Path(workspace_path("evaluation", "helpdesk_prospective_feasibility_v1.json"))
PROTOCOL = Path(processed_data_path("benchmark_fixtures", "helpdesk_closure_timestamp_audit_protocol_v1.json"))
OUTPUT = Path(workspace_path("evaluation", "helpdesk_closure_timestamp_audit_v1.json"))
PROTOCOL_SHA256 = "1b0e0b90434e98569876a87268d925f308fb71bcc692afe33372c072ad8cf173"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("Closure timestamp audit already exists")
    if digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Closure audit protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    if digest(SOURCE) != protocol["source_sha256"] or digest(FEASIBILITY) != protocol["feasibility_result_sha256"]:
        raise ValueError("Pinned audit input changed")
    events: dict[str, dict[str, list[datetime]]] = defaultdict(lambda: defaultdict(list))
    with SOURCE.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        indices = {name: header.index(name) for name in protocol["allowed_columns"]}
        for row in reader:
            activity = row[indices["Activity"]]
            if activity not in {"Closed", "Resolve ticket"}:
                continue
            case_id = row[indices["Case ID"]]
            timestamp = datetime.strptime(row[indices["Complete Timestamp"]], "%Y/%m/%d %H:%M:%S.%f")
            events[case_id][activity].append(timestamp)
    eligible = 0
    exact = 0
    missing_or_multiple = 0
    lag_days = Counter()
    for case in events.values():
        closed = case["Closed"]
        if len(closed) != 1:
            missing_or_multiple += 1
            continue
        preceding = [time for time in case["Resolve ticket"] if time < closed[0]]
        if not preceding:
            missing_or_multiple += 1
            continue
        lag_seconds = (closed[0] - max(preceding)).total_seconds()
        eligible += 1
        exact += abs(lag_seconds - protocol["synthetic_closure_indicator"]["exact_lag_days"] * 86400) <= (
            protocol["synthetic_closure_indicator"]["tolerance_seconds"])
        lag_days[f"{round(lag_seconds / 86400, 3):.3f}"] += 1
    fraction = exact / eligible if eligible else 0.0
    synthetic = fraction >= protocol["synthetic_closure_indicator"]["minimum_fraction"]
    result = {"schema": "sara-helpdesk-closure-timestamp-audit-result-v1",
              "protocol_sha256": PROTOCOL_SHA256, "source_sha256": protocol["source_sha256"],
              "eligible_cases": eligible, "exact_15_day_cases": exact,
              "exact_15_day_fraction": fraction, "missing_or_multiple_endpoint_cases": missing_or_multiple,
              "top_lag_days": lag_days.most_common(12),
              "synthetic_closure_indicator": synthetic,
              "decision": "stop_closed_timestamp_task" if synthetic else "continue_source_review_only",
              "descriptive_columns_read": False, "candidate_scored": False,
              "heldout_features_materialized": False}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
