#!/usr/bin/env python3
"""Audit the hash-pinned MetroPT-3 source without training or scoring a model."""
from __future__ import annotations

import csv
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import zipfile

from sara_engine.utils.project_paths import raw_data_path, workspace_path

SOURCE = Path(raw_data_path("r2_metropt3", "metropt3.zip"))
SOURCE_SHA256 = "aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a"
MEMBER = "MetroPT3(AirCompressor).csv"
OUTPUT = Path(workspace_path("evaluation", "metropt3_source_audit_v2.json"))
FAILURES = (
    ("2020-04-18 00:00:00", "2020-04-18 23:59:59"),
    ("2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    ("2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00"),
)


def _time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("MetroPT-3 source audit already exists")
    if sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("MetroPT-3 source digest changed")
    failure_windows = tuple((_time(start), _time(end)) for start, end in FAILURES)
    rows = 0
    first = last = previous = None
    negative_gaps = duplicate_timestamps = 0
    failure_counts = [0] * len(failure_windows)
    gaps = []
    with zipfile.ZipFile(SOURCE) as archive, archive.open(MEMBER) as raw:
        lines = (line.decode("utf-8-sig") for line in raw)
        reader = csv.DictReader(lines)
        fields = tuple(reader.fieldnames or ())
        for row in reader:
            timestamp = _time(row["timestamp"])
            rows += 1
            first = timestamp if first is None else first
            last = timestamp
            if previous is not None:
                gap = (timestamp - previous).total_seconds()
                negative_gaps += gap < 0
                duplicate_timestamps += gap == 0
                if gap >= 0:
                    gaps.append(gap)
            previous = timestamp
            for index, (start, end) in enumerate(failure_windows):
                failure_counts[index] += start <= timestamp <= end
    expected_fields = ("", "timestamp", "TP2", "TP3", "H1", "DV_pressure",
                       "Reservoirs", "Oil_temperature", "Motor_current", "COMP",
                       "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch",
                       "Oil_level", "Caudal_impulses")
    checks = {
        "row_count": rows == 1516948,
        "field_contract": fields == expected_fields,
        "chronological": negative_gaps == 0,
        "four_failure_windows_have_rows": all(count > 0 for count in failure_counts),
    }
    result = {
        "schema": "sara-metropt3-source-audit-v2",
        "source_sha256": SOURCE_SHA256,
        "member": MEMBER,
        "rows": rows,
        "fields": fields,
        "first_timestamp": first.isoformat() if first else None,
        "last_timestamp": last.isoformat() if last else None,
        "median_nonnegative_gap_seconds": median(gaps),
        "maximum_nonnegative_gap_seconds": max(gaps),
        "negative_gaps": negative_gaps,
        "duplicate_timestamps": duplicate_timestamps,
        "failure_windows": FAILURES,
        "rows_inside_failure_windows": sum(failure_counts),
        "failure_window_row_counts": failure_counts,
        "checks": checks,
        "passed": all(checks.values()),
        "model_run": False,
        "candidate_scoring_authorized": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("rows", "first_timestamp", "last_timestamp",
                       "median_nonnegative_gap_seconds", "maximum_nonnegative_gap_seconds",
                       "rows_inside_failure_windows", "passed")}, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
