#!/usr/bin/env python3
"""One-shot timestamp-only feasibility audit for MetroPT-3 report windows."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import zipfile

from sara_engine.utils.project_paths import processed_data_path, raw_data_path, workspace_path

PROTOCOL = Path(processed_data_path("benchmark_fixtures", "metropt3_report_feasibility_v1.json"))
PROTOCOL_SHA256 = "2058feb93d760777f861348c30041b8f1da19e7689230ab5e90c5c0a8c017476"
SOURCE = Path(raw_data_path("r2_metropt3", "metropt3.zip"))
OUTPUT = Path(workspace_path("evaluation", "metropt3_report_feasibility_v1.json"))


def _time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def _overlap(left: tuple[datetime, datetime], right: tuple[datetime, datetime]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("MetroPT-3 feasibility result already exists")
    raw_protocol = PROTOCOL.read_bytes()
    if sha256(raw_protocol).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("MetroPT-3 feasibility protocol changed")
    protocol = json.loads(raw_protocol)
    if sha256(SOURCE.read_bytes()).hexdigest() != protocol["source_sha256"]:
        raise ValueError("MetroPT-3 source changed")
    contract = protocol["window_contract"]
    windows = []
    failure_buffers = []
    for report in protocol["failure_reports"]:
        start, end = _time(report["start"]), _time(report["end"])
        positive = (start - timedelta(hours=contract["positive_window_hours_before_report_start"]),
                    start)
        negative_end = start - timedelta(days=contract["negative_window_offset_days_before_report_start"])
        negative = (negative_end - timedelta(hours=contract["negative_window_duration_hours"]),
                    negative_end)
        windows.extend((
            {"report_id": report["id"], "split": report["split"], "kind": "positive",
             "start": positive[0], "end": positive[1], "timestamps": []},
            {"report_id": report["id"], "split": report["split"], "kind": "negative",
             "start": negative[0], "end": negative[1], "timestamps": []},
        ))
        buffer = timedelta(hours=contract["failure_exclusion_buffer_hours"])
        failure_buffers.append((start - buffer, end + buffer))
    with zipfile.ZipFile(SOURCE) as archive, archive.open(protocol["source_member"]) as raw:
        reader = csv.DictReader(line.decode("utf-8-sig") for line in raw)
        for row in reader:
            timestamp = _time(row["timestamp"])
            for window in windows:
                if window["start"] <= timestamp < window["end"]:
                    window["timestamps"].append(timestamp)
    serialized = []
    coverage_checks = []
    for window in windows:
        timestamps = window.pop("timestamps")
        gaps = [(right - left).total_seconds() for left, right in zip(timestamps, timestamps[1:])]
        maximum_gap = max(gaps) if gaps else None
        median_gap = median(gaps) if gaps else None
        coverage = (len(timestamps) >= contract["minimum_rows_per_window"]
                    and maximum_gap is not None
                    and maximum_gap <= contract["maximum_within_window_gap_seconds"]
                    and median_gap <= contract["maximum_median_within_window_gap_seconds"])
        coverage_checks.append(coverage)
        serialized.append({**window, "start": window["start"].isoformat(),
                           "end": window["end"].isoformat(), "rows": len(timestamps),
                           "maximum_gap_seconds": maximum_gap,
                           "median_gap_seconds": median_gap,
                           "coverage_passed": coverage})
    negative_windows = [(_time(row["start"].replace("T", " ")),
                         _time(row["end"].replace("T", " ")))
                        for row in serialized if row["kind"] == "negative"]
    all_intervals = [(_time(row["start"].replace("T", " ")),
                      _time(row["end"].replace("T", " ")))
                     for row in serialized]
    counts = {split: sum(report["split"] == split for report in protocol["failure_reports"])
              for split in ("training", "development", "heldout")}
    expected = protocol["acceptance"]
    checks = {
        "report_split_counts": (counts["training"] == expected["training_positive_reports"]
                                and counts["development"] == expected["development_positive_reports"]
                                and counts["heldout"] == expected["heldout_positive_reports_reserved"]),
        "one_matched_negative_per_positive": len(negative_windows) * 2 == len(all_intervals),
        "all_windows_meet_coverage": all(coverage_checks),
        "negative_windows_outside_failure_buffers": all(
            not _overlap(window, failure_buffer)
            for window in negative_windows for failure_buffer in failure_buffers),
        "no_window_overlap": all(not _overlap(left, right)
                                 for index, left in enumerate(all_intervals)
                                 for right in all_intervals[index + 1:]),
    }
    result = {
        "schema": "sara-metropt3-report-feasibility-result-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "source_sha256": protocol["source_sha256"],
        "windows": serialized,
        "checks": checks,
        "feasibility_passed": all(checks.values()),
        "sensor_values_read": False,
        "candidate_scored": False,
        "heldout_sensor_rows_materialized": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"checks": checks, "windows": [
        {key: row[key] for key in ("report_id", "split", "kind", "rows",
                                   "maximum_gap_seconds", "coverage_passed")}
        for row in serialized], "feasibility_passed": result["feasibility_passed"]},
        sort_keys=True))
    return 0 if result["feasibility_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
