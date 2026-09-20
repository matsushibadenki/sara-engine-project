#!/usr/bin/env python3
"""Audit Backblaze 2014 metadata and exclude every 2013 drive identity."""
from __future__ import annotations

import csv
from datetime import date
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

from sara_engine.utils.project_paths import processed_data_path, raw_data_path, workspace_path

SOURCE = Path(raw_data_path("r2_backblaze_2014", "data_2014.zip"))
PRIOR = Path(raw_data_path("r2_backblaze_2013", "data_2013.zip"))
PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "backblaze_2014_source_audit_v1.json"))
OUTPUT = Path(workspace_path("evaluation", "backblaze_2014_source_audit_v1.json"))
PROTOCOL_SHA256 = "60a17620b9099a5655ab1a04766e0ee0058ff1ff3263ff696dff18aafbd41e05"


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _members(archive: zipfile.ZipFile, year: str) -> list[str]:
    return sorted(name for name in archive.namelist()
                  if len(Path(name).parts) == 2 and Path(name).parts[0] == year
                  and name.endswith(".csv"))


def _prior_serials() -> set[str]:
    serials = set()
    with zipfile.ZipFile(PRIOR) as archive:
        for member in _members(archive, "2013"):
            raw = archive.open(member)
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            try:
                reader = csv.reader(text)
                header = next(reader)
                index = header.index("serial_number")
                serials.update(row[index] for row in reader)
            finally:
                text.close()
                raw.close()
    return serials


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("Backblaze 2014 source-audit attempt already exists")
    if _digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Backblaze 2014 audit protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    if _digest(SOURCE) != protocol["source_sha256"] or _digest(PRIOR) != protocol["prior_source_sha256"]:
        raise ValueError("Backblaze source identity changed")
    prior_serials = _prior_serials()
    required = tuple(protocol["required_columns"])
    histories: dict[str, int] = {}
    failures: dict[str, date] = {}
    failure_history: dict[str, int] = {}
    models = set()
    all_serials = set()
    rows = invalid_failure = duplicate_serial_days = date_mismatch = post_failure_rows = 0
    with zipfile.ZipFile(SOURCE) as archive:
        members = _members(archive, "2014")
        for member in members:
            raw = archive.open(member)
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            try:
                reader = csv.reader(text)
                header = next(reader)
                indices = {name: header.index(name) for name in required}
                member_date = date.fromisoformat(Path(member).stem)
                seen = set()
                for row in reader:
                    rows += 1
                    observed = date.fromisoformat(row[indices["date"]])
                    serial = row[indices["serial_number"]]
                    failure = row[indices["failure"]]
                    all_serials.add(serial)
                    date_mismatch += observed != member_date
                    duplicate_serial_days += serial in seen
                    seen.add(serial)
                    if serial in prior_serials:
                        continue
                    models.add(row[indices["model"]])
                    histories[serial] = histories.get(serial, 0) + 1
                    if serial in failures:
                        post_failure_rows += 1
                    if failure not in {"0", "1"}:
                        invalid_failure += 1
                    elif failure == "1" and serial not in failures:
                        failures[serial] = observed
                        failure_history[serial] = histories[serial]
            finally:
                text.close()
                raw.close()
    boundaries = protocol["split_boundaries"]
    training_end = date.fromisoformat(boundaries["training_end"])
    development_start = date.fromisoformat(boundaries["development_start"])
    development_end = date.fromisoformat(boundaries["development_end"])
    heldout_start = date.fromisoformat(boundaries["heldout_start"])
    split_counts = {"training": 0, "development": 0, "heldout": 0}
    for failed in failures.values():
        if failed <= training_end:
            split_counts["training"] += 1
        elif development_start <= failed <= development_end:
            split_counts["development"] += 1
        elif failed >= heldout_start:
            split_counts["heldout"] += 1
    minimums = protocol["minimums"]
    history_eligible = sum(value >= 30 for value in failure_history.values())
    checks = {
        "daily_files": len(members) >= minimums["daily_files"],
        "row_dates_match_members": date_mismatch == 0,
        "one_row_per_drive_day": duplicate_serial_days == 0,
        "binary_failure_flag": invalid_failure == 0,
        "independent_failed_drives_total": len(failures) >= minimums["independent_failed_drives_total"],
        "independent_failed_drives_with_30_prior_rows": history_eligible >= minimums["independent_failed_drives_with_30_prior_rows"],
        "independent_failures_training": split_counts["training"] >= minimums["independent_failures_training"],
        "independent_failures_development": split_counts["development"] >= minimums["independent_failures_development"],
        "independent_failures_heldout": split_counts["heldout"] >= minimums["independent_failures_heldout"],
        "distinct_models": len(models) >= minimums["distinct_models"],
    }
    report = {
        "schema": protocol["schema"], "protocol_sha256": PROTOCOL_SHA256,
        "source_sha256": protocol["source_sha256"], "daily_files": len(members), "rows": rows,
        "prior_serials": len(prior_serials), "all_2014_serials": len(all_serials),
        "independent_2014_serials": len(set(histories)), "distinct_independent_models": len(models),
        "independent_failed_drives": len(failures),
        "independent_failed_drives_with_30_prior_rows": history_eligible,
        "independent_failures_by_split": split_counts,
        "post_failure_rows_recorded_for_exclusion": post_failure_rows,
        "checks": checks, "passed": all(checks.values()), "smart_values_read": False,
        "candidate_scored": False, "development_feature_rows_materialized": False,
        "heldout_feature_rows_materialized": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
