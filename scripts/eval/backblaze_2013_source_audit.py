#!/usr/bin/env python3
"""Audit the frozen Backblaze 2013 source without reading SMART values."""
from __future__ import annotations

import csv
from datetime import date
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

from sara_engine.utils.project_paths import processed_data_path, raw_data_path, workspace_path

SOURCE = Path(raw_data_path("r2_backblaze_2013", "data_2013.zip"))
PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "backblaze_2013_source_audit_v2.json"))
OUTPUT = Path(workspace_path("evaluation", "backblaze_2013_source_audit_v2.json"))
PROTOCOL_SHA256 = "26af53b5c247567f65c804c687b8d53f7a4b3383afd8162649c682dd4309f7f4"


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _daily_members(archive: zipfile.ZipFile) -> list[str]:
    members = []
    for name in archive.namelist():
        parts = Path(name).parts
        if len(parts) == 2 and parts[0] == "2013" and parts[1].endswith(".csv"):
            members.append(name)
    return sorted(members)


def _reader(archive: zipfile.ZipFile, member: str):
    raw = archive.open(member)
    text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
    return raw, text, csv.reader(text)


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("Backblaze 2013 source audit V2 already exists")
    if _digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Backblaze 2013 audit protocol digest changed")
    protocol = json.loads(PROTOCOL.read_text())
    source_digest = _digest(SOURCE)
    if source_digest != protocol["source_sha256"]:
        raise ValueError("Backblaze 2013 source digest changed")

    required = tuple(protocol["required_columns"])
    failure_dates: dict[str, list[date]] = {}
    failure_models: dict[str, set[str]] = {}
    total_rows = 0
    invalid_failure_values = 0
    duplicate_serial_days = 0
    filename_date_mismatches = 0
    missing_required_columns: list[str] = []
    distinct_serials: set[str] = set()
    distinct_models: set[str] = set()

    with zipfile.ZipFile(SOURCE) as archive:
        members = _daily_members(archive)
        for member in members:
            raw, text, reader = _reader(archive, member)
            try:
                header = next(reader)
                indices = {name: header.index(name) for name in required if name in header}
                if len(indices) != len(required):
                    missing_required_columns.append(member)
                    continue
                member_date = date.fromisoformat(Path(member).stem)
                seen_serials: set[str] = set()
                for row in reader:
                    total_rows += 1
                    row_date = date.fromisoformat(row[indices["date"]])
                    serial = row[indices["serial_number"]]
                    model = row[indices["model"]]
                    failure = row[indices["failure"]]
                    filename_date_mismatches += row_date != member_date
                    duplicate_serial_days += serial in seen_serials
                    seen_serials.add(serial)
                    distinct_serials.add(serial)
                    distinct_models.add(model)
                    if failure not in {"0", "1"}:
                        invalid_failure_values += 1
                    elif failure == "1":
                        failure_dates.setdefault(serial, []).append(row_date)
                        failure_models.setdefault(serial, set()).add(model)
            finally:
                text.close()
                raw.close()

        prior_rows = {serial: 0 for serial in failure_dates}
        rows_after_failure = {serial: 0 for serial in failure_dates}
        model_mismatches = 0
        canonical_failure_date = {
            serial: min(dates) for serial, dates in failure_dates.items()
        }
        for member in members:
            raw, text, reader = _reader(archive, member)
            try:
                header = next(reader)
                indices = {name: header.index(name) for name in required if name in header}
                if len(indices) != len(required):
                    continue
                for row in reader:
                    serial = row[indices["serial_number"]]
                    if serial not in canonical_failure_date:
                        continue
                    row_date = date.fromisoformat(row[indices["date"]])
                    if row_date <= canonical_failure_date[serial]:
                        prior_rows[serial] += 1
                    else:
                        rows_after_failure[serial] += 1
                    model_mismatches += row[indices["model"]] not in failure_models[serial]
            finally:
                text.close()
                raw.close()

    training_end = date.fromisoformat(protocol["split_boundaries"]["training_end"])
    development_end = date.fromisoformat(protocol["split_boundaries"]["development_end"])
    heldout_start = date.fromisoformat(protocol["split_boundaries"]["heldout_start"])
    failures_by_split = {"training": 0, "development": 0, "heldout": 0}
    for failure_date in canonical_failure_date.values():
        if failure_date <= training_end:
            failures_by_split["training"] += 1
        elif failure_date <= development_end:
            failures_by_split["development"] += 1
        elif failure_date >= heldout_start:
            failures_by_split["heldout"] += 1

    minimums = protocol["minimums"]
    failed_with_history = sum(value >= 30 for value in prior_rows.values())
    checks = {
        "daily_files": len(members) >= minimums["daily_files"],
        "required_columns": not missing_required_columns,
        "row_dates_match_members": filename_date_mismatches == 0,
        "one_row_per_drive_day": duplicate_serial_days == 0,
        "binary_failure_flag": invalid_failure_values == 0,
        "one_failure_row_per_failed_drive": all(len(dates) == 1 for dates in failure_dates.values()),
        "post_failure_rows_are_deterministically_excluded": all(
            failure_dates[serial] and canonical_failure_date[serial] == min(failure_dates[serial])
            for serial in rows_after_failure),
        "failure_model_consistent": model_mismatches == 0,
        "failed_drives_total": len(failure_dates) >= minimums["failed_drives_total"],
        "failed_drives_with_30_prior_rows": failed_with_history >= minimums["failed_drives_with_30_prior_rows"],
        "failures_per_split": all(
            value >= minimums["failures_per_split"] for value in failures_by_split.values()),
        "distinct_models": len(distinct_models) >= minimums["distinct_models"],
    }
    result = {
        "schema": protocol["schema"],
        "audit_id": protocol["audit_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "source_sha256": source_digest,
        "daily_files": len(members),
        "rows": total_rows,
        "distinct_drives": len(distinct_serials),
        "distinct_models": len(distinct_models),
        "failure_rows": sum(len(dates) for dates in failure_dates.values()),
        "failed_drives": len(failure_dates),
        "failed_drives_with_30_prior_rows": failed_with_history,
        "failures_by_split": failures_by_split,
        "duplicate_serial_days": duplicate_serial_days,
        "filename_date_mismatches": filename_date_mismatches,
        "invalid_failure_values": invalid_failure_values,
        "failed_drive_rows_after_failure": sum(rows_after_failure.values()),
        "failure_model_mismatches": model_mismatches,
        "missing_required_columns": missing_required_columns,
        "checks": checks,
        "passed": all(checks.values()),
        "smart_values_read": False,
        "candidate_scored": False,
        "heldout_feature_rows_materialized": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    summary_keys = (
        "daily_files", "rows", "distinct_drives", "distinct_models", "failed_drives",
        "failed_drives_with_30_prior_rows", "failures_by_split", "passed")
    print(json.dumps({key: result[key] for key in summary_keys}, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
