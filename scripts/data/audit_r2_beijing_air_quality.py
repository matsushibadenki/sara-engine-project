#!/usr/bin/env python3
"""Audit the immutable UCI Beijing air-quality archive for an R2 task."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import zipfile

from sara_engine.utils.project_paths import ensure_parent_directory, raw_data_path, workspace_path


OUTER_SHA256 = "b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4"
INNER_SHA256 = "d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8"
EXPECTED_COLUMNS = [
    "No", "year", "month", "day", "hour", "PM2.5", "PM10", "SO2", "NO2",
    "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM", "station",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_audit() -> dict:
    outer = Path(raw_data_path("r2_beijing_air_quality", "beijing_multi_site_air_quality.zip"))
    inner = Path(raw_data_path("r2_beijing_air_quality", "PRSA2017_Data_20130301-20170228.zip"))
    outer_hash = _sha256(outer)
    inner_hash = _sha256(inner)
    if outer_hash != OUTER_SHA256 or inner_hash != INNER_SHA256:
        raise ValueError("R2 source archive hash mismatch")

    stations = []
    global_start = None
    global_end = None
    with zipfile.ZipFile(inner) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".csv"))
        if len(names) != 12:
            raise ValueError("Expected exactly twelve station files")
        for name in names:
            missing = {column: 0 for column in EXPECTED_COLUMNS}
            row_count = 0
            valid_next_pm25_pairs = 0
            rises = 0
            previous_time = None
            previous_pm25 = None
            chronological = True
            hourly_gaps = 0
            first_time = None
            last_time = None
            with archive.open(name) as raw:
                lines = (line.decode("utf-8-sig") for line in raw)
                reader = csv.DictReader(lines)
                if reader.fieldnames != EXPECTED_COLUMNS:
                    raise ValueError(f"Unexpected columns in {name}")
                for row in reader:
                    current_time = datetime(int(row["year"]), int(row["month"]), int(row["day"]), int(row["hour"]))
                    pm25 = None if row["PM2.5"] == "NA" else float(row["PM2.5"])
                    if previous_time is not None:
                        chronological = chronological and current_time > previous_time
                        if current_time - previous_time != timedelta(hours=1):
                            hourly_gaps += 1
                        if previous_pm25 is not None and pm25 is not None and current_time - previous_time == timedelta(hours=1):
                            valid_next_pm25_pairs += 1
                            rises += int(pm25 > previous_pm25)
                    for column in EXPECTED_COLUMNS:
                        missing[column] += int(row[column] == "NA" or row[column] == "")
                    row_count += 1
                    first_time = first_time or current_time
                    last_time = current_time
                    previous_time = current_time
                    previous_pm25 = pm25
            global_start = first_time if global_start is None else min(global_start, first_time)
            global_end = last_time if global_end is None else max(global_end, last_time)
            stations.append({
                "archive_member": name,
                "station": Path(name).stem.removeprefix("PRSA_Data_").removesuffix("_20130301-20170228"),
                "rows": row_count,
                "start": first_time.isoformat(),
                "end": last_time.isoformat(),
                "strictly_chronological": chronological,
                "non_hourly_gaps": hourly_gaps,
                "missing": missing,
                "valid_consecutive_pm25_pairs": valid_next_pm25_pairs,
                "next_hour_pm25_rise_rate": rises / valid_next_pm25_pairs,
            })
    return {
        "schema": "sara-r2-beijing-source-audit-v1",
        "source": {
            "title": "Beijing Multi-Site Air Quality",
            "publisher": "UCI Machine Learning Repository",
            "dataset_id": 501,
            "doi": "10.24432/C5RK5G",
            "source_url": "https://archive.ics.uci.edu/dataset/501/beijing",
            "download_url": "https://archive.ics.uci.edu/static/public/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip",
            "license": "CC BY 4.0",
            "outer_sha256": outer_hash,
            "inner_sha256": inner_hash,
        },
        "task_candidate": {
            "prediction_time": "hour t after observing the complete row at t",
            "outcome_time": "hour t+1",
            "target": "whether PM2.5 at t+1 is greater than PM2.5 at t",
            "missing_pair_policy": "exclude pairs with a missing PM2.5 value or a non-hourly gap",
            "future_feature_access": False,
        },
        "global_start": global_start.isoformat(),
        "global_end": global_end.isoformat(),
        "station_count": len(stations),
        "total_rows": sum(item["rows"] for item in stations),
        "total_valid_consecutive_pm25_pairs": sum(item["valid_consecutive_pm25_pairs"] for item in stations),
        "all_station_rows_equal": len({item["rows"] for item in stations}) == 1,
        "all_strictly_chronological": all(item["strictly_chronological"] for item in stations),
        "stations": stations,
    }


def main() -> int:
    output = Path(ensure_parent_directory(workspace_path("evaluation", "r2_beijing_source_audit.json")))
    audit = build_audit()
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "stations": audit["station_count"],
        "rows": audit["total_rows"], "valid_pairs": audit["total_valid_consecutive_pm25_pairs"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
