#!/usr/bin/env python3
"""Build the frozen Backblaze drive-level manifest without reading SMART values."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

from sara_engine.utils.project_paths import processed_data_path, raw_data_path, workspace_path

SOURCE = Path(raw_data_path("r2_backblaze_2013", "data_2013.zip"))
PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "backblaze_2013_case_manifest_protocol_v1.json"))
MANIFEST = Path(processed_data_path("benchmark_fixtures", "backblaze_2013_case_manifest_v1.jsonl"))
REPORT = Path(workspace_path("evaluation", "backblaze_2013_case_manifest_v1.json"))
PROTOCOL_SHA256 = "c0d20d0fe81aef11ed019c7ddbf6194e6c28237e6770c53925e0f21eda0d9378"


@dataclass
class Drive:
    first: date
    last: date
    model: str
    observed_days: int = 0
    failure: date | None = None
    observed_through_failure: int | None = None
    model_consistent: bool = True


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split(endpoint: date, boundaries: dict[str, str]) -> str:
    if endpoint <= date.fromisoformat(boundaries["training_end"]):
        return "training"
    if endpoint <= date.fromisoformat(boundaries["development_end"]):
        return "development"
    if endpoint >= date.fromisoformat(boundaries["heldout_start"]):
        return "heldout"
    raise ValueError("Endpoint is outside the frozen split boundaries")


def _case_id(protocol_id: str, serial: str) -> str:
    return sha256(f"{protocol_id}|case|{serial}".encode()).hexdigest()[:24]


def _rank(protocol_id: str, split: str, model: str, serial: str) -> str:
    return sha256(f"{protocol_id}|{split}|{model}|{serial}".encode()).hexdigest()


def main() -> int:
    if MANIFEST.exists() or REPORT.exists():
        raise ValueError("Backblaze case-manifest attempt already exists")
    if _digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Backblaze case-manifest protocol digest changed")
    protocol = json.loads(PROTOCOL.read_text())
    if _digest(SOURCE) != protocol["source_sha256"]:
        raise ValueError("Backblaze source digest changed")

    drives: dict[str, Drive] = {}
    required = ("date", "serial_number", "model", "failure")
    with zipfile.ZipFile(SOURCE) as archive:
        members = sorted(name for name in archive.namelist()
                         if len(Path(name).parts) == 2
                         and Path(name).parts[0] == "2013"
                         and name.endswith(".csv"))
        for member in members:
            raw = archive.open(member)
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            try:
                reader = csv.reader(text)
                header = next(reader)
                indices = {name: header.index(name) for name in required}
                for row in reader:
                    observed = date.fromisoformat(row[indices["date"]])
                    serial = row[indices["serial_number"]]
                    model = row[indices["model"]]
                    drive = drives.get(serial)
                    if drive is None:
                        drive = Drive(first=observed, last=observed, model=model)
                        drives[serial] = drive
                    drive.first = min(drive.first, observed)
                    drive.last = max(drive.last, observed)
                    drive.observed_days += 1
                    drive.model_consistent &= drive.model == model
                    if row[indices["failure"]] == "1":
                        drive.failure = observed if drive.failure is None else min(drive.failure, observed)
                        if drive.observed_through_failure is None:
                            drive.observed_through_failure = drive.observed_days
            finally:
                text.close()
                raw.close()

    minimum_days = protocol["case_definition"]["minimum_observed_days"]
    positives: dict[tuple[str, str], list[tuple[str, Drive]]] = {}
    negatives: dict[tuple[str, str], list[tuple[str, Drive]]] = {}
    for serial, drive in drives.items():
        endpoint = drive.failure or drive.last
        observed_through_endpoint = (
            drive.observed_through_failure
            if drive.failure is not None else drive.observed_days)
        if observed_through_endpoint < minimum_days or not drive.model_consistent:
            continue
        split = _split(endpoint, protocol["split_boundaries"])
        target = positives if drive.failure is not None else negatives
        target.setdefault((split, drive.model), []).append((serial, drive))

    selected: list[dict[str, object]] = []
    unmatched_positive = 0
    protocol_id = protocol["protocol_id"]
    for split in ("training", "development", "heldout"):
        models = sorted({model for split_name, model in positives if split_name == split})
        for model in models:
            positive_rows = sorted(
                positives.get((split, model), []),
                key=lambda item: _rank(protocol_id, split, model, item[0]))
            negative_rows = sorted(
                negatives.get((split, model), []),
                key=lambda item: _rank(protocol_id, split, model, item[0]))
            pair_count = min(len(positive_rows), len(negative_rows))
            unmatched_positive += len(positive_rows) - pair_count
            for label, rows in ((1, positive_rows[:pair_count]), (0, negative_rows[:pair_count])):
                for serial, drive in rows:
                    endpoint = drive.failure or drive.last
                    selected.append({
                        "case_id": _case_id(protocol_id, serial),
                        "split": split,
                        "label": label,
                        "endpoint_date": endpoint.isoformat(),
                        "observed_days": (
                            drive.observed_through_failure
                            if drive.failure is not None else drive.observed_days),
                        "model_group": sha256(model.encode()).hexdigest()[:16],
                        "source_member_start": f"2013/{drive.first.isoformat()}.csv",
                        "source_member_end": f"2013/{endpoint.isoformat()}.csv",
                    })

    selected.sort(key=lambda row: (str(row["split"]), str(row["case_id"])))
    counts = {split: {"positive": 0, "negative": 0} for split in
              ("training", "development", "heldout")}
    for row in selected:
        counts[str(row["split"])]["positive" if row["label"] == 1 else "negative"] += 1
    case_ids = [str(row["case_id"]) for row in selected]
    minimums = protocol["minimum_matched_cases_per_class"]
    checks = {
        "minimums": all(counts[split][label] >= minimums[split]
                        for split in counts for label in ("positive", "negative")),
        "balanced": all(value["positive"] == value["negative"] for value in counts.values()),
        "unique_case_ids": len(case_ids) == len(set(case_ids)),
        "minimum_history": all(int(row["observed_days"]) >= minimum_days for row in selected),
        "model_groups_present": all(bool(row["model_group"]) for row in selected),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w") as output:
        for row in selected:
            output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    report = {
        "schema": "sara-backblaze-2013-case-manifest-result-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "source_sha256": protocol["source_sha256"],
        "manifest_sha256": _digest(MANIFEST),
        "cases": len(selected),
        "counts": counts,
        "unmatched_eligible_positive_cases": unmatched_positive,
        "checks": checks,
        "passed": all(checks.values()),
        "smart_values_read": False,
        "candidate_scored": False,
        "heldout_feature_rows_materialized": False,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
