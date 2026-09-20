#!/usr/bin/env python3
"""Build an independent 2014 case manifest excluding every 2013 drive."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import csv
import io
import json
from pathlib import Path
import zipfile

from sara_engine.utils.project_paths import processed_data_path, raw_data_path, workspace_path

SOURCE = Path(raw_data_path("r2_backblaze_2014", "data_2014.zip"))
PRIOR = Path(raw_data_path("r2_backblaze_2013", "data_2013.zip"))
PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "backblaze_2014_case_manifest_protocol_v1.json"))
MANIFEST = Path(processed_data_path("benchmark_fixtures", "backblaze_2014_case_manifest_v1.jsonl"))
REPORT = Path(workspace_path("evaluation", "backblaze_2014_case_manifest_v1.json"))
PROTOCOL_SHA256 = "9914d86cdb04f10830f506107160f5942206e5c995ad5528a1d6d0cf17217b05"


@dataclass
class Drive:
    first: date
    last: date
    model: str
    rows: int = 0
    failure: date | None = None
    rows_at_failure: int | None = None


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


def _serials(path: Path, year: str) -> set[str]:
    result = set()
    with zipfile.ZipFile(path) as archive:
        for member in _members(archive, year):
            with archive.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
                reader = csv.reader(text)
                header = next(reader)
                index = header.index("serial_number")
                result.update(row[index] for row in reader)
    return result


def _split(endpoint: date, boundaries: dict[str, str]) -> str:
    if endpoint <= date.fromisoformat(boundaries["training_end"]):
        return "training"
    if date.fromisoformat(boundaries["development_start"]) <= endpoint <= date.fromisoformat(boundaries["development_end"]):
        return "development"
    if endpoint >= date.fromisoformat(boundaries["heldout_start"]):
        return "heldout"
    raise ValueError("Endpoint outside frozen split")


def main() -> int:
    if MANIFEST.exists() or REPORT.exists():
        raise ValueError("Backblaze 2014 manifest attempt already exists")
    if _digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Backblaze 2014 manifest protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    prior = _serials(PRIOR, "2013")
    drives: dict[str, Drive] = {}
    with zipfile.ZipFile(SOURCE) as archive:
        for member in _members(archive, "2014"):
            with archive.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
                reader = csv.reader(text)
                header = next(reader)
                indices = {name: header.index(name) for name in ("date", "serial_number", "model", "failure")}
                for row in reader:
                    serial = row[indices["serial_number"]]
                    if serial in prior:
                        continue
                    observed = date.fromisoformat(row[indices["date"]])
                    drive = drives.setdefault(serial, Drive(observed, observed, row[indices["model"]]))
                    drive.last = observed
                    drive.rows += 1
                    if row[indices["failure"]] == "1" and drive.failure is None:
                        drive.failure = observed
                        drive.rows_at_failure = drive.rows
    positives = {}
    negatives = {}
    minimum_days = protocol["case_definition"]["minimum_observed_days"]
    for serial, drive in drives.items():
        observed_days = drive.rows_at_failure if drive.failure else drive.rows
        if observed_days is None or observed_days < minimum_days:
            continue
        endpoint = drive.failure or drive.last
        target = positives if drive.failure else negatives
        target.setdefault((_split(endpoint, protocol["split_boundaries"]), drive.model), []).append((serial, drive))
    selected = []
    pid = protocol["protocol_id"]
    for split in ("training", "development", "heldout"):
        models = sorted(model for split_name, model in positives if split_name == split)
        for model in models:
            rank = lambda item: sha256(f"{pid}|{split}|{model}|{item[0]}".encode()).hexdigest()
            pos = sorted(positives[(split, model)], key=rank)
            neg = sorted(negatives.get((split, model), []), key=rank)
            count = min(len(pos), len(neg))
            for label, candidates in ((1, pos[:count]), (0, neg[:count])):
                for serial, drive in candidates:
                    endpoint = drive.failure or drive.last
                    selected.append({
                        "case_id": sha256(f"{pid}|case|{serial}".encode()).hexdigest()[:24],
                        "split": split, "label": label, "endpoint_date": endpoint.isoformat(),
                        "observed_days": drive.rows_at_failure if drive.failure else drive.rows,
                        "model_group": sha256(model.encode()).hexdigest()[:16],
                        "source_member_start": f"2014/{drive.first.isoformat()}.csv",
                        "source_member_end": f"2014/{endpoint.isoformat()}.csv",
                    })
    selected.sort(key=lambda row: (row["split"], row["case_id"]))
    counts = {split: {"positive": 0, "negative": 0} for split in ("training", "development", "heldout")}
    for row in selected:
        counts[row["split"]]["positive" if row["label"] else "negative"] += 1
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in selected))
    minimums = protocol["minimum_matched_cases_per_class"]
    checks = {
        "minimums": all(counts[split][label] >= minimums[split] for split in counts for label in counts[split]),
        "balanced": all(row["positive"] == row["negative"] for row in counts.values()),
        "unique_cases": len(selected) == len({row["case_id"] for row in selected}),
        "minimum_history": all(row["observed_days"] >= minimum_days for row in selected),
        "prior_identity_excluded": all(serial not in prior for serial in drives),
    }
    report = {"schema": "sara-backblaze-2014-case-manifest-result-v1",
              "protocol_sha256": PROTOCOL_SHA256, "manifest_sha256": _digest(MANIFEST),
              "cases": len(selected), "counts": counts, "checks": checks,
              "passed": all(checks.values()), "smart_values_read": False,
              "candidate_scored": False, "heldout_feature_rows_materialized": False}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
