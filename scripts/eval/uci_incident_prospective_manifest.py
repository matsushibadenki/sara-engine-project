#!/usr/bin/env python3
"""Build a case-isolated prospective closure manifest from outcome metadata only."""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timedelta
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

from sara_engine.utils.project_paths import processed_data_path, raw_data_path, workspace_path

SOURCE = Path(raw_data_path("r2_uci_incident", "incident_management.zip"))
PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "uci_incident_prospective_manifest_protocol_v1.json"))
MANIFEST = Path(processed_data_path(
    "benchmark_fixtures", "uci_incident_prospective_manifest_v1.jsonl"))
REPORT = Path(workspace_path("evaluation", "uci_incident_prospective_manifest_v1.json"))
PROTOCOL_SHA256 = "c2f95eb5e613d67a3454cf4f47568e2348ab3fe6021e0e067c284e43e07b421d"


def _time(value: str) -> datetime | None:
    if not value or value == "?":
        return None
    try:
        return datetime.strptime(value, "%d/%m/%Y %H:%M")
    except ValueError:
        return None


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(protocol_id: str, incident_number: str) -> str:
    return sha256(f"{protocol_id}|case|{incident_number}".encode()).hexdigest()[:24]


def main() -> int:
    if MANIFEST.exists() or REPORT.exists():
        raise ValueError("UCI incident prospective manifest already exists")
    if _digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("UCI incident manifest protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    if _digest(SOURCE) != protocol["source_sha256"]:
        raise ValueError("UCI incident source changed")
    incidents = {}
    with zipfile.ZipFile(SOURCE) as archive:
        with archive.open("incident_event_log.csv") as raw, io.TextIOWrapper(
            raw, encoding="utf-8-sig", newline=""
        ) as source:
            reader = csv.reader(source)
            header = next(reader)
            indices = {name: header.index(name) for name in (
                "number", "incident_state", "opened_at", "sys_updated_at", "resolved_at", "closed_at")}
            for row in reader:
                number = row[indices["number"]]
                incident = incidents.setdefault(number, {
                    "opened": set(), "resolved": set(), "closed": set(), "updates": []})
                for key, column in (("opened", "opened_at"), ("resolved", "resolved_at"),
                                    ("closed", "closed_at")):
                    value = _time(row[indices[column]])
                    if value is not None:
                        incident[key].add(value)
                updated = _time(row[indices["sys_updated_at"]])
                if updated is not None:
                    incident["updates"].append((updated, row[indices["incident_state"]]))
    month_to_split = {month: split for split, months in protocol["case_month_split"].items()
                      for month in months}
    candidates: dict[str, dict[int, list[dict]]] = {
        split: {0: [], 1: []} for split in protocol["case_month_split"]}
    exclusion = Counter()
    horizon = timedelta(days=protocol["outcome_rule"]["horizon_days"])
    for number, incident in incidents.items():
        if len(incident["opened"]) != 1 or len(incident["closed"]) != 1 or len(incident["resolved"]) > 1:
            exclusion["missing_or_conflicting_endpoint"] += 1
            continue
        opened = next(iter(incident["opened"]))
        closed = next(iter(incident["closed"]))
        split = month_to_split.get(opened.strftime("%Y-%m"))
        if split is None:
            exclusion["outside_frozen_months"] += 1
            continue
        resolved = next(iter(incident["resolved"])) if incident["resolved"] else None
        preterminal = sorted({updated for updated, state in incident["updates"]
                              if opened < updated < closed
                              and (resolved is None or updated < resolved)
                              and state.lower() not in {"resolved", "closed", "canceled"}})
        if len(preterminal) < 2:
            exclusion["insufficient_preterminal_updates"] += 1
            continue
        cutoff = preterminal[1]
        if closed <= cutoff:
            exclusion["invalid_cutoff"] += 1
            continue
        label = int(closed <= cutoff + horizon)
        candidates[split][label].append({
            "case_id": _identity(protocol["protocol_id"], number),
            "split": split,
            "label": label,
            "opened_at": opened.isoformat(),
            "feature_cutoff": cutoff.isoformat(),
            "outcome_window_end": (cutoff + horizon).isoformat(),
            "preterminal_update_count": len(preterminal),
        })
    selected = []
    counts = {}
    for split, classes in candidates.items():
        count = min(len(classes[0]), len(classes[1]))
        counts[split] = {"eligible_negative": len(classes[0]),
                         "eligible_positive": len(classes[1]),
                         "selected_per_class": count}
        for label in (0, 1):
            classes[label].sort(key=lambda row: sha256(
                f"{protocol['protocol_id']}|{split}|{row['case_id']}".encode()).hexdigest())
            selected.extend(classes[label][:count])
    selected.sort(key=lambda row: (row["split"], row["case_id"]))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                                for row in selected))
    minimums = protocol["sampling"]["minimum_per_class"]
    checks = {
        "class_support": all(counts[split]["selected_per_class"] >= minimums[split]
                             for split in counts),
        "unique_case_ids": len(selected) == len({row["case_id"] for row in selected}),
        "cutoff_precedes_outcome": all(row["feature_cutoff"] < row["outcome_window_end"]
                                       for row in selected),
        "opened_before_cutoff": all(row["opened_at"] < row["feature_cutoff"] for row in selected),
        "future_outcome_absent_from_features": True,
    }
    result = {
        "schema": "sara-uci-incident-prospective-manifest-result-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "source_sha256": protocol["source_sha256"],
        "manifest_sha256": _digest(MANIFEST),
        "cases": len(selected),
        "counts": counts,
        "exclusions": dict(exclusion),
        "checks": checks,
        "passed": all(checks.values()),
        "non_audit_attributes_read": False,
        "candidate_scored": False,
        "heldout_features_materialized": False,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"cases": len(selected), "counts": counts,
                      "checks": checks, "passed": result["passed"]}, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
