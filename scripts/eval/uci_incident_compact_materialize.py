#!/usr/bin/env python3
"""Materialize sealed pre-cutoff incident events for development only."""
from __future__ import annotations

import csv
from datetime import datetime
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

from sara_engine.utils.project_paths import processed_data_path, raw_data_path, workspace_path

SOURCE = Path(raw_data_path("r2_uci_incident", "incident_management.zip"))
PROTOCOL = Path(processed_data_path("benchmark_fixtures", "uci_incident_compact_development_protocol_v1.json"))
MANIFEST = Path(processed_data_path("benchmark_fixtures", "uci_incident_prospective_manifest_v1.jsonl"))
OUTPUT = Path(processed_data_path("benchmark_fixtures", "uci_incident_compact_development_v1.jsonl"))
REPORT = Path(workspace_path("evaluation", "uci_incident_compact_materialize_v1.json"))
PROTOCOL_SHA256 = "68ad7c0435ec1ad2d5f2553994d536742218cc0a0ff180c9d52224959f2befab"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def count_bucket(value: str) -> str:
    count = int(value)
    if count < 0:
        raise ValueError("Negative incident count")
    return str(count) if count < 3 else "3+"


def elapsed_bucket(hours: float) -> str:
    if hours < 1:
        return "<1"
    if hours < 24:
        return "1-23"
    if hours < 168:
        return "24-167"
    return "168+"


def selected_values(row: list[str], indices: dict[str, int], selected: dict[str, dict]) -> tuple[str, dict, dict, datetime] | None:
    """Gate case, cutoff and terminal state before reading descriptive fields."""
    number = row[indices["number"]]
    identity = sha256(f"uci-incident-7day-closure-v1-2026-09-22|case|{number}".encode()).hexdigest()[:24]
    item = selected.get(identity)
    if item is None:
        return None
    opened_raw = row[indices["opened_at"]]
    opened = datetime.strptime(opened_raw, "%d/%m/%Y %H:%M")
    if opened.isoformat() != item["opened_at"]:
        raise ValueError("Opening timestamp mismatch")
    updated_raw = row[indices["sys_updated_at"]]
    try:
        updated = datetime.strptime(updated_raw, "%d/%m/%Y %H:%M")
    except ValueError:
        return None
    cutoff = datetime.fromisoformat(item["feature_cutoff"])
    if not opened < updated <= cutoff:
        return None
    state = row[indices["incident_state"]]
    if state.lower() in {"resolved", "closed", "canceled"}:
        return None
    values = {name: row[index] for name, index in indices.items() if name != "number"}
    return identity, item, values, updated


def main() -> int:
    if OUTPUT.exists() or REPORT.exists():
        raise ValueError("Incident features were already materialized")
    if digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Compact development protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    if digest(SOURCE) != protocol["source_sha256"] or digest(MANIFEST) != protocol["manifest_sha256"]:
        raise ValueError("Pinned incident source or manifest changed")
    manifest = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
    selected = {item["case_id"]: item for item in manifest if item["split"] in protocol["allowed_splits"]}
    forbidden = set(protocol["forbidden_feature_columns"])
    allowed = tuple(protocol["lookup_only_columns"] + protocol["allowed_feature_columns"])
    if forbidden.intersection(allowed):
        raise ValueError("Feature allowlist overlaps forbidden columns")
    by_case: dict[str, dict[datetime, dict[str, str]]] = {case_id: {} for case_id in selected}
    with zipfile.ZipFile(SOURCE) as archive:
        with archive.open("incident_event_log.csv") as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            indices = {name: header.index(name) for name in allowed}
            for row in reader:
                admitted = selected_values(row, indices, selected)
                if admitted is None:
                    continue
                identity, item, values, updated = admitted
                by_case[identity].setdefault(updated, values)
    output_rows = []
    invalid = []
    for case_id, item in sorted(selected.items()):
        times = sorted(by_case[case_id])
        if len(times) != 2 or times[1].isoformat() != item["feature_cutoff"]:
            invalid.append(case_id)
            continue
        first, second = (by_case[case_id][stamp] for stamp in times)
        tokens = [
            "first_state=" + first["incident_state"],
            "second_state=" + second["incident_state"],
            "cutoff_reassignment_bucket=" + count_bucket(second["reassignment_count"]),
            "cutoff_reopen_bucket=" + count_bucket(second["reopen_count"]),
            "cutoff_impact=" + second["impact"],
            "cutoff_urgency=" + second["urgency"],
            "cutoff_priority=" + second["priority"],
            "cutoff_contact_type=" + second["contact_type"],
            "elapsed_hours_bucket=" + elapsed_bucket((times[1] - times[0]).total_seconds() / 3600),
        ]
        output_rows.append({"case_id": case_id, "split": item["split"], "label": item["label"],
                            "feature_cutoff": item["feature_cutoff"], "tokens": tokens})
    checks = {
        "all_development_cases": len(output_rows) == len(selected) and not invalid,
        "heldout_sealed": all(row["split"] in protocol["allowed_splits"] for row in output_rows),
        "exact_token_count": all(len(row["tokens"]) == 9 for row in output_rows),
        "cutoff_verified": not invalid,
        "feature_allowlist_disjoint": not forbidden.intersection(allowed),
    }
    passed = all(checks.values())
    if passed:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in output_rows))
    result = {"schema": "sara-uci-incident-compact-materialization-v1", "protocol_sha256": PROTOCOL_SHA256,
              "manifest_sha256": protocol["manifest_sha256"], "cases": len(output_rows),
              "invalid_case_count": len(invalid), "checks": checks, "passed": passed,
              "materialized_sha256": digest(OUTPUT) if passed else None,
              "candidate_scored": False, "heldout_features_materialized": False}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
