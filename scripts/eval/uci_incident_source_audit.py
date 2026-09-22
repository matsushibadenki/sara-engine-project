#!/usr/bin/env python3
"""Audit prospective incident-closure support using only source metadata columns."""
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
PROTOCOL = Path(processed_data_path("benchmark_fixtures", "uci_incident_source_audit_v1.json"))
OUTPUT = Path(workspace_path("evaluation", "uci_incident_source_audit_v1.json"))
PROTOCOL_SHA256 = "15182623c0bfbe35c7f70ad456e7f73086a5c958189bcb71330c392b5571b2c2"
MEMBER = "incident_event_log.csv"


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _time(value: str) -> datetime | None:
    if not value or value == "?":
        return None
    try:
        return datetime.strptime(value, "%d/%m/%Y %H:%M")
    except ValueError:
        return None


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("UCI incident source audit already exists")
    if _digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("UCI incident audit protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    if _digest(SOURCE) != protocol["source_sha256"]:
        raise ValueError("UCI incident source changed")
    histories: dict[str, dict] = {}
    event_rows = 0
    missing_required = []
    with zipfile.ZipFile(SOURCE) as archive:
        with archive.open(MEMBER) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source)
            header = next(reader)
            required = tuple(protocol["required_columns"])
            missing_required = [name for name in required if name not in header]
            if missing_required:
                raise ValueError("Required audit column is missing")
            indices = {name: header.index(name) for name in required}
            for row in reader:
                event_rows += 1
                case_id = row[indices["number"]]
                incident = histories.setdefault(case_id, {
                    "opened": None, "closed": None, "updates": [], "resolved": None,
                    "close_conflicts": 0, "open_conflicts": 0,
                })
                opened = _time(row[indices["opened_at"]])
                closed = _time(row[indices["closed_at"]])
                resolved = _time(row[indices["resolved_at"]])
                updated = _time(row[indices["sys_updated_at"]])
                if opened is not None:
                    if incident["opened"] is not None and incident["opened"] != opened:
                        incident["open_conflicts"] += 1
                    incident["opened"] = opened if incident["opened"] is None else incident["opened"]
                if closed is not None:
                    if incident["closed"] is not None and incident["closed"] != closed:
                        incident["close_conflicts"] += 1
                    incident["closed"] = closed if incident["closed"] is None else incident["closed"]
                if resolved is not None:
                    incident["resolved"] = resolved if incident["resolved"] is None else min(incident["resolved"], resolved)
                if updated is not None:
                    incident["updates"].append((updated, row[indices["incident_state"]]))
    outcomes = Counter()
    opening_months = Counter()
    eligible_months = Counter()
    conflicts = Counter()
    horizon = timedelta(days=protocol["candidate_task"]["horizon_days"])
    for incident in histories.values():
        opened = incident["opened"]
        closed = incident["closed"]
        conflicts["open"] += incident["open_conflicts"]
        conflicts["close"] += incident["close_conflicts"]
        if opened is not None:
            opening_months[opened.strftime("%Y-%m")] += 1
        if opened is None or closed is None or incident["open_conflicts"] or incident["close_conflicts"]:
            outcomes["missing_or_conflicting_endpoint"] += 1
            continue
        updates = sorted(set(incident["updates"]))
        preterminal = [time for time, state in updates
                       if time < closed
                       and (incident["resolved"] is None or time < incident["resolved"])
                       and state.lower() not in {"resolved", "closed", "canceled"}]
        unique_times = sorted(set(preterminal))
        if len(unique_times) < 2:
            outcomes["insufficient_preterminal_updates"] += 1
            continue
        cutoff = unique_times[1]
        if cutoff <= opened or closed <= cutoff:
            outcomes["invalid_cutoff"] += 1
            continue
        label = "positive" if closed <= cutoff + horizon else "negative"
        outcomes[label] += 1
        eligible_months[opened.strftime("%Y-%m")] += 1
    minimums = protocol["minimums"]
    checks = {
        "events": event_rows >= minimums["events"],
        "independent_cases": len(histories) >= minimums["independent_cases"],
        "eligible_positives": outcomes["positive"] >= minimums["eligible_positives"],
        "eligible_negatives": outcomes["negative"] >= minimums["eligible_negatives"],
        "audit_columns_present": not missing_required,
    }
    result = {
        "schema": protocol["schema"],
        "protocol_sha256": PROTOCOL_SHA256,
        "source_sha256": protocol["source_sha256"],
        "events": event_rows,
        "independent_cases": len(histories),
        "outcomes": dict(outcomes),
        "opening_months": dict(sorted(opening_months.items())),
        "eligible_opening_months": dict(sorted(eligible_months.items())),
        "endpoint_conflict_rows": dict(conflicts),
        "checks": checks,
        "passed": all(checks.values()),
        "non_audit_attributes_read": False,
        "candidate_scored": False,
        "heldout_features_materialized": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
