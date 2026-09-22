#!/usr/bin/env python3
"""Check untouched incident month support using frozen metadata only."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from sara_engine.utils.project_paths import processed_data_path, workspace_path

PROTOCOL = Path(processed_data_path("benchmark_fixtures", "uci_incident_fresh_boundary_audit_protocol_v1.json"))
SOURCE_AUDIT = Path(workspace_path("evaluation", "uci_incident_source_audit_v1.json"))
MANIFEST = Path(processed_data_path("benchmark_fixtures", "uci_incident_prospective_manifest_v1.jsonl"))
OUTPUT = Path(workspace_path("evaluation", "uci_incident_fresh_boundary_audit_v1.json"))
PROTOCOL_SHA256 = "6608591d6287556a05fc447e3ce337fc040bc088592c9ce5d312392c4bce7820"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("Fresh boundary audit already exists")
    if digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Fresh boundary protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    if digest(SOURCE_AUDIT) != protocol["source_audit_sha256"] or digest(MANIFEST) != protocol["prospective_manifest_sha256"]:
        raise ValueError("Pinned metadata changed")
    audit = json.loads(SOURCE_AUDIT.read_text())
    minimum_total = 2 * protocol["minimum_cases_per_class"]
    unused = {month: count for month, count in audit["eligible_opening_months"].items()
              if month not in protocol["consumed_months"]}
    eligible = {month: count for month, count in unused.items() if count >= minimum_total}
    result = {
        "schema": "sara-uci-incident-fresh-boundary-audit-result-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "input_audit_sha256": protocol["source_audit_sha256"],
        "unused_month_upper_bounds": unused,
        "minimum_total_before_balance": minimum_total,
        "maximum_unused_month_support": max(unused.values()),
        "potentially_eligible_months": eligible,
        "decision": "new_month_available" if eligible else protocol["decision_if_no_month_qualifies"],
        "raw_source_rows_read": False,
        "descriptive_features_read": False,
        "candidate_scored": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
