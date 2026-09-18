#!/usr/bin/env python3
"""Audit locally available R2 candidates without reading outcomes or scoring models."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from sara_engine.utils.project_paths import project_path, processed_data_path, workspace_path

PROTOCOL = Path(processed_data_path("benchmark_fixtures", "r2_entry_gate_audit_v1.json"))
OUTPUT = Path(workspace_path("evaluation", "r2_entry_gate_audit_v1.json"))
def main() -> int:
    if OUTPUT.exists():
        raise ValueError("R2 entry audit already exists")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["decision"] != "blocked_until_new_source" or protocol["candidate_scoring_authorized"]:
        raise ValueError("Entry-gate boundary changed")
    rows = []
    for candidate in protocol["candidates"]:
        path = Path(project_path(candidate["path"]))
        if not path.exists():
            raise ValueError(f"Pinned candidate missing: {candidate['id']}")
        rows.append({**candidate, "sha256": sha256(path.read_bytes()).hexdigest(),
                     "bytes": path.stat().st_size})
    result = {
        "schema": "sara-r2-entry-gate-audit-result-v1",
        "protocol_sha256": sha256(PROTOCOL.read_bytes()).hexdigest(),
        "candidates": rows,
        "eligible_candidates": [],
        "decision": "blocked_until_new_source",
        "candidate_scoring_authorized": False,
        "outcomes_read": False,
        "model_run": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"decision": result["decision"], "candidate_count": len(rows),
                      "eligible_candidates": result["eligible_candidates"]}))


if __name__ == "__main__":
    main()
