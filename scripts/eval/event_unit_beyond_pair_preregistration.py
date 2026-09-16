#!/usr/bin/env python3
"""Validate the immutable beyond-pair event-unit protocol."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path

PROTOCOL_PATH = Path(processed_data_path("benchmark_fixtures", "event_unit_beyond_pair_v1.json"))
PROTOCOL_SHA256 = "9a8005ed0d9e08adabadbf44249386dba332fdac9e758e1e366281adac6dd1af"
EXPECTED_ARMS = ("P_temporal_pair", "T_bounded_triplet", "D_triplet_branch")


def validate() -> dict:
    raw = PROTOCOL_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PROTOCOL_SHA256:
        raise ValueError("Frozen beyond-pair protocol changed")
    protocol = json.loads(raw)
    checks = {
        "schema": protocol.get("schema") == "sara-event-unit-beyond-pair-preregistration-v1",
        "arms": tuple(protocol.get("arms", {})) == EXPECTED_ARMS,
        "five_seeds": len(protocol["fresh_identity"]["seeds"]) == 5,
        "held_out_closed": protocol["fresh_identity"]["held_out_consumed"] is False,
        "one_attempt": protocol["resource_budgets"]["maximum_tuning_attempts"] == 1,
        "no_backward_events": protocol["resource_budgets"]["maximum_backward_events"] == 0,
        "prior_held_out_closed": protocol["boundaries"]["event_unit_v1_v2_held_out_used"] is False,
        "production_closed": protocol["boundaries"]["production_authorized"] is False,
    }
    if not all(checks.values()):
        raise ValueError("Beyond-pair protocol contract is invalid")
    report = {
        "schema": "sara-event-unit-beyond-pair-preregistration-validation-v1",
        "protocol_sha256": digest,
        "checks": checks,
        "passed": True,
        "candidate_implemented": False,
        "held_out_consumed": False,
        "production_authorized": False,
    }
    output = Path(ensure_parent_directory(workspace_path("evaluation", "event_unit_beyond_pair_preregistration_v1.json")))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"output": str(output), **report}


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
