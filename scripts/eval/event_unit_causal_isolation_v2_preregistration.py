#!/usr/bin/env python3
"""Validate the immutable event-unit v2 causal-isolation protocol."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path

PROTOCOL_PATH = Path(processed_data_path("benchmark_fixtures", "event_unit_causal_isolation_v2.json"))
PROTOCOL_SHA256 = "07afe1d0bd4389c1d71acecc30e360335ade36daa380579c68e8137d865a5494"
EXPECTED_ARMS = (
    "B_compact_event",
    "C_temporal_state",
    "R_temporal_refractory",
    "D_temporal_dendritic",
)


def validate() -> dict:
    raw = PROTOCOL_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PROTOCOL_SHA256:
        raise ValueError("Frozen event-unit v2 protocol changed")
    protocol = json.loads(raw)
    identity = protocol["fresh_identity"]
    boundaries = protocol["boundaries"]
    checks = {
        "schema": protocol.get("schema") == "sara-event-unit-causal-isolation-preregistration-v2",
        "arm_order": tuple(protocol.get("arms", {})) == EXPECTED_ARMS,
        "five_fresh_seeds": len(identity["seeds"]) == 5 and identity["v1_seed_overlap"] == 0,
        "three_contrasts": set(protocol["ordered_contrasts"]) == {"temporal_state", "refractory", "branch_structure"},
        "threshold_independence": protocol["acceptance"]["thresholds_selected_from_v1_scores"] is False,
        "one_attempt": protocol["resource_budgets"]["maximum_tuning_attempts"] == 1,
        "no_backward_events": protocol["resource_budgets"]["maximum_backward_events"] == 0,
        "held_out_closed": identity["held_out_consumed"] is False,
        "v1_held_out_closed": boundaries["v1_held_out_split_must_remain_closed"] is True,
        "production_closed": boundaries["production_authorized"] is False,
    }
    if not all(checks.values()):
        raise ValueError("Event-unit v2 preregistration contract is invalid")
    report = {
        "schema": "sara-event-unit-causal-isolation-v2-preregistration-validation-v1",
        "protocol_sha256": digest,
        "checks": checks,
        "passed": True,
        "candidate_implemented": False,
        "held_out_consumed": False,
        "production_authorized": False,
    }
    output = Path(ensure_parent_directory(workspace_path("evaluation", "event_unit_causal_isolation_v2_preregistration.json")))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"output": str(output), **report}


if __name__ == "__main__":
    result = validate()
    print(json.dumps(result, sort_keys=True))
