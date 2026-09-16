#!/usr/bin/env python3
"""Validate the immutable minimal event-unit causal-isolation protocol."""
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

PROTOCOL_PATH = Path(processed_data_path("benchmark_fixtures", "event_unit_causal_isolation_v1.json"))
PROTOCOL_SHA256 = "f8c2c2d2122af7ff4691e8738a9f8b490d94e8af029195737fe04fab459feac7"
EXPECTED_ARMS = (
    "A_scalar_local",
    "B_compact_event",
    "C_stateful_spiking",
    "D_dendritic_structural",
)


def validate() -> dict:
    raw = PROTOCOL_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PROTOCOL_SHA256:
        raise ValueError("Frozen event-unit causal-isolation protocol changed")
    protocol = json.loads(raw)
    checks = {
        "schema": protocol.get("schema") == "sara-event-unit-causal-isolation-preregistration-v1",
        "arm_order": tuple(protocol.get("arms", {})) == EXPECTED_ARMS,
        "five_seeds": len(protocol.get("splits", {}).get("seeds", ())) == 5,
        "held_out_families": protocol.get("splits", {}).get("held_out_causal_families", 0) > 0,
        "real_tests_sealed": protocol.get("splits", {}).get("frozen_real_event_tests_used_for_tuning") is False,
        "no_backward_events": protocol.get("resource_budgets", {}).get("maximum_backward_events") == 0,
        "single_tuning_attempt": protocol.get("resource_budgets", {}).get("maximum_tuning_attempts") == 1,
        "production_closed": protocol.get("production_authorized") is False,
    }
    if not all(checks.values()):
        raise ValueError("Event-unit causal-isolation protocol contract is invalid")
    report = {
        "schema": "sara-event-unit-causal-isolation-preregistration-validation-v1",
        "protocol_sha256": digest,
        "checks": checks,
        "passed": True,
        "candidate_implemented": False,
        "frozen_real_event_tests_opened": False,
        "production_authorized": False,
    }
    output = Path(ensure_parent_directory(workspace_path("evaluation", "event_unit_causal_isolation_preregistration_v1.json")))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"output": str(output), **report}


def main() -> int:
    result = validate()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
