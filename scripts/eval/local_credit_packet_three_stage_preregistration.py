#!/usr/bin/env python3
"""Validate the three-stage targeted-replay protocol before implementation."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

PROTOCOL = Path(
    processed_data_path(
        "benchmark_fixtures", "local_credit_packet_three_stage_v1.json"
    )
)
EXPECTED_SHA256 = "ea61d37c8cf55d8c125d9b1be7c359fd659b9002f50fe775b449cf7cf9daf709"


def validate() -> dict[str, object]:
    raw = PROTOCOL.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Frozen three-stage protocol changed")
    protocol = json.loads(raw)
    task = protocol["task"]
    anchor = protocol["episodic_anchor"]
    identity = protocol["identity"]
    leakage = protocol["leakage_boundaries"]
    budgets = protocol["resource_budgets"]
    checks = {
        "schema": protocol["schema"]
        == "sara-local-credit-packet-three-stage-preregistration-v1",
        "three_stage_depth": task["causal_depth"] == 3,
        "delays_exceed_direct_ttl": min(task["outcome_delay_steps"])
        > task["direct_eligibility_ttl"],
        "overlap_required": task["minimum_overlapping_eligibilities"] >= 4
        and task["maximum_outstanding_eligibilities"] >= 8,
        "five_arms": list(protocol["arms"]) == [
            "no_credit",
            "outcome_broadcast",
            "direct_packet",
            "packet_targeted_replay",
            "gradient_like_control",
        ],
        "bounded_targeted_anchor": anchor["maximum_entries"] <= 8
        and anchor["maximum_bytes_per_entry"] <= 48
        and anchor["targeted_lookup_only"]
        and not anchor["global_history_scan_allowed"],
        "anchor_no_label_key": anchor["lookup_key_may_encode_label"] is False,
        "required_controls": set(protocol["controls"])
        == {
            "replay_disabled",
            "anchor_route_shuffle",
            "anchor_context_shuffle",
            "packet_sign_shuffle",
            "outcome_shuffle",
            "anchor_expired",
            "causal_depth_two",
            "capacity_matched_direct_packet",
        },
        "five_seeds": len(identity["seeds"]) == 5,
        "heldout_closed": identity["heldout_materialized"] is False
        and identity["heldout_consumed"] is False,
        "zero_shot_development": leakage["development_updates"] is False,
        "no_gradient": leakage["global_loss_differentiation"] is False
        and leakage["cross_network_jacobian"] is False,
        "no_label_leakage": leakage["outcome_or_anchor_id_may_encode_label"]
        is False
        and leakage["route_digest_may_encode_label"] is False
        and leakage["future_activity_in_anchor"] is False,
        "bounded_resources": budgets["maximum_packet_bytes"] <= 64
        and budgets["maximum_anchor_lookups_per_outcome"] == 1
        and budgets["maximum_anchor_entries"] <= 8
        and budgets["maximum_tuning_attempts"] == 1,
        "candidate_absent_at_registration": protocol["boundaries"]["candidate_implemented"] is False,
        "production_closed": protocol["boundaries"]["production_authorized"] is False,
    }
    if not all(checks.values()):
        raise ValueError("Three-stage Local Credit Packet protocol invalid")
    report = {
        "schema": "sara-local-credit-packet-three-stage-validation-v1",
        "protocol_sha256": digest,
        "checks": checks,
        "passed": True,
        "candidate_implemented_now": (
            ROOT / "src" / "sara_engine" / "evaluation" / "local_credit_packet_three_stage.py"
        ).exists(),
        "development_materialized_now": Path(
            processed_data_path(
                "benchmark_fixtures", "local_credit_packet_three_stage_rows_v1.jsonl"
            )
        ).exists(),
        "heldout_consumed": False,
        "production_authorized": False,
    }
    output = Path(
        ensure_parent_directory(
            workspace_path(
                "evaluation", "local_credit_packet_three_stage_preregistration.json"
            )
        )
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"output": str(output), **report}


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
