#!/usr/bin/env python3
"""Validate the frozen multi-hop credit protocol before task materialization."""
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
    processed_data_path("benchmark_fixtures", "local_credit_packet_multihop_v1.json")
)
EXPECTED_SHA256 = "be5a941959c7919efb236b16495e604d8822716fbeeeef8fcab315a42aa38fb2"
PARENT = Path(workspace_path("evaluation", "local_credit_packet_three_stage_development.json"))


def validate() -> dict[str, object]:
    raw = PROTOCOL.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Frozen multi-hop protocol changed")
    protocol = json.loads(raw)
    task = protocol["task"]
    schedule = protocol["training_schedule"]
    packet = protocol["packet_contract"]
    identity = protocol["identity"]
    budget = protocol["resource_budgets"]
    parent_digest = hashlib.sha256(PARENT.read_bytes()).hexdigest()
    checks = {
        "schema": protocol["schema"]
        == "sara-local-credit-packet-multihop-preregistration-v1",
        "parent_frozen": parent_digest == protocol["parent_development_result_sha256"],
        "two_private_circuits": task["A_private_cue_count"] == 8
        and task["B_private_cue_count"] == 8
        and task["A_cannot_observe_B_cue_or_B_local_target"]
        and task["B_cannot_observe_A_cue_or_A_target"],
        "two_backward_edges": packet["maximum_causal_depth"] == 2
        and protocol["task"]["backward_edges"]
        == ["outcome_to_circuit_B", "circuit_B_to_circuit_A"],
        "both_circuits_trainable": schedule["main_training_updates_B_from_local_target"]
        and schedule["main_training_updates_A_only_from_received_credit"],
        "forced_forward_match": schedule["forced_A_and_B_actions_in_main_training"]
        and schedule["same_forward_rows_and_forced_actions_for_all_arms"],
        "zero_shot_development": schedule["development_updates"] is False,
        "one_edge_packets": packet["fanout_per_hop"] == 1
        and packet["maximum_backward_events_per_episode"] == 2
        and packet["direct_outcome_to_A_allowed"] is False,
        "packet_no_private_target": packet["B_private_cue_or_local_target_in_packet_allowed"]
        is False,
        "no_gradient_or_global_scan": packet["numerical_gradient_allowed"] is False
        and packet["global_history_scan_allowed"] is False,
        "five_arms": list(protocol["arms"])
        == [
            "no_A_credit", "global_outcome_broadcast", "direct_anchor_shortcut",
            "local_credit_packet", "oracle_control",
        ],
        "shortcuts_controlled": set(protocol["controls"])
        == {
            "B_local_map_reset", "B_local_target_shuffle", "B_to_A_route_shuffle",
            "B_to_A_sign_shuffle", "A_eligibility_reset", "packet_delivery_disabled",
            "global_outcome_shuffle", "capacity_matched_direct_shortcut",
        },
        "five_fresh_seeds": len(identity["seeds"]) == 5
        and len(set(identity["seeds"])) == 5,
        "heldout_closed": identity["heldout_materialized"] is False
        and identity["heldout_consumed"] is False,
        "bounded_resources": packet["maximum_packet_bytes"] <= 64
        and budget["maximum_A_features"] <= 16
        and budget["maximum_B_features"] <= 16
        and budget["maximum_tuning_attempts"] == 1,
        "auxiliary_supervision_disclosed": task["B_local_target_is_auxiliary_supervision"]
        and protocol["boundaries"][
            "B_auxiliary_supervision_limits_generalization_claim"
        ],
        "production_closed": protocol["boundaries"]["production_authorized"] is False,
    }
    if not all(checks.values()):
        raise ValueError("Multi-hop credit preregistration invalid")
    report = {
        "schema": "sara-local-credit-packet-multihop-validation-v1",
        "protocol_sha256": digest,
        "parent_result_sha256": parent_digest,
        "checks": checks,
        "passed": True,
        "candidate_implemented_now": (
            ROOT / "src" / "sara_engine" / "evaluation" / "local_credit_packet_multihop.py"
        ).exists(),
        "heldout_consumed": False,
        "production_authorized": False,
    }
    output = Path(
        ensure_parent_directory(
            workspace_path("evaluation", "local_credit_packet_multihop_preregistration.json")
        )
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"output": str(output), **report}


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
