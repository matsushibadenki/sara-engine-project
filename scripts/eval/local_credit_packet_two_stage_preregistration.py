#!/usr/bin/env python3
"""Validate the Local Credit Packet protocol before candidate implementation."""
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
        "benchmark_fixtures", "local_credit_packet_two_stage_v1.json"
    )
)
EXPECTED_SHA256 = "d7cdfc6550fa1e1b2bf2441a401442a50ca2ebfb9e8df709f37bd24b4e0bfaa5"
EXPECTED_FIELDS = [
    "source_event",
    "outcome_id",
    "sign",
    "magnitude_bucket",
    "age",
    "causal_depth",
    "confidence",
]


def validate() -> dict[str, object]:
    raw = PROTOCOL.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Frozen Local Credit Packet protocol changed")
    protocol = json.loads(raw)
    packet = protocol["packet_schema"]
    fairness = protocol["fairness"]
    leakage = protocol["leakage_boundaries"]
    identity = protocol["identity"]
    budgets = protocol["resource_budgets"]
    checks = {
        "schema": protocol["schema"]
        == "sara-local-credit-packet-two-stage-preregistration-v1",
        "two_stage_depth": protocol["task"]["causal_depth"] == 2,
        "packet_fields_exact": packet["fields"] == EXPECTED_FIELDS,
        "no_numerical_gradient": packet["numerical_gradient_allowed"] is False,
        "sparse_delivery": packet["inactive_branch_delivery_allowed"] is False
        and packet["packet_fanout"] == 1,
        "four_arms": list(protocol["arms"]) == [
            "no_credit",
            "outcome_broadcast",
            "gradient_like_control",
            "local_credit_packet",
        ],
        "fair_forward_path": all(
            fairness[key]
            for key in (
                "same_forward_routes",
                "same_initial_state",
                "same_training_episodes",
                "same_prediction_budget",
                "same_delayed_outcome",
                "same_maximum_update_opportunities",
            )
        ),
        "oracle_not_adoptable": fairness[
            "gradient_like_control_not_eligible_for_adoption"
        ],
        "five_seeds": len(identity["seeds"]) == 5,
        "heldout_closed": identity["heldout_materialized"] is False
        and identity["heldout_consumed"] is False,
        "zero_shot_development": leakage["development_updates"] is False,
        "no_label_address_leakage": all(
            leakage[key] is False
            for key in (
                "outcome_id_may_encode_label",
                "source_event_may_encode_label",
                "route_address_may_encode_label",
                "future_activity_in_eligibility",
            )
        ),
        "no_global_differentiation": leakage["global_loss_differentiation"] is False
        and leakage["cross_network_jacobian"] is False,
        "bounded_packets": budgets["maximum_packet_bytes"] <= 64
        and budgets["maximum_packets_per_outcome"] <= 2
        and budgets["maximum_backward_events_per_episode"] <= 2,
        "one_attempt": budgets["maximum_tuning_attempts"] == 1,
        "required_controls": set(protocol["controls"])
        == {
            "packet_route_shuffle",
            "packet_sign_shuffle",
            "eligibility_reset_before_outcome",
            "packet_ttl_zero",
            "causal_depth_one",
            "replay_disabled",
            "outcome_shuffle",
            "capacity_matched_broadcast",
        },
        "production_closed": protocol["boundaries"]["production_authorized"] is False,
    }
    if not all(checks.values()):
        raise ValueError("Local Credit Packet protocol invalid")
    report = {
        "schema": "sara-local-credit-packet-two-stage-validation-v1",
        "protocol_sha256": digest,
        "checks": checks,
        "passed": True,
        "candidate_implemented": False,
        "development_materialized": False,
        "heldout_consumed": False,
        "production_authorized": False,
    }
    output = Path(
        ensure_parent_directory(
            workspace_path(
                "evaluation", "local_credit_packet_two_stage_preregistration.json"
            )
        )
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"output": str(output), **report}


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
