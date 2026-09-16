#!/usr/bin/env python3
"""Run the frozen two-stage Local Credit Packet development gate."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sara_engine.evaluation.local_credit_packet import ARMS, run_credit_arm  # noqa: E402
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

PROTOCOL = Path(
    processed_data_path("benchmark_fixtures", "local_credit_packet_two_stage_v1.json")
)
MATERIALIZATION = Path(
    processed_data_path(
        "benchmark_fixtures", "local_credit_packet_two_stage_materialization_v1.json"
    )
)
ROWS = Path(
    processed_data_path(
        "benchmark_fixtures", "local_credit_packet_two_stage_rows_v1.jsonl"
    )
)
PROTOCOL_SHA256 = "d7cdfc6550fa1e1b2bf2441a401442a50ca2ebfb9e8df709f37bd24b4e0bfaa5"
MATERIALIZATION_SHA256 = "49b53490c85cfeb9328289e38b61efe084bb36070d40dce9dfa61c668bc8a0bd"
ROWS_SHA256 = "fc8b8395dcccf60989b8ef1c0758f2d42d422a468d3752b017f6cc422011b91b"


def main() -> int:
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol changed")
    if hashlib.sha256(MATERIALIZATION.read_bytes()).hexdigest() != MATERIALIZATION_SHA256:
        raise ValueError("Frozen materialization changed")
    if hashlib.sha256(ROWS.read_bytes()).hexdigest() != ROWS_SHA256:
        raise ValueError("Materialized rows changed")
    protocol = json.loads(PROTOCOL.read_text())
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    per_seed = {}
    aggregate = {arm: [] for arm in ARMS}
    control_rows = defaultdict_list = {
        name: []
        for name in (
            "packet_route_shuffle",
            "packet_sign_shuffle",
            "eligibility_reset_before_outcome",
            "packet_ttl_zero",
            "causal_depth_one",
            "replay_disabled",
            "outcome_shuffle",
            "capacity_matched_broadcast",
        )
    }
    for seed in protocol["identity"]["seeds"]:
        training = [row for row in rows if row["seed"] == seed and row["split"] == "training"]
        development = [row for row in rows if row["seed"] == seed and row["split"] == "development"]
        arms = {arm: run_credit_arm(arm, training, development) for arm in ARMS}
        outcomes = [int(row["scheduled_training_success"]) for row in training]
        random.Random(seed + 991).shuffle(outcomes)
        controls = {
            name: run_credit_arm(
                "local_credit_packet",
                training,
                development,
                intervention=name,
            )
            for name in (
                "packet_route_shuffle",
                "packet_sign_shuffle",
                "eligibility_reset_before_outcome",
                "packet_ttl_zero",
                "causal_depth_one",
                "replay_disabled",
            )
        }
        controls["outcome_shuffle"] = run_credit_arm(
            "local_credit_packet",
            training,
            development,
            outcome_override=outcomes,
        )
        controls["capacity_matched_broadcast"] = run_credit_arm(
            "outcome_broadcast", training, development, reserve=4096
        )
        for arm, result in arms.items():
            aggregate[arm].append(result)
        for name, result in controls.items():
            control_rows[name].append(result)
        per_seed[str(seed)] = {"arms": arms, "controls": controls}
    summarize = lambda values: {  # noqa: E731
        key: sum(float(row[key]) for row in values) / len(values)
        for key in (
            "accuracy",
            "updates",
            "packet_count",
            "backward_events",
            "maximum_packet_bytes",
            "maximum_eligibility_entries",
            "feature_count",
            "state_bytes",
            "maximum_forward_event_work_per_episode",
        )
    }
    arms_summary = {arm: summarize(values) for arm, values in aggregate.items()}
    controls_summary = {
        name: summarize(values) for name, values in control_rows.items()
    }
    packet_accuracy = arms_summary["local_credit_packet"]["accuracy"]
    deltas = {
        "packet_minus_no_credit": packet_accuracy - arms_summary["no_credit"]["accuracy"],
        "packet_minus_broadcast": packet_accuracy - arms_summary["outcome_broadcast"]["accuracy"],
        "gradient_like_minus_packet": arms_summary["gradient_like_control"]["accuracy"] - packet_accuracy,
        "route_shuffle_drop": packet_accuracy - controls_summary["packet_route_shuffle"]["accuracy"],
        "sign_shuffle_drop": packet_accuracy - controls_summary["packet_sign_shuffle"]["accuracy"],
        "eligibility_reset_drop": packet_accuracy - controls_summary["eligibility_reset_before_outcome"]["accuracy"],
        "ttl_zero_drop": packet_accuracy - controls_summary["packet_ttl_zero"]["accuracy"],
        "depth_one_drop": packet_accuracy - controls_summary["causal_depth_one"]["accuracy"],
    }
    acceptance = protocol["acceptance"]
    tolerance = 1e-12
    checks = {
        "packet_accuracy": packet_accuracy >= acceptance["minimum_packet_accuracy"],
        "gain_over_no_credit": deltas["packet_minus_no_credit"] >= acceptance["minimum_gain_over_no_credit"],
        "gain_over_broadcast": deltas["packet_minus_broadcast"] >= acceptance["minimum_gain_over_broadcast"],
        "near_gradient_like": deltas["gradient_like_minus_packet"] <= acceptance["maximum_gap_to_gradient_like_control"],
        "route_control": deltas["route_shuffle_drop"] >= acceptance["minimum_route_shuffle_drop"],
        "sign_control": deltas["sign_shuffle_drop"] >= acceptance["minimum_sign_shuffle_drop"],
        "eligibility_control": deltas["eligibility_reset_drop"] >= acceptance["minimum_eligibility_reset_drop"],
        "ttl_control": deltas["ttl_zero_drop"] >= acceptance["minimum_ttl_zero_drop"],
        "depth_control": deltas["depth_one_drop"] >= acceptance["minimum_depth_one_drop"],
        "all_five_seed_packet_gains_positive": all(
            result["arms"]["local_credit_packet"]["accuracy"]
            > result["arms"]["outcome_broadcast"]["accuracy"]
            for result in per_seed.values()
        ),
    }
    budgets = protocol["resource_budgets"]
    all_forward = [
        result["forward_trace_sha256"]
        for seed_result in per_seed.values()
        for result in seed_result["arms"].values()
    ]
    # Four equal traces per seed; seed identities intentionally differ.
    matched_forward = all(
        len({row["forward_trace_sha256"] for row in seed_result["arms"].values()}) == 1
        for seed_result in per_seed.values()
    )
    replays = all(
        run_credit_arm(
            "local_credit_packet",
            [row for row in rows if row["seed"] == seed and row["split"] == "training"],
            [row for row in rows if row["seed"] == seed and row["split"] == "development"],
        )["prediction_trace_sha256"]
        == per_seed[str(seed)]["arms"]["local_credit_packet"]["prediction_trace_sha256"]
        for seed in protocol["identity"]["seeds"]
    )
    harness = {
        "matched_forward_traces": matched_forward and len(all_forward) == 20,
        "exact_replay": replays,
        "zero_development_updates": all(
            result["development_updates"] == 0
            for seed_result in per_seed.values()
            for result in seed_result["arms"].values()
        ),
        "packet_size": max(row["maximum_packet_bytes"] for row in aggregate["local_credit_packet"])
        <= budgets["maximum_packet_bytes"],
        "bounded_eligibility": max(
            row["maximum_eligibility_entries"]
            for row in aggregate["local_credit_packet"]
        )
        <= 1,
        "backward_events": max(row["maximum_backward_events_per_episode"] for row in aggregate["local_credit_packet"])
        <= budgets["maximum_backward_events_per_episode"],
        "forward_work": max(row["maximum_forward_event_work_per_episode"] for row in aggregate["local_credit_packet"])
        <= budgets["maximum_forward_event_work_per_episode"],
        "state": max(row["state_bytes"] for row in aggregate["local_credit_packet"])
        <= budgets["maximum_total_state_bytes"],
        "features": max(row["feature_count"] for row in aggregate["local_credit_packet"])
        <= budgets["maximum_features"],
        "capacity_matched_broadcast_trace": all(
            seed_result["controls"]["capacity_matched_broadcast"]["prediction_trace_sha256"]
            == seed_result["arms"]["outcome_broadcast"]["prediction_trace_sha256"]
            for seed_result in per_seed.values()
        ),
    }
    report = {
        "schema": "sara-local-credit-packet-two-stage-development-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "materialization_sha256": MATERIALIZATION_SHA256,
        "rows_sha256": ROWS_SHA256,
        "arms": arms_summary,
        "controls": controls_summary,
        "deltas": deltas,
        "per_seed": per_seed,
        "development_checks": checks,
        "harness_checks": harness,
        "development_gate_passed": all(checks.values()),
        "harness_passed": all(harness.values()),
        "heldout_consumed": False,
        "production_authorized": False,
    }
    output = Path(
        ensure_parent_directory(
            workspace_path("evaluation", "local_credit_packet_two_stage_development.json")
        )
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "gate": report["development_gate_passed"], "harness": report["harness_passed"], "deltas": deltas}, sort_keys=True))
    return 0 if report["development_gate_passed"] and report["harness_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
