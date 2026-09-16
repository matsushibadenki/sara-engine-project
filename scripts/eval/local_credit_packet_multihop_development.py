#!/usr/bin/env python3
"""Run the frozen two-circuit, one-edge-per-hop development gate."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sara_engine.evaluation.local_credit_packet_multihop import (  # noqa: E402
    ARMS,
    run_multihop_arm,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

PROTOCOL = Path(processed_data_path("benchmark_fixtures", "local_credit_packet_multihop_v1.json"))
SUPPLEMENT = Path(processed_data_path("benchmark_fixtures", "local_credit_packet_multihop_materialization_v1.json"))
ROWS = Path(processed_data_path("benchmark_fixtures", "local_credit_packet_multihop_rows_v1.jsonl"))
PROTOCOL_SHA256 = "be5a941959c7919efb236b16495e604d8822716fbeeeef8fcab315a42aa38fb2"
SUPPLEMENT_SHA256 = "ef3cf6f26122ca1964c7bf49a53ef3bc0d7c4430b8ec7689dd05cddca86f1bc5"
ROWS_SHA256 = "789b8a7755fecf4d73b1ee78191ae5c2296e632a59d1685a985131db4bc71b1b"


def _verify(path: Path, expected: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Frozen input changed: {path.name}")


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def main() -> int:
    _verify(PROTOCOL, PROTOCOL_SHA256)
    _verify(SUPPLEMENT, SUPPLEMENT_SHA256)
    _verify(ROWS, ROWS_SHA256)
    protocol = json.loads(PROTOCOL.read_text())
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    results = {}
    for seed in protocol["identity"]["seeds"]:
        calibration = [row for row in rows if row["seed"] == seed and row["phase"] == "B_calibration"]
        training = [row for row in rows if row["seed"] == seed and row["phase"] == "main_training"]
        development = [row for row in rows if row["seed"] == seed and row["phase"] == "development"]
        run = lambda arm, **kwargs: run_multihop_arm(  # noqa: E731
            arm, calibration, training, development, **kwargs
        )
        arms = {arm: run(arm) for arm in ARMS}
        B_targets = [int(row["B_local_target"]) for row in calibration + training]
        random.Random(seed + 606).shuffle(B_targets)
        outcomes = [int(row["global_success"]) for row in training]
        random.Random(seed + 707).shuffle(outcomes)
        controls = {
            name: run("local_credit_packet", intervention=name)
            for name in (
                "B_local_map_reset", "B_to_A_route_shuffle", "B_to_A_sign_shuffle",
                "A_eligibility_reset", "packet_delivery_disabled",
            )
        }
        controls["B_local_target_shuffle"] = run(
            "local_credit_packet", shuffled_B_targets=B_targets
        )
        controls["global_outcome_shuffle"] = run(
            "local_credit_packet", shuffled_global_outcomes=outcomes
        )
        controls["capacity_matched_direct_shortcut"] = run(
            "direct_anchor_shortcut", reserve=512
        )
        results[str(seed)] = {"arms": arms, "controls": controls}
    keys = (
        "accuracy", "A_accuracy", "B_local_accuracy", "A_updates", "B_local_updates",
        "packet_count", "maximum_packet_bytes", "maximum_backward_events_per_episode",
        "maximum_A_eligibility_entries", "A_feature_count", "B_feature_count", "state_bytes", "peak_state_bytes",
        "maximum_forward_event_work_per_episode",
    )
    arm_summary = {
        arm: {key: _mean([row["arms"][arm] for row in results.values()], key) for key in keys}
        for arm in ARMS
    }
    control_names = tuple(next(iter(results.values()))["controls"])
    control_summary = {
        name: {key: _mean([row["controls"][name] for row in results.values()], key) for key in keys}
        for name in control_names
    }
    packet = arm_summary["local_credit_packet"]["accuracy"]
    deltas = {
        "packet_minus_broadcast": packet - arm_summary["global_outcome_broadcast"]["accuracy"],
        "packet_minus_direct_shortcut": packet - arm_summary["direct_anchor_shortcut"]["accuracy"],
        "oracle_minus_packet": arm_summary["oracle_control"]["accuracy"] - packet,
        "B_map_reset_drop": packet - control_summary["B_local_map_reset"]["accuracy"],
        "B_local_target_shuffle_drop": packet - control_summary["B_local_target_shuffle"]["accuracy"],
        "route_shuffle_drop": packet - control_summary["B_to_A_route_shuffle"]["accuracy"],
        "sign_shuffle_drop": packet - control_summary["B_to_A_sign_shuffle"]["accuracy"],
    }
    acceptance = protocol["acceptance"]
    tolerance = 1e-12
    checks = {
        "packet_accuracy": packet >= acceptance["minimum_packet_global_accuracy"],
        "broadcast_gain": deltas["packet_minus_broadcast"] >= acceptance["minimum_gain_over_broadcast"] - tolerance,
        "shortcut_gain": deltas["packet_minus_direct_shortcut"] >= acceptance["minimum_gain_over_direct_shortcut"] - tolerance,
        "oracle_gap": deltas["oracle_minus_packet"] <= acceptance["maximum_gap_to_oracle"] + tolerance,
        "B_map_reset": deltas["B_map_reset_drop"] >= acceptance["minimum_B_map_reset_drop"] - tolerance,
        "B_local_shuffle": deltas["B_local_target_shuffle_drop"] >= acceptance["minimum_B_local_target_shuffle_drop"] - tolerance,
        "route_shuffle": deltas["route_shuffle_drop"] >= acceptance["minimum_route_shuffle_drop"] - tolerance,
        "sign_shuffle": deltas["sign_shuffle_drop"] >= acceptance["minimum_sign_shuffle_drop"] - tolerance,
        "five_seed_gains": len(results) == 5 and all(
            result["arms"]["local_credit_packet"]["accuracy"]
            > result["arms"]["direct_anchor_shortcut"]["accuracy"]
            for result in results.values()
        ),
    }
    budget = protocol["resource_budgets"]
    packet_budget = protocol["packet_contract"]
    all_arms = [row for result in results.values() for row in result["arms"].values()]
    harness = {
        "matched_forward_traces": all(
            len({row["forward_trace_sha256"] for row in result["arms"].values()}) == 1
            for result in results.values()
        ),
        "exact_replay": all(
            run_multihop_arm(
                "local_credit_packet",
                [row for row in rows if row["seed"] == seed and row["phase"] == "B_calibration"],
                [row for row in rows if row["seed"] == seed and row["phase"] == "main_training"],
                [row for row in rows if row["seed"] == seed and row["phase"] == "development"],
            )["prediction_trace_sha256"]
            == results[str(seed)]["arms"]["local_credit_packet"]["prediction_trace_sha256"]
            for seed in protocol["identity"]["seeds"]
        ),
        "zero_development_updates": all(row["development_updates"] == 0 for row in all_arms),
        "A_feature_budget": max(row["A_feature_count"] for row in all_arms) <= budget["maximum_A_features"],
        "B_feature_budget": max(row["B_feature_count"] for row in all_arms) <= budget["maximum_B_features"],
        "state_budget": max(row["peak_state_bytes"] for row in all_arms) <= budget["maximum_total_state_bytes"],
        "forward_work": max(row["maximum_forward_event_work_per_episode"] for row in all_arms) <= budget["maximum_forward_event_work_per_episode"],
        "packet_bytes": max(row["maximum_packet_bytes"] for row in all_arms) <= packet_budget["maximum_packet_bytes"],
        "two_backward_edges": max(row["maximum_backward_events_per_episode"] for row in all_arms) <= packet_budget["maximum_backward_events_per_episode"],
        "bounded_A_eligibility": max(row["maximum_A_eligibility_entries"] for row in all_arms) <= 1,
        "capacity_control_exact": all(
            result["controls"]["capacity_matched_direct_shortcut"]["prediction_trace_sha256"]
            == result["arms"]["direct_anchor_shortcut"]["prediction_trace_sha256"]
            for result in results.values()
        ),
        "capacity_control_state_allowance": all(
            result["controls"]["capacity_matched_direct_shortcut"]["peak_state_bytes"]
            >= result["arms"]["local_credit_packet"]["peak_state_bytes"]
            for result in results.values()
        ),
    }
    report = {
        "schema": "sara-local-credit-packet-multihop-development-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "materialization_sha256": SUPPLEMENT_SHA256,
        "rows_sha256": ROWS_SHA256,
        "arms": arm_summary,
        "controls": control_summary,
        "deltas": deltas,
        "per_seed": results,
        "development_checks": checks,
        "harness_checks": harness,
        "development_gate_passed": all(checks.values()),
        "harness_passed": all(harness.values()),
        "heldout_consumed": False,
        "production_authorized": False,
    }
    output = Path(ensure_parent_directory(workspace_path(
        "evaluation", "local_credit_packet_multihop_development.json"
    )))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "gate": report["development_gate_passed"],
                      "harness": report["harness_passed"], "deltas": deltas,
                      "checks": checks}, sort_keys=True))
    return 0 if report["development_gate_passed"] and report["harness_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
