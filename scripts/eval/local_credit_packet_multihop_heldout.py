#!/usr/bin/env python3
"""Execute the frozen independent multi-hop held-out gate once."""
from __future__ import annotations

import hashlib
import json
import os
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

PROTOCOL = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_multihop_heldout_v1.json"
))
EXECUTION = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_multihop_heldout_execution_v1.json"
))
ROWS = Path(processed_data_path(
    "benchmark_fixtures", "local_credit_packet_multihop_heldout_rows_v1.jsonl"
))
MATERIALIZATION = Path(workspace_path(
    "evaluation", "local_credit_packet_multihop_heldout_materialization.json"
))
PROTOCOL_SHA256 = "5b542db726cf652c69158fa15cb007c3afa06ff615016a201bde9c809470402f"
EXECUTION_SHA256 = "ed4c4f541c46f4655d5908456b55277d82c694cbc2959523dcb434bd178e992d"
ROWS_SHA256 = "529923443cb4ff1b912493fbd96be1dd1ff3a2234c96ea12b10c8a8d0ec1cb4c"
OUTPUT = Path(workspace_path("evaluation", "local_credit_packet_multihop_heldout_result_v1.json"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: list[dict], key: str) -> float:
    return sum(float(value[key]) for value in values) / len(values)


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError("Held-out result already exists; execution is forbidden")
    if _sha(PROTOCOL) != PROTOCOL_SHA256 or _sha(EXECUTION) != EXECUTION_SHA256:
        raise ValueError("Frozen held-out protocol or execution plan changed")
    protocol = json.loads(PROTOCOL.read_text())
    plan = json.loads(EXECUTION.read_text())
    if plan["parent_protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("Execution plan parent mismatch")
    if _sha(ROWS) != ROWS_SHA256 or plan["materialized_rows_sha256"] != ROWS_SHA256:
        raise ValueError("Frozen held-out rows changed")
    materialization = json.loads(MATERIALIZATION.read_text())
    if not materialization["passed"] or materialization["rows_sha256"] != ROWS_SHA256:
        raise ValueError("Independent materialization audit did not pass")
    sources = protocol["frozen_sources"]
    source_digests = {
        name: _sha(ROOT / sources[f"{name}_path"])
        for name in ("candidate", "packet")
    }
    if any(source_digests[name] != sources[f"{name}_sha256"] for name in source_digests):
        raise ValueError("Frozen candidate source changed")
    if _sha(Path(workspace_path("evaluation", "local_credit_packet_multihop_development.json"))) != protocol["development_result_sha256"]:
        raise ValueError("Development evidence changed")
    output = Path(ensure_parent_directory(OUTPUT))
    # Exclusive creation is the durable one-shot marker, even if later scoring fails.
    with output.open("x+") as receipt:
        receipt.write(json.dumps({
            "schema": "sara-local-credit-packet-multihop-heldout-attempt-v1",
            "protocol_sha256": PROTOCOL_SHA256,
            "heldout_consumed": True,
            "status": "started",
        }, sort_keys=True) + "\n")
        receipt.flush()
        os.fsync(receipt.fileno())
        rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
        results = {}
        for seed in protocol["identity"]["seeds"]:
            calibration = [row for row in rows if row["seed"] == seed and row["phase"] == "B_calibration"]
            training = [row for row in rows if row["seed"] == seed and row["phase"] == "main_training"]
            heldout = [row for row in rows if row["seed"] == seed and row["phase"] == "heldout"]
            run = lambda arm, **kwargs: run_multihop_arm(  # noqa: E731
                arm, calibration, training, heldout, **kwargs
            )
            arms = {arm: run(arm) for arm in ARMS}
            B_targets = [int(row["B_local_target"]) for row in calibration + training]
            random.Random(seed + 1606).shuffle(B_targets)
            outcomes = [int(row["global_success"]) for row in training]
            random.Random(seed + 1707).shuffle(outcomes)
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
                "direct_anchor_shortcut",
                reserve=plan["capacity_matched_direct_shortcut_reserve_bytes"],
            )
            results[str(seed)] = {"arms": arms, "controls": controls}
        keys = (
            "accuracy", "A_accuracy", "B_local_accuracy", "A_updates",
            "B_local_updates", "packet_count", "maximum_packet_bytes",
            "maximum_backward_events_per_episode", "maximum_A_eligibility_entries",
            "A_feature_count", "B_feature_count", "peak_state_bytes",
            "maximum_forward_event_work_per_episode",
        )
        arms_summary = {
            arm: {key: _mean([result["arms"][arm] for result in results.values()], key) for key in keys}
            for arm in ARMS
        }
        control_names = tuple(next(iter(results.values()))["controls"])
        controls_summary = {
            name: {key: _mean([result["controls"][name] for result in results.values()], key) for key in keys}
            for name in control_names
        }
        packet_accuracy = arms_summary["local_credit_packet"]["accuracy"]
        deltas = {
            "packet_minus_broadcast": packet_accuracy - arms_summary["global_outcome_broadcast"]["accuracy"],
            "packet_minus_direct_shortcut": packet_accuracy - arms_summary["direct_anchor_shortcut"]["accuracy"],
            "oracle_minus_packet": arms_summary["oracle_control"]["accuracy"] - packet_accuracy,
            "B_map_reset_drop": packet_accuracy - controls_summary["B_local_map_reset"]["accuracy"],
            "B_local_target_shuffle_drop": packet_accuracy - controls_summary["B_local_target_shuffle"]["accuracy"],
            "route_shuffle_drop": packet_accuracy - controls_summary["B_to_A_route_shuffle"]["accuracy"],
            "sign_shuffle_drop": packet_accuracy - controls_summary["B_to_A_sign_shuffle"]["accuracy"],
        }
        acceptance = protocol["acceptance"]
        tolerance = 1e-12
        checks = {
            "packet_accuracy": packet_accuracy >= acceptance["minimum_packet_accuracy"],
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
        all_arms = [row for result in results.values() for row in result["arms"].values()]
        harness = {
            "source_frozen": all(source_digests[name] == sources[f"{name}_sha256"] for name in source_digests),
            "independent_oracle_agreement": materialization["checks"]["independent_oracle_agreement"],
            "matched_forward": all(
                len({row["forward_trace_sha256"] for row in result["arms"].values()}) == 1
                for result in results.values()
            ),
            "exact_replay": all(
                run_multihop_arm(
                    "local_credit_packet",
                    [row for row in rows if row["seed"] == seed and row["phase"] == "B_calibration"],
                    [row for row in rows if row["seed"] == seed and row["phase"] == "main_training"],
                    [row for row in rows if row["seed"] == seed and row["phase"] == "heldout"],
                )["prediction_trace_sha256"]
                == results[str(seed)]["arms"]["local_credit_packet"]["prediction_trace_sha256"]
                for seed in protocol["identity"]["seeds"]
            ),
            "zero_heldout_updates": all(row["development_updates"] == 0 for row in all_arms),
            "A_features": max(row["A_feature_count"] for row in all_arms) <= budget["maximum_A_features"],
            "B_features": max(row["B_feature_count"] for row in all_arms) <= budget["maximum_B_features"],
            "packet_bytes": max(row["maximum_packet_bytes"] for row in all_arms) <= budget["maximum_packet_bytes"],
            "backward_events": max(row["maximum_backward_events_per_episode"] for row in all_arms) <= budget["maximum_backward_events_per_episode"],
            "A_eligibility": max(row["maximum_A_eligibility_entries"] for row in all_arms) <= budget["maximum_A_eligibility_entries"],
            "peak_state": max(row["peak_state_bytes"] for row in all_arms) <= budget["maximum_peak_state_bytes"],
            "forward_work": max(row["maximum_forward_event_work_per_episode"] for row in all_arms) <= budget["maximum_forward_event_work_per_episode"],
            "capacity_prediction_equivalence": all(
                result["controls"]["capacity_matched_direct_shortcut"]["prediction_trace_sha256"]
                == result["arms"]["direct_anchor_shortcut"]["prediction_trace_sha256"]
                for result in results.values()
            ),
            "capacity_state_allowance": all(
                result["controls"]["capacity_matched_direct_shortcut"]["peak_state_bytes"]
                >= result["arms"]["local_credit_packet"]["peak_state_bytes"]
                for result in results.values()
            ),
        }
        report = {
            "schema": "sara-local-credit-packet-multihop-heldout-result-v1",
            "protocol_sha256": PROTOCOL_SHA256,
            "execution_sha256": EXECUTION_SHA256,
            "rows_sha256": ROWS_SHA256,
            "source_sha256": source_digests,
            "arms": arms_summary,
            "controls": controls_summary,
            "deltas": deltas,
            "per_seed": results,
            "acceptance_checks": checks,
            "harness_checks": harness,
            "heldout_gate_passed": all(checks.values()),
            "harness_passed": all(harness.values()),
            "heldout_consumed": True,
            "execution_count": 1,
            "production_authorized": False,
        }
        receipt.seek(0)
        receipt.truncate()
        receipt.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        receipt.flush()
        os.fsync(receipt.fileno())
    print(json.dumps({"output": str(output), "gate": report["heldout_gate_passed"],
                      "harness": report["harness_passed"], "deltas": deltas,
                      "checks": checks}, sort_keys=True))
    return 0 if report["heldout_gate_passed"] and report["harness_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
