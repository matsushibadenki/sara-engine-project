#!/usr/bin/env python3
"""Run the frozen three-stage targeted-replay development comparison."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sara_engine.evaluation.local_credit_packet_three_stage import (  # noqa: E402
    ARMS,
    run_three_stage_arm,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

PROTOCOL = Path(processed_data_path("benchmark_fixtures", "local_credit_packet_three_stage_v1.json"))
SUPPLEMENT = Path(processed_data_path("benchmark_fixtures", "local_credit_packet_three_stage_materialization_v1.json"))
ROWS = Path(processed_data_path("benchmark_fixtures", "local_credit_packet_three_stage_rows_v1.jsonl"))
PROTOCOL_SHA256 = "ea61d37c8cf55d8c125d9b1be7c359fd659b9002f50fe775b449cf7cf9daf709"
SUPPLEMENT_SHA256 = "afbe5cb3eb5c2703bab3fb7ea17dce8b0f0bec9ef0934c0f8dd4fbc3767218df"
ROWS_SHA256 = "dcde1cc4c641b0e09fa7351d90f13a277ceb3f62ed12d9e15b4c755ef6dc481a"


def _verify(path: Path, digest: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Frozen input changed: {path.name}")


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def main() -> int:
    _verify(PROTOCOL, PROTOCOL_SHA256)
    _verify(SUPPLEMENT, SUPPLEMENT_SHA256)
    _verify(ROWS, ROWS_SHA256)
    protocol = json.loads(PROTOCOL.read_text())
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    namespace = protocol["identity"]["namespace"]
    seeds = protocol["identity"]["seeds"]
    results = {}
    for seed in seeds:
        training = [row for row in rows if row["seed"] == seed and row["split"] == "training"]
        development = [row for row in rows if row["seed"] == seed and row["split"] == "development"]
        run = lambda arm, **kwargs: run_three_stage_arm(  # noqa: E731
            arm, training, development, namespace=namespace, **kwargs
        )
        arms = {arm: run(arm) for arm in ARMS}
        shuffled = [int(row["outcome_success"]) for row in training]
        random.Random(seed + 73).shuffle(shuffled)
        controls = {
            name: run("packet_targeted_replay", intervention=name)
            for name in (
                "replay_disabled", "anchor_route_shuffle", "anchor_context_shuffle",
                "packet_sign_shuffle", "anchor_expired", "causal_depth_two",
            )
        }
        controls["outcome_shuffle"] = run(
            "packet_targeted_replay", shuffled_outcomes=shuffled
        )
        # The direct arm retains the same bounded anchor state but cannot replay it.
        controls["capacity_matched_direct_packet"] = run("direct_packet")
        results[str(seed)] = {"arms": arms, "controls": controls}
    arm_summary = {
        arm: {key: _mean([result["arms"][arm] for result in results.values()], key)
              for key in ("accuracy", "updates", "backward_events", "anchor_lookups",
                          "successful_replays", "expired_direct_count", "peak_direct_entries",
                          "peak_anchor_entries", "maximum_packet_bytes", "maximum_anchor_bytes",
                          "feature_count", "state_bytes", "peak_state_bytes")}
        for arm in ARMS
    }
    control_names = tuple(next(iter(results.values()))["controls"])
    control_summary = {
        name: {key: _mean([result["controls"][name] for result in results.values()], key)
               for key in ("accuracy", "updates", "backward_events", "anchor_lookups",
                           "successful_replays", "peak_anchor_entries", "state_bytes")}
        for name in control_names
    }
    replay_accuracy = arm_summary["packet_targeted_replay"]["accuracy"]
    deltas = {
        "replay_minus_broadcast": replay_accuracy - arm_summary["outcome_broadcast"]["accuracy"],
        "replay_minus_direct": replay_accuracy - arm_summary["direct_packet"]["accuracy"],
        "gradient_like_minus_replay": arm_summary["gradient_like_control"]["accuracy"] - replay_accuracy,
        "replay_disabled_drop": replay_accuracy - control_summary["replay_disabled"]["accuracy"],
        "anchor_route_shuffle_drop": replay_accuracy - control_summary["anchor_route_shuffle"]["accuracy"],
        "anchor_context_shuffle_drop": replay_accuracy - control_summary["anchor_context_shuffle"]["accuracy"],
        "anchor_expiry_drop": replay_accuracy - control_summary["anchor_expired"]["accuracy"],
        "depth_two_drop": replay_accuracy - control_summary["causal_depth_two"]["accuracy"],
    }
    acceptance = protocol["acceptance"]
    tolerance = 1e-12
    gates = {
        "replay_accuracy": replay_accuracy >= acceptance["minimum_packet_replay_accuracy"],
        "broadcast_gain": deltas["replay_minus_broadcast"] >= acceptance["minimum_gain_over_broadcast"] - tolerance,
        "direct_gain": deltas["replay_minus_direct"] >= acceptance["minimum_gain_over_direct_packet"] - tolerance,
        "oracle_gap": deltas["gradient_like_minus_replay"] <= acceptance["maximum_gap_to_gradient_like_control"] + tolerance,
        "replay_disabled": deltas["replay_disabled_drop"] >= acceptance["minimum_replay_disabled_drop"] - tolerance,
        "route_shuffle": deltas["anchor_route_shuffle_drop"] >= acceptance["minimum_anchor_route_shuffle_drop"] - tolerance,
        "context_shuffle": deltas["anchor_context_shuffle_drop"] >= acceptance["minimum_anchor_context_shuffle_drop"] - tolerance,
        "anchor_expiry": deltas["anchor_expiry_drop"] >= acceptance["minimum_anchor_expiry_drop"] - tolerance,
        "depth_two": deltas["depth_two_drop"] >= acceptance["minimum_depth_two_drop"] - tolerance,
        "all_five_seed_gains": len(results) == 5 and all(
            result["arms"]["packet_targeted_replay"]["accuracy"]
            > result["arms"]["direct_packet"]["accuracy"]
            for result in results.values()
        ),
    }
    budgets = protocol["resource_budgets"]
    all_arms = [row for result in results.values() for row in result["arms"].values()]
    forward_match = all(
        len({row["forward_trace_sha256"] for row in result["arms"].values()}) == 1
        for result in results.values()
    )
    replay_exact = all(
        run_three_stage_arm(
            "packet_targeted_replay",
            [row for row in rows if row["seed"] == seed and row["split"] == "training"],
            [row for row in rows if row["seed"] == seed and row["split"] == "development"],
            namespace=namespace,
        )["prediction_trace_sha256"]
        == results[str(seed)]["arms"]["packet_targeted_replay"]["prediction_trace_sha256"]
        for seed in seeds
    )
    harness = {
        "matched_forward": forward_match,
        "exact_replay": replay_exact,
        "zero_development_updates": all(row["development_updates"] == 0 for row in all_arms),
        "all_direct_expired": all(row["expired_direct_count"] == protocol["identity"]["training_episode_count_per_seed"] for row in all_arms),
        "overlap": all(row["peak_direct_entries"] == 8 and row["peak_anchor_entries"] == 8 for row in all_arms),
        "packet_bytes": max(row["maximum_packet_bytes"] for row in all_arms) <= budgets["maximum_packet_bytes"],
        "backward_events": max(row["maximum_backward_events_per_episode"] for row in all_arms) <= budgets["maximum_backward_events_per_episode"],
        "anchor_lookups": max(row["maximum_anchor_lookups_per_outcome"] for row in all_arms) <= budgets["maximum_anchor_lookups_per_outcome"],
        "anchor_entries": max(row["peak_anchor_entries"] for row in all_arms) <= budgets["maximum_anchor_entries"],
        "state": max(row["peak_state_bytes"] for row in all_arms) <= budgets["maximum_total_state_bytes"],
        "anchor_bytes": max(row["maximum_anchor_bytes"] for row in all_arms) <= protocol["episodic_anchor"]["maximum_bytes_per_entry"],
        "forward_work": max(row["maximum_forward_event_work_per_episode"] for row in all_arms) <= budgets["maximum_forward_event_work_per_episode"],
        "capacity_control": all(
            result["controls"]["capacity_matched_direct_packet"]["prediction_trace_sha256"]
            == result["arms"]["direct_packet"]["prediction_trace_sha256"]
            for result in results.values()
        ),
    }
    report = {
        "schema": "sara-local-credit-packet-three-stage-development-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "materialization_sha256": SUPPLEMENT_SHA256,
        "rows_sha256": ROWS_SHA256,
        "arms": arm_summary,
        "controls": control_summary,
        "deltas": deltas,
        "per_seed": results,
        "development_checks": gates,
        "harness_checks": harness,
        "development_gate_passed": all(gates.values()),
        "harness_passed": all(harness.values()),
        "heldout_consumed": False,
        "production_authorized": False,
    }
    output = Path(ensure_parent_directory(workspace_path("evaluation", "local_credit_packet_three_stage_development.json")))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "gate": report["development_gate_passed"],
                      "harness": report["harness_passed"], "deltas": deltas,
                      "checks": gates}, sort_keys=True))
    return 0 if report["development_gate_passed"] and report["harness_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
