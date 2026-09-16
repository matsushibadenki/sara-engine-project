#!/usr/bin/env python3
"""Run development controls for event-unit causal isolation v2."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sara_engine.evaluation.event_unit_causal_isolation import EventUnitV2Learner, generate_episodes, run_v2_development_arm
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path

PROTOCOL_PATH = Path(processed_data_path("benchmark_fixtures", "event_unit_causal_isolation_v2.json"))
PROTOCOL_SHA256 = "07afe1d0bd4389c1d71acecc30e360335ade36daa380579c68e8137d865a5494"


def _episodes(seeds, count, split):
    return generate_episodes(
        seeds=seeds, count_per_family=count, split=split, namespace="event-unit-v2"
    )


def main() -> int:
    raw = PROTOCOL_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen event-unit v2 protocol changed")
    protocol = json.loads(raw)
    identity = protocol["fresh_identity"]
    seeds = identity["seeds"]
    training = _episodes(seeds, identity["training_count_per_family_per_seed"], "training")
    development = _episodes(seeds, identity["development_count_per_family_per_seed"], "development")
    arms = {
        arm: run_v2_development_arm(arm, training, development)
        for arm in EventUnitV2Learner.ARM_NAMES
    }
    controls = {
        "time_shuffle_on_C": run_v2_development_arm(
            "C_temporal_state", training, development, intervention="time_shuffle"
        ),
        "temporal_state_reset_on_C": run_v2_development_arm(
            "C_temporal_state", training, development, intervention="state_reset"
        ),
        "refractory_disable_on_R": run_v2_development_arm(
            "R_temporal_refractory", training, development, intervention="refractory_disable"
        ),
        "branch_assignment_shuffle_on_D": run_v2_development_arm(
            "D_temporal_dendritic", training, development, intervention="branch_assignment_shuffle"
        ),
        "outcome_shuffle_on_D": run_v2_development_arm(
            "D_temporal_dendritic", training, development, outcome_shuffle_seed=917_991
        ),
        "capacity_matched_compact_event": run_v2_development_arm(
            "B_compact_event", training, development, capacity_reserve_bytes=26_240
        ),
    }
    replay = {
        arm: run_v2_development_arm(arm, training, development)["prediction_trace_sha256"]
        for arm in EventUnitV2Learner.ARM_NAMES
    }
    budgets = protocol["resource_budgets"]
    harness_checks = {
        "fresh_namespace": all(row.identity.startswith("event-unit-v2:") for row in training + development),
        "held_out_closed": True,
        "deterministic_replay": all(replay[arm] == arms[arm]["prediction_trace_sha256"] for arm in replay),
        "event_work": all(row["maximum_event_work"] <= budgets["maximum_event_work"] for row in arms.values()),
        "state": all(row["state_bytes"] <= budgets["maximum_retained_state_bytes"] for row in arms.values()),
        "features": all(row["feature_count"] <= budgets["maximum_learned_scalars"] for row in arms.values()),
    }
    per_seed = {}
    for seed in seeds:
        train = _episodes((seed,), identity["training_count_per_family_per_seed"], "training")
        dev = _episodes((seed,), identity["development_count_per_family_per_seed"], "development")
        scores = {arm: run_v2_development_arm(arm, train, dev)["accuracy"] for arm in EventUnitV2Learner.ARM_NAMES}
        per_seed[str(seed)] = {
            "accuracy": scores,
            "contrasts": {
                "C_minus_B": scores["C_temporal_state"] - scores["B_compact_event"],
                "R_minus_C": scores["R_temporal_refractory"] - scores["C_temporal_state"],
                "D_minus_C": scores["D_temporal_dendritic"] - scores["C_temporal_state"],
            },
        }
    minimum = protocol["acceptance"]["minimum_targeted_accuracy_delta"]
    deltas = {
        "temporal_state": arms["C_temporal_state"]["accuracy"] - arms["B_compact_event"]["accuracy"],
        "refractory": arms["R_temporal_refractory"]["accuracy"] - arms["C_temporal_state"]["accuracy"],
        "branch_structure": arms["D_temporal_dendritic"]["accuracy"] - arms["C_temporal_state"]["accuracy"],
        "time_control": arms["C_temporal_state"]["accuracy"] - controls["time_shuffle_on_C"]["accuracy"],
        "state_control": arms["C_temporal_state"]["accuracy"] - controls["temporal_state_reset_on_C"]["accuracy"],
        "refractory_control": (
            arms["R_temporal_refractory"]["accuracy"] - controls["refractory_disable_on_R"]["accuracy"]
        ),
        "branch_control": (
            arms["D_temporal_dendritic"]["accuracy_by_family"]["branch_specific_conjunction"]
            - controls["branch_assignment_shuffle_on_D"]["accuracy_by_family"]["branch_specific_conjunction"]
        ),
    }
    development_checks = {
        "temporal_state": deltas["temporal_state"] >= minimum,
        "temporal_time_control": deltas["time_control"] >= minimum,
        "temporal_state_control": deltas["state_control"] >= minimum,
        "refractory": deltas["refractory"] >= minimum,
        "refractory_control": deltas["refractory_control"] >= minimum,
        "branch_structure": deltas["branch_structure"] >= minimum,
        "branch_control": deltas["branch_control"] >= minimum,
        "capacity_equivalence": (
            controls["capacity_matched_compact_event"]["prediction_trace_sha256"]
            == arms["B_compact_event"]["prediction_trace_sha256"]
        ),
        "per_seed_signs": all(
            row["contrasts"]["C_minus_B"] > 0
            and row["contrasts"]["D_minus_C"] > 0
            and row["contrasts"]["R_minus_C"] == 0
            for row in per_seed.values()
        ),
    }
    for group in (arms, controls):
        for row in group.values():
            del row["prediction_rows"]
    report = {
        "schema": "sara-event-unit-causal-isolation-v2-development-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "runtime": {"python": sys.version, "executable": sys.executable},
        "arms": arms,
        "controls": controls,
        "per_seed": per_seed,
        "deltas": deltas,
        "harness_checks": harness_checks,
        "development_checks": development_checks,
        "harness_passed": all(harness_checks.values()),
        "development_gate_passed": all(development_checks.values()),
        "held_out_consumed": False,
        "production_authorized": False,
    }
    output = Path(ensure_parent_directory(workspace_path("evaluation", "event_unit_causal_isolation_v2_development.json")))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "harness_passed": report["harness_passed"], "development_gate_passed": report["development_gate_passed"], "deltas": deltas, "development_checks": development_checks}, sort_keys=True))
    return 0 if report["harness_passed"] and report["development_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
