#!/usr/bin/env python3
"""Run development-only smoke checks for the frozen event-unit protocol."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sara_engine.evaluation.event_unit_causal_isolation import EventUnitLearner, generate_episodes, run_development_arm
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path

PROTOCOL_PATH = Path(processed_data_path("benchmark_fixtures", "event_unit_causal_isolation_v1.json"))
PROTOCOL_SHA256 = "f8c2c2d2122af7ff4691e8738a9f8b490d94e8af029195737fe04fab459feac7"


def main() -> int:
    raw = PROTOCOL_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen event-unit protocol changed")
    protocol = json.loads(raw)
    seeds = protocol["splits"]["seeds"]
    training = generate_episodes(seeds=seeds, count_per_family=20, split="training")
    development = generate_episodes(seeds=seeds, count_per_family=10, split="development")
    arms = {
        arm: run_development_arm(arm, training, development)
        for arm in EventUnitLearner.ARM_NAMES
    }
    interventions = {
        "time_shuffle": run_development_arm("C_stateful_spiking", training, development, intervention="time_shuffle"),
        "state_reset": run_development_arm("C_stateful_spiking", training, development, intervention="state_reset"),
        "spike_count_preserving_shuffle": run_development_arm(
            "C_stateful_spiking", training, development, intervention="spike_count_preserving_shuffle"
        ),
        "refractory_disable": run_development_arm(
            "C_stateful_spiking", training, development, intervention="refractory_disable"
        ),
        "branch_assignment_shuffle": run_development_arm(
            "D_dendritic_structural", training, development, intervention="branch_assignment_shuffle"
        ),
        "outcome_shuffle": run_development_arm(
            "D_dendritic_structural", training, development, outcome_shuffle_seed=916_777
        ),
        "capacity_matched_scalar_state": run_development_arm(
            "A_scalar_local", training, development, capacity_reserve_bytes=26_240
        ),
    }
    replay = {
        arm: run_development_arm(arm, training, development)["prediction_trace_sha256"]
        for arm in EventUnitLearner.ARM_NAMES
    }
    replay_deterministic = all(
        replay[arm] == arms[arm]["prediction_trace_sha256"] for arm in EventUnitLearner.ARM_NAMES
    )
    per_seed = {}
    for seed in seeds:
        seed_training = generate_episodes(seeds=(seed,), count_per_family=20, split="training")
        seed_development = generate_episodes(seeds=(seed,), count_per_family=10, split="development")
        scores = {
            arm: run_development_arm(arm, seed_training, seed_development)["accuracy"]
            for arm in EventUnitLearner.ARM_NAMES
        }
        per_seed[str(seed)] = {
            "accuracy": scores,
            "ordered_gains": {
                "B_minus_A": scores["B_compact_event"] - scores["A_scalar_local"],
                "C_minus_B": scores["C_stateful_spiking"] - scores["B_compact_event"],
                "D_minus_C": scores["D_dendritic_structural"] - scores["C_stateful_spiking"],
            },
        }
    compact_equivalence = arms["A_scalar_local"]["prediction_rows"] == arms["B_compact_event"]["prediction_rows"]
    budgets = protocol["resource_budgets"]
    harness_checks = {
        "development_only": True,
        "real_frozen_tests_closed": True,
        "scalar_compact_equivalence": compact_equivalence,
        "deterministic_replay": replay_deterministic,
        "event_work": all(row["maximum_event_work"] <= budgets["maximum_event_work"] for row in arms.values()),
        "state": all(row["state_bytes"] <= budgets["maximum_retained_state_bytes"] for row in arms.values()),
        "features": all(row["feature_count"] <= budgets["maximum_learned_scalars"] for row in arms.values()),
        "routes": all(row["neuron_count"] <= budgets["maximum_routes"] for row in arms.values()),
    }
    causal_deltas = {
        "time": arms["C_stateful_spiking"]["accuracy"] - interventions["time_shuffle"]["accuracy"],
        "state": arms["C_stateful_spiking"]["accuracy"] - interventions["state_reset"]["accuracy"],
        "spike_association": arms["C_stateful_spiking"]["accuracy"] - interventions["spike_count_preserving_shuffle"]["accuracy"],
        "refractory": (
            arms["C_stateful_spiking"]["accuracy_by_family"]["refractory_suppression"]
            - interventions["refractory_disable"]["accuracy_by_family"]["refractory_suppression"]
        ),
        "branch_structure": (
            arms["D_dendritic_structural"]["accuracy_by_family"]["branch_specific_conjunction"]
            - interventions["branch_assignment_shuffle"]["accuracy_by_family"]["branch_specific_conjunction"]
        ),
        "outcome_signal": arms["D_dendritic_structural"]["accuracy"] - interventions["outcome_shuffle"]["accuracy"],
    }
    minimum_delta = protocol["acceptance"]["minimum_targeted_ablation_accuracy_drop"]
    development_checks = {
        "time": causal_deltas["time"] >= minimum_delta,
        "state": causal_deltas["state"] >= minimum_delta,
        "spike_association": causal_deltas["spike_association"] >= minimum_delta,
        "refractory": causal_deltas["refractory"] >= minimum_delta,
        "branch_structure": causal_deltas["branch_structure"] >= minimum_delta,
        "outcome_signal": causal_deltas["outcome_signal"] >= minimum_delta,
        "capacity_control_equivalence": (
            interventions["capacity_matched_scalar_state"]["prediction_trace_sha256"]
            == arms["A_scalar_local"]["prediction_trace_sha256"]
        ),
        "capacity_control_at_least_d_state": (
            interventions["capacity_matched_scalar_state"]["state_bytes"]
            >= arms["D_dendritic_structural"]["state_bytes"]
        ),
        "per_seed_ordered_signs": all(
            row["ordered_gains"]["B_minus_A"] == 0.0
            and row["ordered_gains"]["C_minus_B"] > 0.0
            and row["ordered_gains"]["D_minus_C"] > 0.0
            for row in per_seed.values()
        ),
    }
    for row in arms.values():
        del row["prediction_rows"]
    for row in interventions.values():
        del row["prediction_rows"]
    report = {
        "schema": "sara-event-unit-causal-isolation-development-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "runtime": {"python": sys.version, "executable": sys.executable},
        "training_episodes": len(training),
        "development_episodes": len(development),
        "arms": arms,
        "interventions": interventions,
        "per_seed": per_seed,
        "harness_checks": harness_checks,
        "causal_deltas": causal_deltas,
        "development_checks": development_checks,
        "harness_passed": all(harness_checks.values()),
        "development_gate_passed": all(development_checks.values()),
        "passed": all(harness_checks.values()) and all(development_checks.values()),
        "held_out_causal_families_consumed": False,
        "production_authorized": False,
    }
    output = Path(ensure_parent_directory(workspace_path("evaluation", "event_unit_causal_isolation_development_v1.json")))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "passed": report["passed"],
        "harness_checks": harness_checks,
        "development_checks": development_checks,
        "causal_deltas": causal_deltas,
        "arms": arms,
    }, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
