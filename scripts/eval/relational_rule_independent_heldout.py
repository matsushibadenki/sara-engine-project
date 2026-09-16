#!/usr/bin/env python3
"""Execute the frozen independent relational held-out gate exactly once."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sara_engine.evaluation.independent_relational_replication import (  # noqa: E402
    generate_independent_episodes,
    run_independent_arm,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

PROTOCOL = Path(
    processed_data_path(
        "benchmark_fixtures", "relational_rule_independent_heldout_v1.json"
    )
)
PROTOCOL_SHA256 = "d79c3463fe74debc85d92348944a7425303d3aa9a221a5954684f97cc4c05e51"
OUTPUT = Path(
    workspace_path("evaluation", "relational_rule_independent_heldout_v1.json")
)


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError("Held-out result already exists; repeated execution is forbidden")
    raw = PROTOCOL.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen held-out protocol changed")
    protocol = json.loads(raw)
    candidate = protocol["frozen_candidate"]
    candidate_digest = hashlib.sha256((ROOT / candidate["module"]).read_bytes()).hexdigest()
    if candidate_digest != candidate["sha256"]:
        raise ValueError("Frozen candidate source changed")
    identity = protocol["fresh_identity"]
    common = {"seeds": identity["seeds"], "namespace": identity["namespace"]}
    training = generate_independent_episodes(
        values=identity["training_values"],
        count_per_context=identity["training_count_per_context_per_seed"],
        split="training",
        **common,
    )
    heldout = generate_independent_episodes(
        values=identity["heldout_values"],
        count_per_context=identity["heldout_count_per_context_per_seed"],
        split="development",
        **common,
    )
    run = lambda arm, **kwargs: run_independent_arm(  # noqa: E731
        arm, training, heldout, **kwargs
    )
    categorical = run("categorical_zero_shot")
    relational = run("contextual_relational_zero_shot")
    controls = {
        "context_shuffle": run(
            "contextual_relational_zero_shot", intervention="context_shuffle"
        ),
        "relation_shuffle": run(
            "contextual_relational_zero_shot", intervention="relation_shuffle"
        ),
        "capacity_matched_categorical": run("categorical_zero_shot", reserve=12_000),
    }
    replay = {
        "categorical_zero_shot": run("categorical_zero_shot")["trace_sha256"],
        "contextual_relational_zero_shot": run("contextual_relational_zero_shot")[
            "trace_sha256"
        ],
    }
    seed_gains = {}
    for seed in identity["seeds"]:
        seed_training = [row for row in training if f":training:{seed}:" in row.identity]
        seed_heldout = [row for row in heldout if f":development:{seed}:" in row.identity]
        left = run_independent_arm("categorical_zero_shot", seed_training, seed_heldout)
        right = run_independent_arm(
            "contextual_relational_zero_shot", seed_training, seed_heldout
        )
        seed_gains[str(seed)] = right["accuracy"] - left["accuracy"]
    deltas = {
        "relational_minus_categorical": relational["accuracy"] - categorical["accuracy"],
        "context_shuffle_drop": relational["accuracy"]
        - controls["context_shuffle"]["accuracy"],
        "relation_shuffle_drop": relational["accuracy"]
        - controls["relation_shuffle"]["accuracy"],
    }
    acceptance = protocol["acceptance"]
    tolerance = 1e-12
    checks = {
        "categorical_ceiling": categorical["accuracy"]
        <= acceptance["maximum_categorical_accuracy"],
        "relational_accuracy": relational["accuracy"]
        >= acceptance["minimum_relational_accuracy"],
        "gain": deltas["relational_minus_categorical"]
        >= acceptance["minimum_gain_over_categorical"] - tolerance,
        "context_control": deltas["context_shuffle_drop"]
        >= acceptance["minimum_context_shuffle_drop"] - tolerance,
        "relation_control": deltas["relation_shuffle_drop"]
        >= acceptance["minimum_relation_shuffle_drop"] - tolerance,
        "all_five_seed_gain_signs_positive": len(seed_gains) == 5
        and all(gain > 0 for gain in seed_gains.values()),
    }
    budgets = protocol["resource_budgets"]
    arms = {
        "categorical_zero_shot": categorical,
        "contextual_relational_zero_shot": relational,
    }
    harness = {
        "candidate_source_frozen": candidate_digest == candidate["sha256"],
        "zero_shot_no_heldout_updates": all(
            row["development_updates"] == 0 for row in arms.values()
        ),
        "exact_replay": replay["categorical_zero_shot"] == categorical["trace_sha256"]
        and replay["contextual_relational_zero_shot"] == relational["trace_sha256"],
        "capacity_trace_match": controls["capacity_matched_categorical"]["trace_sha256"]
        == categorical["trace_sha256"],
        "event_work": all(
            row["maximum_event_work"] <= budgets["maximum_event_work"]
            for row in arms.values()
        ),
        "state": all(
            row["state_bytes"] <= budgets["maximum_state_bytes"]
            for row in arms.values()
        ),
        "features": all(
            row["feature_count"] <= budgets["maximum_features"]
            for row in arms.values()
        ),
    }
    report = {
        "schema": "sara-relational-rule-independent-heldout-result-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "candidate_sha256": candidate_digest,
        "arms": arms,
        "controls": controls,
        "deltas": deltas,
        "seed_gains": seed_gains,
        "harness_checks": harness,
        "acceptance_checks": checks,
        "harness_passed": all(harness.values()),
        "heldout_gate_passed": all(checks.values()),
        "heldout_consumed": True,
        "execution_count": 1,
        "production_authorized": False,
    }
    Path(ensure_parent_directory(OUTPUT)).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "harness": report["harness_passed"],
                "gate": report["heldout_gate_passed"],
                "deltas": deltas,
                "seed_gains": seed_gains,
            },
            sort_keys=True,
        )
    )
    return 0 if report["harness_passed"] and report["heldout_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
