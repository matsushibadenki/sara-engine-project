#!/usr/bin/env python3
"""Run the preregistered independent zero-shot relational replication."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sara_engine.evaluation.independent_relational_replication import (  # noqa: E402
    IndependentRelationalLearner,
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
        "benchmark_fixtures", "relational_rule_independent_replication_v2.json"
    )
)
EXPECTED_SHA256 = "b382a2fca20e8bf9a7eae010c88af8ec818048e64a15db222add30c780b41c5f"


def evaluator_label(context: str, values: tuple[int, ...]) -> int:
    """Independent oracle, deliberately separate from generator implementation."""
    if context == "ascending_or_equal":
        return int(values[1] >= values[0])
    if context == "same_parity":
        return int(values[0] % 2 == values[1] % 2)
    if context == "bounded_jump":
        return int(abs(values[1] - values[0]) <= 2)
    if context == "interval_direction":
        return int(values[2] - values[1] > values[1] - values[0])
    raise ValueError(f"Unknown context: {context}")


def main() -> int:
    raw = PROTOCOL.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Frozen v2 independent replication protocol changed")
    protocol = json.loads(raw)
    identity = protocol["fresh_identity"]
    common = {"seeds": identity["seeds"], "namespace": identity["namespace"]}
    training = generate_independent_episodes(
        values=identity["training_values"],
        count_per_context=identity["training_count_per_context_per_seed"],
        split="training",
        **common,
    )
    development = generate_independent_episodes(
        values=identity["development_values"],
        count_per_context=identity["development_count_per_context_per_seed"],
        split="development",
        **common,
    )
    agreement = all(
        episode.label == evaluator_label(episode.context, episode.values)
        for episode in training + development
    )
    unique_development = all(
        len({episode.values for episode in development if episode.context == context})
        == len([episode for episode in development if episode.context == context])
        for context in protocol["contexts"]
    )
    # Uniqueness is required within a seed/context, not across independent seeds.
    unique_per_seed_context = all(
        len(
            {
                episode.values
                for episode in development
                if f":{seed}:{context}:" in episode.identity
            }
        )
        == identity["development_count_per_context_per_seed"]
        for seed in identity["seeds"]
        for context in protocol["contexts"]
    )
    run = lambda arm, **kwargs: run_independent_arm(  # noqa: E731
        arm, training, development, **kwargs
    )
    arms = {arm: run(arm) for arm in IndependentRelationalLearner.ARMS}
    controls = {
        "context_shuffle": run(
            "contextual_relational_zero_shot", intervention="context_shuffle"
        ),
        "relation_shuffle": run(
            "contextual_relational_zero_shot", intervention="relation_shuffle"
        ),
        "context_relation_decouple": run(
            "contextual_relational_zero_shot",
            intervention="context_relation_decouple",
        ),
        "outcome_shuffle": run(
            "contextual_relational_zero_shot", outcome_shuffle_seed=933_991
        ),
        "capacity_matched_categorical": run(
            "categorical_zero_shot", reserve=12_000
        ),
    }
    categorical = arms["categorical_zero_shot"]
    relational = arms["contextual_relational_zero_shot"]
    deltas = {
        "relational_minus_categorical": relational["accuracy"]
        - categorical["accuracy"],
        "context_shuffle_drop": relational["accuracy"]
        - controls["context_shuffle"]["accuracy"],
        "relation_shuffle_drop": relational["accuracy"]
        - controls["relation_shuffle"]["accuracy"],
    }
    seed_gains = {}
    for seed in identity["seeds"]:
        seed_training = [
            episode for episode in training if f":training:{seed}:" in episode.identity
        ]
        seed_development = [
            episode
            for episode in development
            if f":development:{seed}:" in episode.identity
        ]
        seed_categorical = run_independent_arm(
            "categorical_zero_shot", seed_training, seed_development
        )["accuracy"]
        seed_relational = run_independent_arm(
            "contextual_relational_zero_shot", seed_training, seed_development
        )["accuracy"]
        seed_gains[str(seed)] = seed_relational - seed_categorical
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
        and all(gain > 0.0 for gain in seed_gains.values()),
    }
    replay = {arm: run(arm)["trace_sha256"] for arm in arms}
    budgets = protocol["resource_budgets"]
    harness = {
        "oracle_agreement": agreement,
        "development_unique_per_seed_context": unique_per_seed_context,
        "development_global_uniqueness_not_required": not unique_development,
        "zero_shot_no_development_updates": all(
            row["development_updates"] == 0 for row in arms.values()
        ),
        "exact_replay": all(replay[arm] == arms[arm]["trace_sha256"] for arm in arms),
        "event_work": all(row["maximum_event_work"] <= budgets["maximum_event_work"] for row in arms.values()),
        "state": all(row["state_bytes"] <= budgets["maximum_state_bytes"] for row in arms.values()),
        "features": all(row["feature_count"] <= budgets["maximum_features"] for row in arms.values()),
        "capacity_trace_match": controls["capacity_matched_categorical"]["trace_sha256"]
        == categorical["trace_sha256"],
    }
    report = {
        "schema": "sara-relational-rule-independent-replication-development-v2",
        "protocol_sha256": digest,
        "arms": arms,
        "controls": controls,
        "deltas": deltas,
        "seed_gains": seed_gains,
        "harness_checks": harness,
        "development_checks": checks,
        "harness_passed": all(harness.values()),
        "development_gate_passed": all(checks.values()),
        "held_out_consumed": False,
        "production_authorized": False,
    }
    output = Path(
        ensure_parent_directory(
            workspace_path(
                "evaluation",
                "relational_rule_independent_replication_v2_development.json",
            )
        )
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "harness": report["harness_passed"],
                "gate": report["development_gate_passed"],
                "deltas": deltas,
                "checks": checks,
            },
            sort_keys=True,
        )
    )
    return 0 if report["harness_passed"] and report["development_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
