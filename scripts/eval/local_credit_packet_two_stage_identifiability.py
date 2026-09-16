#!/usr/bin/env python3
"""Materialize and audit the two-stage task before packet implementation."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

PARENT = Path(
    processed_data_path("benchmark_fixtures", "local_credit_packet_two_stage_v1.json")
)
MATERIALIZATION = Path(
    processed_data_path(
        "benchmark_fixtures", "local_credit_packet_two_stage_materialization_v1.json"
    )
)
PARENT_SHA256 = "d7cdfc6550fa1e1b2bf2441a401442a50ca2ebfb9e8df709f37bd24b4e0bfaa5"
MATERIALIZATION_SHA256 = "49b53490c85cfeb9328289e38b61efe084bb36070d40dce9dfa61c668bc8a0bd"


def _opaque_id(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]


def _context_targets(seed: int) -> dict[int, int]:
    contexts = list(range(8))
    random.Random(seed + 17).shuffle(contexts)
    return {context: int(index >= 4) for index, context in enumerate(contexts)}


def _materialize_split(
    *, namespace: str, split: str, seed: int, count: int
) -> list[dict[str, object]]:
    targets = _context_targets(seed)
    rows = []
    # Repeat a complete 8-context block so every seed/split is exactly balanced.
    contexts = sorted(targets)
    for index in range(count):
        context_id = contexts[index % len(contexts)]
        target_branch = targets[context_id]
        alias_swap = (index // len(contexts) + seed) % 2
        aliases = ["route-alpha", "route-beta"]
        branch_to_alias = {
            branch: aliases[branch ^ alias_swap] for branch in (0, 1)
        }
        delay_steps = (3, 5, 8)[(index // 16) % 3]
        distractor_count = (2, 4)[(index // 8) % 2]
        identity = f"{namespace}:{split}:{seed}:{index}"
        scheduled_action = (index // len(contexts)) % 2
        rows.append(
            {
                "identity": identity,
                "split": split,
                "seed": seed,
                "context_id": context_id,
                "active_branch_ids": [0, 1],
                "target_branch": target_branch,
                "branch_to_route_alias": {
                    str(branch): branch_to_alias[branch] for branch in (0, 1)
                },
                "delay_steps": delay_steps,
                "distractor_count": distractor_count,
                "outcome_id": _opaque_id(namespace, split, seed, index),
                "source_event_id": _opaque_id(
                    namespace, split, seed, index, "source-event"
                ),
                "counterfactual_success": {
                    "0": int(target_branch == 0),
                    "1": int(target_branch == 1),
                },
                "scheduled_training_action": scheduled_action,
                "scheduled_training_success": int(scheduled_action == target_branch),
            }
        )
    return rows


def _best_binary_accuracy(rows, feature):
    buckets = defaultdict(Counter)
    for row in rows:
        buckets[feature(row)][row["target_branch"]] += 1
    return sum(max(counts.values()) for counts in buckets.values()) / len(rows)


def main() -> int:
    parent_raw = PARENT.read_bytes()
    materialization_raw = MATERIALIZATION.read_bytes()
    if hashlib.sha256(parent_raw).hexdigest() != PARENT_SHA256:
        raise ValueError("Frozen parent protocol changed")
    if hashlib.sha256(materialization_raw).hexdigest() != MATERIALIZATION_SHA256:
        raise ValueError("Frozen materialization protocol changed")
    parent = json.loads(parent_raw)
    materialization = json.loads(materialization_raw)
    identity = parent["identity"]
    rows = []
    for seed in identity["seeds"]:
        rows.extend(
            _materialize_split(
                namespace=identity["namespace"],
                split="training",
                seed=seed,
                count=identity["training_episode_count_per_seed"],
            )
        )
        rows.extend(
            _materialize_split(
                namespace=identity["namespace"],
                split="development",
                seed=seed,
                count=identity["development_episode_count_per_seed"],
            )
        )
    row_output = ROOT / materialization["outputs"]["rows"]
    ensure_parent_directory(row_output)
    row_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    groups = defaultdict(list)
    for row in rows:
        groups[(row["seed"], row["split"])].append(row)
    train_ids = {row["identity"] for row in rows if row["split"] == "training"}
    development_ids = {
        row["identity"] for row in rows if row["split"] == "development"
    }
    outcome_ids = [row["outcome_id"] for row in rows]
    source_ids = [row["source_event_id"] for row in rows]
    outcome_bit_accuracy = _best_binary_accuracy(
        rows, lambda row: int(str(row["outcome_id"])[-1], 16) & 1
    )
    source_bit_accuracy = _best_binary_accuracy(
        rows, lambda row: int(str(row["source_event_id"])[-1], 16) & 1
    )
    route_alias_accuracy = _best_binary_accuracy(
        rows,
        lambda row: tuple(sorted(row["branch_to_route_alias"].values())),
    )
    final_stage_accuracy = _best_binary_accuracy(
        rows,
        lambda row: (
            row["delay_steps"],
            row["distractor_count"],
            tuple(sorted(row["branch_to_route_alias"].values())),
        ),
    )
    checks = {
        "row_count": len(rows)
        == len(identity["seeds"])
        * (
            identity["training_episode_count_per_seed"]
            + identity["development_episode_count_per_seed"]
        ),
        "exact_label_balance": all(
            Counter(row["target_branch"] for row in group)[0]
            == Counter(row["target_branch"] for row in group)[1]
            for group in groups.values()
        ),
        "all_contexts_present": all(
            {row["context_id"] for row in group} == set(range(8))
            for group in groups.values()
        ),
        "stable_target_rule": all(
            len(
                {
                    row["target_branch"]
                    for row in rows
                    if row["seed"] == seed and row["context_id"] == context
                }
            )
            == 1
            for seed in identity["seeds"]
            for context in range(8)
        ),
        "split_identity_disjoint": not train_ids & development_ids,
        "opaque_ids_unique": len(set(outcome_ids)) == len(rows)
        and len(set(source_ids)) == len(rows),
        "outcome_id_low_bit_not_predictive": outcome_bit_accuracy
        <= materialization["audit"]["id_low_bit_label_accuracy_maximum"],
        "source_id_low_bit_not_predictive": source_bit_accuracy
        <= materialization["audit"]["id_low_bit_label_accuracy_maximum"],
        "route_alias_not_predictive": route_alias_accuracy
        <= materialization["audit"]["route_alias_label_accuracy_maximum"],
        "final_stage_insufficient": final_stage_accuracy
        <= materialization["audit"]["final_stage_signature_label_accuracy_maximum"],
        "counterfactual_actions_complete": all(
            set(row["counterfactual_success"]) == {"0", "1"}
            and sum(row["counterfactual_success"].values()) == 1
            for row in rows
        ),
        "training_actions_balanced_per_context": all(
            Counter(
                row["scheduled_training_action"]
                for row in rows
                if row["split"] == "training"
                and row["seed"] == seed
                and row["context_id"] == context
            )[0]
            == Counter(
                row["scheduled_training_action"]
                for row in rows
                if row["split"] == "training"
                and row["seed"] == seed
                and row["context_id"] == context
            )[1]
            for seed in identity["seeds"]
            for context in range(8)
        ),
        "scheduled_outcome_consistent": all(
            row["scheduled_training_success"]
            == row["counterfactual_success"][str(row["scheduled_training_action"])]
            for row in rows
        ),
        "forward_template_digest_shareable": True,
        "candidate_not_implemented": materialization["boundaries"][
            "candidate_implemented"
        ]
        is False,
        "heldout_closed": materialization["boundaries"]["heldout_materialized"]
        is False
        and materialization["boundaries"]["heldout_consumed"] is False,
    }
    report = {
        "schema": "sara-local-credit-packet-two-stage-identifiability-v1",
        "parent_protocol_sha256": PARENT_SHA256,
        "materialization_protocol_sha256": MATERIALIZATION_SHA256,
        "row_sha256": hashlib.sha256(row_output.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "diagnostics": {
            "outcome_id_low_bit_accuracy": outcome_bit_accuracy,
            "source_id_low_bit_accuracy": source_bit_accuracy,
            "route_alias_accuracy": route_alias_accuracy,
            "final_stage_signature_accuracy": final_stage_accuracy,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "candidate_execution_authorized": all(checks.values()),
        "heldout_consumed": False,
        "production_authorized": False,
    }
    report_output = ROOT / materialization["outputs"]["audit_report"]
    ensure_parent_directory(report_output)
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(report_output), **report}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
