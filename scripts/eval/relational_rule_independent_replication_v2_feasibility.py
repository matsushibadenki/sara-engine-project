#!/usr/bin/env python3
"""Exhaustively audit v2 label support before generator implementation."""
from __future__ import annotations

import hashlib
import itertools
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
        "benchmark_fixtures", "relational_rule_independent_replication_v2.json"
    )
)
EXPECTED_SHA256 = "b382a2fca20e8bf9a7eae010c88af8ec818048e64a15db222add30c780b41c5f"


def independent_label(context: str, values: tuple[int, ...]) -> int:
    if context == "ascending_or_equal":
        return int(values[1] >= values[0])
    if context == "same_parity":
        return int((values[0] & 1) == (values[1] & 1))
    if context == "bounded_jump":
        return int(abs(values[1] - values[0]) <= 2)
    if context == "interval_direction":
        return int((values[2] - values[1]) > (values[1] - values[0]))
    raise ValueError(f"Unknown context: {context}")


def audit() -> dict[str, object]:
    raw = PROTOCOL.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Frozen v2 independent replication protocol changed")
    protocol = json.loads(raw)
    identity = protocol["fresh_identity"]
    support: dict[str, dict[str, object]] = {}
    for split, key in (
        ("training", "training_values"),
        ("development", "development_values"),
    ):
        values = identity[key]
        support[split] = {}
        for context in protocol["contexts"]:
            width = 3 if context == "interval_direction" else 2
            counts = {0: 0, 1: 0}
            for row in itertools.product(values, repeat=width):
                counts[independent_label(context, row)] += 1
            support[split][context] = {
                "label_counts": {str(label): count for label, count in counts.items()},
                "both_labels_reachable": all(count > 0 for count in counts.values()),
                "balanced_sampling_capacity": min(counts.values()),
            }
    # Training rows may sample observable signatures repeatedly. Development rows
    # are unique within a context, so only that split needs enough distinct
    # signatures to fill half of its balanced row budget.
    minimum_required = {
        "training": 1,
        "development": identity["development_count_per_context_per_seed"] // 2,
    }
    checks = {
        "all_contexts_have_both_labels": all(
            row["both_labels_reachable"]
            for split in support.values()
            for row in split.values()
        ),
        "balanced_sampling_capacity_sufficient": all(
            row["balanced_sampling_capacity"] >= minimum_required[split_name]
            for split_name, split in support.items()
            for row in split.values()
        ),
        "mixed_parity_splits": all(
            {value % 2 for value in identity[key]} == {0, 1}
            for key in ("training_values", "development_values")
        ),
        "value_sets_disjoint": not set(identity["training_values"])
        & set(identity["development_values"]),
        "candidate_not_implemented": True,
        "held_out_closed": identity["held_out_consumed"] is False,
    }
    passed = all(checks.values())
    report = {
        "schema": "sara-relational-rule-independent-replication-feasibility-v2",
        "protocol_sha256": digest,
        "support": support,
        "minimum_distinct_examples_per_label": minimum_required,
        "checks": checks,
        "passed": passed,
        "candidate_execution_authorized": passed,
        "held_out_consumed": False,
        "production_authorized": False,
    }
    output = Path(
        ensure_parent_directory(
            workspace_path(
                "evaluation",
                "relational_rule_independent_replication_v2_feasibility.json",
            )
        )
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"output": str(output), **report}


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)
