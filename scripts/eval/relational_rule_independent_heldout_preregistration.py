#!/usr/bin/env python3
"""Validate held-out relational replication without materializing its rows."""
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
        "benchmark_fixtures", "relational_rule_independent_heldout_v1.json"
    )
)
EXPECTED_SHA256 = "d79c3463fe74debc85d92348944a7425303d3aa9a221a5954684f97cc4c05e51"
PRIOR_VALUES = {
    101, 103, 107, 109, 113, 127, 211, 223, 227, 229, 233, 239,
    130, 131, 134, 135, 138, 139, 250, 251, 254, 255, 258, 259,
}


def label(context: str, values: tuple[int, ...]) -> int:
    if context == "ascending_or_equal":
        return int(values[1] >= values[0])
    if context == "same_parity":
        return int(values[0] % 2 == values[1] % 2)
    if context == "bounded_jump":
        return int(abs(values[1] - values[0]) <= 2)
    if context == "interval_direction":
        return int(values[2] - values[1] > values[1] - values[0])
    raise ValueError(f"Unknown context: {context}")


def validate() -> dict[str, object]:
    raw = PROTOCOL.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Frozen held-out protocol changed")
    protocol = json.loads(raw)
    identity = protocol["fresh_identity"]
    candidate = protocol["frozen_candidate"]
    all_values = set(identity["training_values"]) | set(identity["heldout_values"])
    source_digest = hashlib.sha256((ROOT / candidate["module"]).read_bytes()).hexdigest()
    support = {}
    for split, key in (("training", "training_values"), ("heldout", "heldout_values")):
        values = identity[key]
        support[split] = {}
        for context in protocol["contexts"]:
            width = 3 if context == "interval_direction" else 2
            counts = {0: 0, 1: 0}
            for row in itertools.product(values, repeat=width):
                counts[label(context, row)] += 1
            support[split][context] = counts
    checks = {
        "schema": protocol["schema"]
        == "sara-relational-rule-independent-heldout-preregistration-v1",
        "candidate_source_frozen": source_digest == candidate["sha256"],
        "fresh_values": not all_values & PRIOR_VALUES,
        "train_heldout_disjoint": not set(identity["training_values"])
        & set(identity["heldout_values"]),
        "mixed_parity": all(
            {value % 2 for value in identity[key]} == {0, 1}
            for key in ("training_values", "heldout_values")
        ),
        "both_labels_reachable": all(
            all(count > 0 for count in counts.values())
            for split in support.values()
            for counts in split.values()
        ),
        "heldout_unique_capacity": all(
            min(counts.values()) >= identity["heldout_count_per_context_per_seed"] // 2
            for counts in support["heldout"].values()
        ),
        "one_shot": protocol["evaluation_boundary"]["maximum_executions"] == 1,
        "zero_shot": protocol["evaluation_boundary"]["heldout_updates"] is False,
        "heldout_closed": identity["heldout_materialized"] is False
        and identity["heldout_consumed"] is False,
        "composition_closed": protocol["boundaries"]["composition_allowed"] is False,
        "credit_packet_closed": protocol["boundaries"]["local_credit_packet_allowed"] is False,
        "production_closed": protocol["boundaries"]["production_authorized"] is False,
    }
    if not all(checks.values()):
        raise ValueError("Held-out relational preregistration invalid")
    report = {
        "schema": "sara-relational-rule-independent-heldout-validation-v1",
        "protocol_sha256": digest,
        "candidate_sha256": source_digest,
        "support": support,
        "checks": checks,
        "passed": True,
        "heldout_materialized": False,
        "heldout_consumed": False,
        "execution_authorized": True,
        "production_authorized": False,
    }
    output = Path(
        ensure_parent_directory(
            workspace_path(
                "evaluation", "relational_rule_independent_heldout_preregistration.json"
            )
        )
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"output": str(output), **report}


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
