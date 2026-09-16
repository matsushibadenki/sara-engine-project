#!/usr/bin/env python3
"""Validate the frozen v2 independent-replication protocol."""
from __future__ import annotations

import hashlib
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


def validate() -> dict[str, object]:
    raw = PROTOCOL.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Frozen v2 independent replication protocol changed")
    protocol = json.loads(raw)
    generator = protocol["generator_independence"]
    identity = protocol["fresh_identity"]
    evaluation = protocol["evaluation_boundary"]
    boundaries = protocol["boundaries"]
    checks = {
        "schema": protocol["schema"]
        == "sara-relational-rule-independent-replication-preregistration-v2",
        "supersedes_v1": protocol["supersedes_protocol"]
        == "relational-rule-independent-replication-v1-2026-09-16",
        "two_arms": protocol["arms"]
        == ["categorical_zero_shot", "contextual_relational_zero_shot"],
        "new_generator": generator["new_generator_required"]
        and not generator["prior_transition_generator_import_allowed"],
        "separate_evaluator": generator[
            "label_evaluator_implemented_separately_from_candidate"
        ],
        "five_fresh_seeds": len(identity["seeds"]) == 5
        and not set(identity["seeds"])
        & {922027, 922141, 922253, 922369, 922481},
        "mixed_parity_splits": all(
            {value % 2 for value in identity[key]} == {0, 1}
            for key in ("training_values", "development_values")
        ),
        "values_disjoint": not set(identity["training_values"])
        & set(identity["development_values"])
        and identity["value_overlap"] == 0,
        "zero_shot": evaluation["development_updates"] is False
        and evaluation["score_before_any_adaptation"],
        "unique_pairs": evaluation["development_value_pairs_unique_within_context"],
        "held_out_closed": identity["held_out_consumed"] is False,
        "composition_non_gating": boundaries["composition_is_non_gating"],
        "candidate_gated": boundaries[
            "candidate_implementation_authorized_only_after_feasibility_pass"
        ],
        "no_backward": protocol["resource_budgets"]["maximum_backward_events"] == 0,
        "one_attempt": protocol["resource_budgets"]["maximum_tuning_attempts"] == 1,
        "production_closed": boundaries["production_authorized"] is False,
    }
    if not all(checks.values()):
        raise ValueError("Independent replication v2 protocol invalid")
    report = {
        "schema": "sara-relational-rule-independent-replication-preregistration-validation-v2",
        "protocol_sha256": digest,
        "checks": checks,
        "passed": True,
        "generator_implemented": False,
        "audit_executed": False,
        "held_out_consumed": False,
        "production_authorized": False,
    }
    output = Path(
        ensure_parent_directory(
            workspace_path(
                "evaluation",
                "relational_rule_independent_replication_v2_preregistration.json",
            )
        )
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"output": str(output), **report}


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
