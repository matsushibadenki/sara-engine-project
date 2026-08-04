#!/usr/bin/env python3
"""Build the immutable Phase 34 cache-separation follow-up draft."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.evaluation.phase34_memory_cache_preregistration import ARMS  # noqa: E402
from sara_engine.evaluation.phase34_separation_preregistration import (  # noqa: E402
    CASE_FAMILIES,
    EXPECTED_RELATIONS,
    EXPERIMENT_ID,
    PARENT_EXPERIMENT_ID,
    PARENT_PROTOCOL_FINGERPRINT,
    PARENT_REPORT_FINGERPRINT,
    REPLICATE_SEEDS,
    SCHEMA,
    build_registered_manifest,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

CASE_SCHEMA = "sara-phase34-memory-cache-separation-case-v1"
DEFAULT_FIXTURE = processed_data_path(
    "benchmark_fixtures", "phase34_memory_cache_separation_cases.jsonl"
)
DEFAULT_DRAFT = workspace_path(
    "evaluation", "phase34_memory_cache_separation_preregistration_draft.json"
)
DEFAULT_ENVIRONMENT = workspace_path(
    "evaluation", "phase34_memory_cache_separation_environment.json"
)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_fixture(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    validate_fixture(rows)
    return rows


def validate_fixture(rows: Sequence[Mapping[str, Any]]) -> None:
    errors: List[str] = []
    if tuple(row.get("family") for row in rows) != CASE_FAMILIES:
        errors.append("fixture_case_families_do_not_match_frozen_followup")
    case_ids = [row.get("case_id") for row in rows]
    if len(case_ids) != len(set(case_ids)) or any(
        not isinstance(case_id, str) or not case_id for case_id in case_ids
    ):
        errors.append("fixture_case_ids_must_be_unique")
    observed_relations = set()
    for row in rows:
        if row.get("schema") != CASE_SCHEMA:
            errors.append("unsupported_separation_case_schema")
        if row.get("observed_only") is not True:
            errors.append("separation_fixture_must_be_observed_only")
        horizon = row.get("horizon_events")
        if type(horizon) is not int or not 1 <= horizon <= 128:
            errors.append("invalid_separation_horizon")
        stream = row.get("checkpoint_stream")
        if not isinstance(stream, list) or not 9 <= len(stream) <= 16:
            errors.append("checkpoint_stream_must_force_capacity_pressure")
        elif len(stream) != len(set(stream)) or any(
            not isinstance(item, str) or not item for item in stream
        ):
            errors.append("checkpoint_stream_ids_must_be_unique")
        query_ids = row.get("query_ids")
        if not isinstance(query_ids, list) or not 1 <= len(query_ids) <= 8:
            errors.append("invalid_separation_query")
        relation = row.get("expected_relation")
        if relation not in EXPECTED_RELATIONS:
            errors.append("invalid_expected_relation")
        else:
            observed_relations.add(relation)
        if row.get("durable_mutation_allowed") is not False:
            errors.append("separation_fixture_cannot_mutate_durable_state")
    if observed_relations != set(EXPECTED_RELATIONS):
        errors.append("fixture_missing_expected_relation_control")
    if errors:
        raise ValueError("; ".join(sorted(set(errors))))


def environment_descriptor() -> Dict[str, Any]:
    return {
        "schema": "sara-phase34-memory-cache-separation-environment-v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "cpu_only": True,
        "gpu_required": False,
        "matrix_calculation": False,
        "backpropagation": False,
    }


def build_draft(
    rows: Sequence[Mapping[str, Any]], environment: Mapping[str, Any]
) -> Dict[str, Any]:
    validate_fixture(rows)
    draft = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": PARENT_EXPERIMENT_ID,
        "parent_protocol_fingerprint": PARENT_PROTOCOL_FINGERPRINT,
        "parent_report_fingerprint": PARENT_REPORT_FINGERPRINT,
        "registered_before_execution": True,
        "fixture_fingerprint": _digest(list(rows)),
        "environment_fingerprint": _digest(dict(environment)),
        "arms": list(ARMS),
        "case_families": list(CASE_FAMILIES),
        "replicate_seeds": list(REPLICATE_SEEDS),
        "replicates_per_condition": 5,
        "expected_relations": list(EXPECTED_RELATIONS),
        "budgets": {
            "source_events_per_case": 128,
            "attempted_checkpoints_per_case": 16,
            "max_checkpoints": 8,
            "max_selected_checkpoints": 2,
            "max_summary_ids_per_checkpoint": 8,
            "max_total_state_bytes": 8192,
            "max_local_interactions_per_case": 256,
            "max_latency_ms": 50,
            "max_merges_per_event": 2,
            "tuning_trials_per_arm": 1,
            "restart_count_per_arm": 0,
        },
        "resource_accounting": {
            "same_generated_stream_across_arms_and_replays": True,
            "same_seed_across_arms": True,
            "equal_source_events_across_arms": True,
            "equal_state_byte_ceiling_across_arms": True,
            "equal_tuning_allowance_across_arms": True,
            "hidden_dense_summary_allowed": False,
            "unbounded_checkpoint_scan_allowed": False,
        },
        "thresholds": {
            "pairwise_separation_rate": {"direction": "minimum", "limit": 0.5},
            "logarithmic_old_recall_delta": {"direction": "minimum", "limit": 0.1},
            "topk_pollution_precision_delta": {"direction": "minimum", "limit": 0.1},
            "equal_recent_resolution_delta": {"direction": "minimum", "limit": 0.1},
            "safety_integrity": {"direction": "minimum", "limit": 1.0},
            "state_bytes": {"direction": "maximum", "limit": 8192},
            "event_cost": {"direction": "maximum", "limit": 256},
            "latency_ms": {"direction": "maximum", "limit": 50},
            "deterministic_replay": {"direction": "minimum", "limit": 1.0},
        },
        "execution_policy": {
            "cpu_only": True,
            "gpu_required": False,
            "matrix_calculation": False,
            "backpropagation": False,
            "learned_router": False,
            "softmax": False,
            "checkpoint_parameter_averaging": False,
            "default_off": True,
            "production_mutation": False,
            "durable_admission": False,
            "physical_energy_claim": False,
            "independent_evidence_required": True,
        },
    }
    build_registered_manifest(draft, managed_path=True)
    return draft


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE)
    parser.add_argument("--draft-path", default=DEFAULT_DRAFT)
    parser.add_argument("--environment-path", default=DEFAULT_ENVIRONMENT)
    args = parser.parse_args(argv)
    rows = load_fixture(args.fixture_path)
    environment = environment_descriptor()
    draft = build_draft(rows, environment)
    for path, value in ((args.environment_path, environment), (args.draft_path, draft)):
        with open(ensure_parent_directory(path), "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    print(
        json.dumps(
            {
                "schema": "sara-phase34-memory-cache-separation-draft-receipt-v1",
                "case_count": len(rows),
                "replicate_count": len(REPLICATE_SEEDS),
                "fixture_fingerprint": draft["fixture_fingerprint"],
                "environment_fingerprint": draft["environment_fingerprint"],
                "draft_path": os.path.realpath(args.draft_path),
                "environment_path": os.path.realpath(args.environment_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
