#!/usr/bin/env python3
"""Build the immutable Phase 34 retention-by-selection factorial draft."""

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

from sara_engine.evaluation.phase34_factorial_preregistration import (  # noqa: E402
    ARMS,
    CASE_FAMILIES,
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

CASE_SCHEMA = "sara-phase34-memory-cache-factorial-case-v1"
DEFAULT_FIXTURE = processed_data_path(
    "benchmark_fixtures", "phase34_memory_cache_factorial_cases.jsonl"
)
DEFAULT_DRAFT = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_preregistration_draft.json"
)
DEFAULT_ENVIRONMENT = workspace_path(
    "evaluation", "phase34_memory_cache_factorial_environment.json"
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
        errors.append("fixture_case_families_do_not_match_frozen_factorial")
    case_ids = [row.get("case_id") for row in rows]
    if len(case_ids) != len(set(case_ids)) or any(
        not isinstance(case_id, str) or not case_id for case_id in case_ids
    ):
        errors.append("fixture_case_ids_must_be_unique")
    observed_focus = set()
    for row in rows:
        if row.get("schema") != CASE_SCHEMA:
            errors.append("unsupported_factorial_case_schema")
        if row.get("observed_only") is not True:
            errors.append("factorial_fixture_must_be_observed_only")
        focus = row.get("factor_focus")
        if focus not in {"selection", "retention", "safety"}:
            errors.append("invalid_factor_focus")
        else:
            observed_focus.add(focus)
        horizon = row.get("horizon_events")
        if type(horizon) is not int or not 1 <= horizon <= 128:
            errors.append("invalid_factorial_horizon")
        stream = row.get("checkpoint_stream")
        if not isinstance(stream, list) or not 8 <= len(stream) <= 16:
            errors.append("invalid_factorial_checkpoint_stream")
            continue
        if len(stream) != len(set(stream)) or any(
            not isinstance(item, str) or not item for item in stream
        ):
            errors.append("factorial_checkpoint_ids_must_be_unique")
        query_ids = row.get("query_ids")
        if not isinstance(query_ids, list) or not 1 <= len(query_ids) <= 8:
            errors.append("invalid_factorial_query")
            continue
        if row.get("target_must_be_retained_for_selection") is True:
            retained = stream[-8:]
            if not any(
                query_id in summary_id or summary_id in query_id
                for query_id in query_ids
                for summary_id in retained
            ):
                errors.append("selection_target_not_in_equal_retained_set")
        if row.get("durable_mutation_allowed") is not False:
            errors.append("factorial_fixture_cannot_mutate_durable_state")
    if observed_focus != {"selection", "retention", "safety"}:
        errors.append("fixture_missing_factor_focus")
    if errors:
        raise ValueError("; ".join(sorted(set(errors))))


def environment_descriptor() -> Dict[str, Any]:
    return {
        "schema": "sara-phase34-memory-cache-factorial-environment-v1",
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
        "factorial_design": {
            "retention_factors": ["equal", "logarithmic"],
            "selection_factors": ["retrieve_all", "sparse_topk"],
            "control_arm_outside_factorial": True,
            "same_retained_set_within_retention_pair": True,
            "selection_runs_after_retention": True,
            "query_visible_during_retention": False,
            "query_visible_during_selection": True,
        },
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
            "retention_state_frozen_before_selection": True,
            "retained_set_digest_must_match_within_pair": True,
            "retention_bytes_reported_separately": True,
            "selection_bytes_reported_separately": True,
            "equal_total_state_ceiling_across_arms": True,
            "equal_tuning_allowance_across_arms": True,
            "query_aware_admission_allowed": False,
            "hidden_dense_summary_allowed": False,
            "unbounded_checkpoint_scan_allowed": False,
        },
        "thresholds": {
            "selection_precision_main_effect": {"direction": "minimum", "limit": 0.1},
            "selection_recall_noninferiority": {"direction": "minimum", "limit": -0.01},
            "retention_old_recall_main_effect": {"direction": "minimum", "limit": 0.1},
            "retention_recent_resolution_main_effect": {"direction": "minimum", "limit": 0.05},
            "selection_retention_interaction_abs": {"direction": "maximum", "limit": 0.25},
            "safety_integrity": {"direction": "minimum", "limit": 1.0},
            "retained_set_identity": {"direction": "minimum", "limit": 1.0},
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
                "schema": "sara-phase34-memory-cache-factorial-draft-receipt-v1",
                "case_count": len(rows),
                "condition_count": len(rows) * len(ARMS) * len(REPLICATE_SEEDS),
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
