#!/usr/bin/env python3
"""Build the immutable Phase 34 memory checkpoint cache draft."""

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

from sara_engine.evaluation.phase34_memory_cache_preregistration import (  # noqa: E402
    ARMS,
    CASE_FAMILIES,
    EXPERIMENT_ID,
    SCHEMA,
    build_registered_manifest,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

CASE_SCHEMA = "sara-phase34-memory-checkpoint-cache-case-v1"
DEFAULT_FIXTURE = processed_data_path("benchmark_fixtures", "phase34_memory_checkpoint_cache_cases.jsonl")
DEFAULT_DRAFT = workspace_path("evaluation", "phase34_memory_checkpoint_cache_preregistration_draft.json")
DEFAULT_ENVIRONMENT = workspace_path("evaluation", "phase34_memory_checkpoint_cache_environment.json")


def _digest(value: Any) -> str:
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_fixture(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    validate_fixture(rows)
    return rows


def validate_fixture(rows: Sequence[Mapping[str, Any]]) -> None:
    errors: List[str] = []
    if tuple(row.get("family") for row in rows) != CASE_FAMILIES:
        errors.append("fixture_case_families_do_not_match_frozen_phase34")
    case_ids = [row.get("case_id") for row in rows]
    if len(case_ids) != len(set(case_ids)) or any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        errors.append("fixture_case_ids_must_be_unique")
    for row in rows:
        if row.get("schema") != CASE_SCHEMA:
            errors.append("unsupported_memory_cache_case_schema")
        if row.get("observed_only") is not True:
            errors.append("memory_cache_fixture_must_be_observed_only")
        for field in ("source_revision", "runtime_fingerprint", "schema_fingerprint"):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                errors.append(f"memory_cache_case_missing_{field}")
        events = row.get("events")
        checkpoints = row.get("checkpoints")
        if not isinstance(events, list) or not 1 <= len(events) <= 128:
            errors.append("memory_cache_event_budget_exceeded")
        if not isinstance(checkpoints, list) or not 1 <= len(checkpoints) <= 8:
            errors.append("memory_cache_checkpoint_budget_exceeded")
        if isinstance(checkpoints, list):
            checkpoint_ids = [item.get("checkpoint_id") for item in checkpoints if isinstance(item, Mapping)]
            if len(checkpoint_ids) != len(set(checkpoint_ids)):
                errors.append("memory_cache_checkpoint_ids_must_be_unique")
            for item in checkpoints:
                summaries = item.get("summary_ids") if isinstance(item, Mapping) else None
                if not isinstance(summaries, list) or not 1 <= len(summaries) <= 8:
                    errors.append("memory_cache_summary_budget_exceeded")
        query = row.get("query")
        if not isinstance(query, Mapping) or not isinstance(query.get("summary_ids"), list):
            errors.append("invalid_memory_cache_query")
        expected = row.get("expected")
        if not isinstance(expected, Mapping) or expected.get("decision") not in {"retrieve", "abstain", "reject_stale", "reject_contradiction", "evict", "merge"} or expected.get("durable_mutation_allowed") is not False:
            errors.append("invalid_memory_cache_expected_contract")
    if errors:
        raise ValueError("; ".join(sorted(set(errors))))


def environment_descriptor() -> Dict[str, Any]:
    return {
        "schema": "sara-phase34-memory-checkpoint-cache-environment-v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "cpu_only": True,
        "gpu_required": False,
        "matrix_calculation": False,
        "backpropagation": False,
    }


def build_draft(rows: Sequence[Mapping[str, Any]], environment: Mapping[str, Any]) -> Dict[str, Any]:
    validate_fixture(rows)
    draft = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "registered_before_execution": True,
        "fixture_fingerprint": _digest(list(rows)),
        "environment_fingerprint": _digest(dict(environment)),
        "arms": list(ARMS),
        "case_families": list(CASE_FAMILIES),
        "segmentation": {
            "semantic_boundaries_required": True,
            "equal_segment_event_span": 4,
            "logarithmic_retention_tiers": [1, 2, 4, 8],
            "merge_order": "oldest_first",
            "preserve_provenance": True,
            "parameter_averaging": False,
        },
        "selection": {
            "selected_k": 2,
            "scoring": "deterministic_scalar_overlap_verified_recency",
            "summary_overlap_weight": 4,
            "verified_source_weight": 2,
            "recency_weight": 1,
            "exclude_contradicted": True,
            "exclude_stale_runtime": True,
            "exclude_stale_schema": True,
            "tie_break": "event_start_then_event_end_then_checkpoint_id",
            "learned_router": False,
            "softmax": False,
        },
        "budgets": {
            "source_events_per_case": 128,
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
            "equal_source_events_across_arms": True,
            "equal_state_byte_ceiling_across_arms": True,
            "equal_tuning_allowance_across_arms": True,
            "cache_count_bounded_independent_of_sequence_length": True,
            "hidden_dense_summary_allowed": False,
            "unbounded_checkpoint_scan_allowed": False,
        },
        "thresholds": {
            "delayed_recall_quality": {"direction": "minimum", "limit": 0.75},
            "revision_uptake": {"direction": "minimum", "limit": 1.0},
            "contradiction_rejection": {"direction": "minimum", "limit": 1.0},
            "abstention_integrity": {"direction": "minimum", "limit": 1.0},
            "selection_precision": {"direction": "minimum", "limit": 0.75},
            "selection_recall": {"direction": "minimum", "limit": 0.75},
            "useful_checkpoint_rate": {"direction": "minimum", "limit": 0.5},
            "retained_temporal_resolution": {"direction": "minimum", "limit": 0.5},
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
    print(json.dumps({"schema": "sara-phase34-memory-checkpoint-cache-draft-receipt-v1", "case_count": len(rows), "fixture_fingerprint": draft["fixture_fingerprint"], "environment_fingerprint": draft["environment_fingerprint"], "draft_path": os.path.realpath(args.draft_path), "environment_path": os.path.realpath(args.environment_path)}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
