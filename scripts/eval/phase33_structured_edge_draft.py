#!/usr/bin/env python3
"""Build the managed Phase 33 fixture and environment preregistration draft."""

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

from sara_engine.evaluation.phase33_preregistration import (  # noqa: E402
    MECHANISM_ARMS,
    REQUIRED_CASE_FAMILIES,
    SCHEMA,
    SIMPLIFICATION_LEVELS,
    build_registered_manifest,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

CASE_SCHEMA = "sara-phase33-structured-edge-case-v1"
DEFAULT_FIXTURE = processed_data_path(
    "benchmark_fixtures",
    "phase33_structured_edge_cases.jsonl",
)
DEFAULT_DRAFT = workspace_path(
    "evaluation",
    "phase33_structured_edge_preregistration_draft.json",
)
DEFAULT_ENVIRONMENT = workspace_path(
    "evaluation",
    "phase33_structured_edge_environment.json",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def load_fixture(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    validate_fixture(rows)
    return rows


def validate_fixture(rows: Sequence[Mapping[str, Any]]) -> None:
    families = [row.get("family") for row in rows]
    case_ids = [row.get("case_id") for row in rows]
    errors = []
    if tuple(families) != REQUIRED_CASE_FAMILIES:
        errors.append("fixture_case_families_do_not_match_frozen_protocol")
    if (
        len(case_ids) != len(set(case_ids))
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
    ):
        errors.append("fixture_case_ids_must_be_unique")
    for row in rows:
        if row.get("schema") != CASE_SCHEMA:
            errors.append("unsupported_phase33_case_schema")
        if row.get("observed_only") is not True:
            errors.append("phase33_fixture_must_be_observed_only")
        if not isinstance(row.get("source_revision"), str):
            errors.append("phase33_case_missing_source_revision")
        if (
            not isinstance(row.get("outer_relation"), list)
            or len(row.get("outer_relation", [])) != 2
        ):
            errors.append("phase33_case_invalid_outer_relation")
        contacts = row.get("contacts")
        events = row.get("events")
        if not isinstance(contacts, list) or not 1 <= len(contacts) <= 4:
            errors.append("phase33_case_contact_budget_exceeded")
        if not isinstance(events, list) or not 1 <= len(events) <= 128:
            errors.append("phase33_case_event_budget_exceeded")
        expected = row.get("expected")
        if (
            not isinstance(expected, Mapping)
            or expected.get("durable_mutation_allowed") is not False
        ):
            errors.append("phase33_case_durable_mutation_not_blocked")
    if errors:
        raise ValueError("; ".join(sorted(set(errors))))


def environment_descriptor() -> Dict[str, Any]:
    return {
        "schema": "sara-phase33-environment-v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "cpu_only": True,
        "gpu_required": False,
        "matrix_calculation": False,
        "backpropagation": False,
        "physical_energy_claim": False,
    }


def build_draft(
    rows: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
) -> Dict[str, Any]:
    validate_fixture(rows)
    draft = {
        "schema": SCHEMA,
        "experiment_id": "phase33-structured-edge-observed-v1",
        "registered_before_execution": True,
        "fixture_fingerprint": _digest(list(rows)),
        "environment_fingerprint": _digest(dict(environment)),
        "mechanism_arms": list(MECHANISM_ARMS),
        "simplification_levels": [
            dict(level) for level in SIMPLIFICATION_LEVELS
        ],
        "case_families": list(REQUIRED_CASE_FAMILIES),
        "replicates_per_condition": 5,
        "replicate_seeds": [101, 211, 307, 401, 503],
        "budgets": {
            "source_events_per_case": 128,
            "max_total_state_bytes": 4096,
            "max_local_interactions_per_case": 256,
            "max_latency_ms": 50,
            "max_outer_nodes": 64,
            "max_outer_routes": 128,
            "max_contacts_per_relation": 4,
            "max_branch_slots_per_relation": 4,
            "max_internal_interactions_per_relation": 8,
            "max_contact_rewrites_per_event": 2,
        },
        "resource_accounting": {
            "equal_source_events_across_arms": True,
            "same_replicate_seeds_across_arms": True,
            "contacts_count_toward_total_state": True,
            "internal_interactions_count_toward_total_state": True,
            "internal_interactions_count_toward_event_cost": True,
            "same_latency_ceiling_across_arms": True,
            "simplification_may_not_increase_total_budget": True,
        },
        "thresholds": {
            "ambiguous_relation_quality": {"direction": "minimum", "limit": 0.7},
            "calibration_error": {"direction": "maximum", "limit": 0.15},
            "abstention_integrity": {"direction": "minimum", "limit": 0.95},
            "contradiction_recovery": {"direction": "maximum", "limit": 8},
            "contact_failure_tolerance": {"direction": "minimum", "limit": 0.8},
            "iso_quality_total_complexity_reduction": {
                "direction": "minimum",
                "limit": 0.1,
            },
            "state_bytes": {"direction": "maximum", "limit": 4096},
            "event_cost": {"direction": "maximum", "limit": 256},
            "latency_ms": {"direction": "maximum", "limit": 50},
            "contact_churn": {"direction": "maximum", "limit": 0.1},
            "deterministic_replay": {"direction": "minimum", "limit": 1.0},
        },
        "execution_policy": {
            "cpu_only": True,
            "gpu_required": False,
            "matrix_calculation": False,
            "backpropagation": False,
            "default_off": True,
            "production_mutation": False,
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
    for path, value in (
        (args.environment_path, environment),
        (args.draft_path, draft),
    ):
        with open(ensure_parent_directory(path), "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    print(
        json.dumps(
            {
                "schema": "sara-phase33-draft-build-receipt-v1",
                "case_count": len(rows),
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
