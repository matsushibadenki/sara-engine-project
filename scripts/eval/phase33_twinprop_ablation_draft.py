#!/usr/bin/env python3
"""Build the Phase 33 TwinProp-inspired immutable follow-up draft."""

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

from sara_engine.evaluation.phase33_twinprop_preregistration import (  # noqa: E402
    ABLATION_ARMS,
    CASE_FAMILIES,
    EXPERIMENT_ID,
    INTERACTION_ORDERS,
    PLACEMENT_CONDITIONS,
    SCHEMA,
    build_registered_manifest,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)

CASE_SCHEMA = "sara-phase33-twinprop-ablation-case-v1"
DEFAULT_FIXTURE = processed_data_path(
    "benchmark_fixtures",
    "phase33_twinprop_ablation_cases.jsonl",
)
DEFAULT_DRAFT = workspace_path(
    "evaluation",
    "phase33_twinprop_ablation_preregistration_draft.json",
)
DEFAULT_ENVIRONMENT = workspace_path(
    "evaluation",
    "phase33_twinprop_ablation_environment.json",
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
    errors: List[str] = []
    if tuple(row.get("family") for row in rows) != CASE_FAMILIES:
        errors.append("fixture_case_families_do_not_match_frozen_followup")
    case_ids = [row.get("case_id") for row in rows]
    if (
        len(case_ids) != len(set(case_ids))
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
    ):
        errors.append("fixture_case_ids_must_be_unique")
    observed_orders = set()
    observed_placements = set()
    for row in rows:
        if row.get("schema") != CASE_SCHEMA:
            errors.append("unsupported_twinprop_case_schema")
        if row.get("observed_only") is not True:
            errors.append("twinprop_fixture_must_be_observed_only")
        if not isinstance(row.get("source_revision"), str):
            errors.append("twinprop_case_missing_source_revision")
        order = row.get("interaction_order")
        if type(order) is not int or not 1 <= order <= 4:
            errors.append("invalid_interaction_order")
        elif order in INTERACTION_ORDERS:
            observed_orders.add(order)
        placement = row.get("placement")
        if placement not in PLACEMENT_CONDITIONS:
            errors.append("invalid_placement_condition")
        else:
            observed_placements.add(placement)
        contacts = row.get("contacts")
        events = row.get("events")
        if not isinstance(contacts, list) or not 1 <= len(contacts) <= 8:
            errors.append("twinprop_contact_budget_exceeded")
        if not isinstance(events, list) or not 1 <= len(events) <= 64:
            errors.append("twinprop_event_budget_exceeded")
        if isinstance(contacts, list) and any(
            type(contact.get("branch")) is not int
            or not 0 <= contact.get("branch", -1) < 4
            for contact in contacts
        ):
            errors.append("twinprop_branch_budget_exceeded")
        expected = row.get("expected")
        if (
            not isinstance(expected, Mapping)
            or type(expected.get("readout_target")) is not bool
            or expected.get("durable_mutation_allowed") is not False
        ):
            errors.append("invalid_twinprop_expected_contract")
    if observed_orders != set(INTERACTION_ORDERS):
        errors.append("fixture_missing_interaction_order_sweep")
    if observed_placements != set(PLACEMENT_CONDITIONS):
        errors.append("fixture_missing_placement_control")
    if errors:
        raise ValueError("; ".join(sorted(set(errors))))


def environment_descriptor() -> Dict[str, Any]:
    return {
        "schema": "sara-phase33-twinprop-environment-v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "cpu_only": True,
        "gpu_required": False,
        "matrix_calculation": False,
        "backpropagation": False,
        "dense_digital_twin": False,
    }


def build_draft(
    rows: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
) -> Dict[str, Any]:
    validate_fixture(rows)
    draft = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": "phase33-structured-edge-observed-v1",
        "parent_protocol_fingerprint": (
            "63168395ac7f5235d4173072fb52823712b89895e16610856ced77adf70d64ff"
        ),
        "registered_before_execution": True,
        "fixture_fingerprint": _digest(list(rows)),
        "environment_fingerprint": _digest(dict(environment)),
        "ablation_arms": list(ABLATION_ARMS),
        "case_families": list(CASE_FAMILIES),
        "interaction_orders": list(INTERACTION_ORDERS),
        "placement_conditions": list(PLACEMENT_CONDITIONS),
        "replicates_per_condition": 5,
        "replicate_seeds": [107, 223, 311, 419, 521],
        "fixed_readout": {
            "type": "spike_count_threshold",
            "decision_window_ticks": 4,
            "threshold": 2,
            "same_for_all_arms": True,
            "trainable": False,
            "deep_decoder_allowed": False,
        },
        "budgets": {
            "source_events_per_case": 64,
            "max_total_state_bytes": 4096,
            "max_local_interactions_per_case": 128,
            "max_latency_ms": 50,
            "max_contacts_per_relation": 8,
            "max_branch_slots_per_relation": 4,
            "max_slow_state_slots_per_relation": 4,
            "tuning_trials_per_arm": 1,
            "restart_count_per_arm": 0,
        },
        "resource_accounting": {
            "equal_input_events_across_arms": True,
            "equal_contact_budget_across_arms": True,
            "equal_state_budget_across_arms": True,
            "equal_tuning_allowance_across_arms": True,
            "same_readout_across_arms": True,
            "hidden_input_expansion_allowed": False,
            "gradient_selected_contact_locations": False,
        },
        "thresholds": {
            "fixed_readout_quality": {"direction": "minimum", "limit": 0.75},
            "branch_participation_monotonicity": {
                "direction": "minimum",
                "limit": 1.0,
            },
            "structured_over_shuffled_delta": {
                "direction": "minimum",
                "limit": 0.1,
            },
            "intact_over_passive_delta": {
                "direction": "minimum",
                "limit": 0.1,
            },
            "intact_over_collapsed_delta": {
                "direction": "minimum",
                "limit": 0.1,
            },
            "intact_over_no_slow_state_delta": {
                "direction": "minimum",
                "limit": 0.1,
            },
            "abstention_integrity": {"direction": "minimum", "limit": 1.0},
            "state_bytes": {"direction": "maximum", "limit": 4096},
            "event_cost": {"direction": "maximum", "limit": 128},
            "latency_ms": {"direction": "maximum", "limit": 50},
            "deterministic_replay": {"direction": "minimum", "limit": 1.0},
        },
        "execution_policy": {
            "cpu_only": True,
            "gpu_required": False,
            "matrix_calculation": False,
            "backpropagation": False,
            "dense_digital_twin": False,
            "pca_runtime": False,
            "default_off": True,
            "production_mutation": False,
            "physical_energy_claim": False,
            "biological_learning_claim": False,
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
                "schema": "sara-phase33-twinprop-draft-receipt-v1",
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
