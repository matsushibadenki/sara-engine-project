"""Immutable preregistration contract for Phase 33 structured edges."""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Dict, List, Mapping, Tuple

from sara_engine.utils.project_paths import WORKSPACE_DIR


SCHEMA = "sara-phase33-structured-edge-preregistration-v1"
MECHANISM_ARMS = (
    "single_scalar_contact",
    "linear_multi_contact",
    "typed_independent_contacts",
    "branch_local_contacts",
    "branch_local_contacts_with_add_prune",
)
SIMPLIFICATION_LEVELS = (
    {
        "name": "baseline",
        "outer_node_fraction": 1.0,
        "outer_route_fraction": 1.0,
        "processing_depth_fraction": 1.0,
    },
    {
        "name": "moderate",
        "outer_node_fraction": 0.75,
        "outer_route_fraction": 0.75,
        "processing_depth_fraction": 0.75,
    },
    {
        "name": "strong",
        "outer_node_fraction": 0.5,
        "outer_route_fraction": 0.5,
        "processing_depth_fraction": 0.5,
    },
)
REQUIRED_CASE_FAMILIES = (
    "delay_dependent_meaning",
    "polarity_context_switch",
    "same_count_different_order",
    "branch_local_coincidence",
    "partial_contact_failure",
    "repeated_support",
    "delayed_contradiction",
    "outer_route_deletion",
    "shuffled_contact_identity",
    "shuffled_branch_placement",
    "duplicated_contact",
    "missing_contact",
    "stale_source_revision",
    "all_linear",
    "all_same_delay",
    "no_reuse",
    "random_cluster",
)
REQUIRED_BUDGETS = (
    "source_events_per_case",
    "max_total_state_bytes",
    "max_local_interactions_per_case",
    "max_latency_ms",
    "max_outer_nodes",
    "max_outer_routes",
    "max_contacts_per_relation",
    "max_branch_slots_per_relation",
    "max_internal_interactions_per_relation",
    "max_contact_rewrites_per_event",
)
REQUIRED_METRICS = (
    "ambiguous_relation_quality",
    "calibration_error",
    "abstention_integrity",
    "contradiction_recovery",
    "contact_failure_tolerance",
    "iso_quality_total_complexity_reduction",
    "state_bytes",
    "event_cost",
    "latency_ms",
    "contact_churn",
    "deterministic_replay",
)
ALLOWED_THRESHOLD_DIRECTIONS = frozenset(
    {"maximize", "minimize", "minimum", "maximum"}
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def is_managed_preregistration_path(path: str) -> bool:
    if not path:
        return False
    resolved = os.path.realpath(os.path.abspath(path))
    workspace_root = os.path.realpath(WORKSPACE_DIR)
    try:
        return os.path.commonpath([resolved, workspace_root]) == workspace_root
    except ValueError:
        return False


def preregistration_fingerprint(manifest: Mapping[str, Any]) -> str:
    payload = {
        str(key): value
        for key, value in manifest.items()
        if key != "protocol_fingerprint"
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_preregistration(
    manifest: Mapping[str, Any],
    *,
    managed_path: bool,
) -> Dict[str, Any]:
    errors: List[str] = []
    if manifest.get("schema") != SCHEMA:
        errors.append("unsupported_preregistration_schema")
    if not isinstance(manifest.get("experiment_id"), str) or not str(
        manifest.get("experiment_id", "")
    ).strip():
        errors.append("missing_experiment_id")
    if manifest.get("registered_before_execution") is not True:
        errors.append("not_registered_before_execution")
    if not managed_path:
        errors.append("preregistration_path_not_managed")
    for field in ("fixture_fingerprint", "environment_fingerprint"):
        if not _is_sha256(manifest.get(field)):
            errors.append(f"invalid_{field}")

    arms = manifest.get("mechanism_arms")
    if not isinstance(arms, (list, tuple)) or tuple(arms) != MECHANISM_ARMS:
        errors.append("mechanism_arms_do_not_match_frozen_protocol")
    levels = manifest.get("simplification_levels")
    if not isinstance(levels, (list, tuple)) or tuple(levels) != SIMPLIFICATION_LEVELS:
        errors.append("simplification_levels_do_not_match_frozen_protocol")
    families = manifest.get("case_families")
    if (
        not isinstance(families, (list, tuple))
        or tuple(families) != REQUIRED_CASE_FAMILIES
    ):
        errors.append("case_families_do_not_match_frozen_protocol")

    if manifest.get("replicates_per_condition") != 5:
        errors.append("replicates_per_condition_must_equal_five")
    seeds = manifest.get("replicate_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 5
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != 5
    ):
        errors.append("five_unique_non_negative_replicate_seeds_required")

    budgets = manifest.get("budgets")
    if not isinstance(budgets, Mapping):
        errors.append("budgets_must_be_mapping")
    else:
        budget_keys = {
            key for key in budgets if isinstance(key, str)
        }
        if len(budget_keys) != len(budgets):
            errors.append("budget_names_must_be_strings")
        missing = sorted(set(REQUIRED_BUDGETS) - budget_keys)
        extra = sorted(budget_keys - set(REQUIRED_BUDGETS))
        if missing:
            errors.append("missing_budgets:" + ",".join(missing))
        if extra:
            errors.append("unknown_budgets:" + ",".join(extra))
        for key in REQUIRED_BUDGETS:
            value = budgets.get(key)
            if type(value) is not int or value < 1:
                errors.append(f"invalid_budget:{key}")

    accounting = manifest.get("resource_accounting")
    required_accounting = {
        "equal_source_events_across_arms": True,
        "same_replicate_seeds_across_arms": True,
        "contacts_count_toward_total_state": True,
        "internal_interactions_count_toward_total_state": True,
        "internal_interactions_count_toward_event_cost": True,
        "same_latency_ceiling_across_arms": True,
        "simplification_may_not_increase_total_budget": True,
    }
    if not isinstance(accounting, Mapping) or any(
        accounting.get(key) is not expected
        for key, expected in required_accounting.items()
    ):
        errors.append("resource_accounting_does_not_prevent_hidden_complexity")

    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, Mapping):
        errors.append("thresholds_must_be_mapping")
    else:
        threshold_keys = {
            key for key in thresholds if isinstance(key, str)
        }
        if len(threshold_keys) != len(thresholds):
            errors.append("threshold_names_must_be_strings")
        missing = sorted(set(REQUIRED_METRICS) - threshold_keys)
        extra = sorted(threshold_keys - set(REQUIRED_METRICS))
        if missing:
            errors.append("missing_thresholds:" + ",".join(missing))
        if extra:
            errors.append("unknown_thresholds:" + ",".join(extra))
        for metric in REQUIRED_METRICS:
            threshold = thresholds.get(metric)
            if not isinstance(threshold, Mapping):
                if metric in thresholds:
                    errors.append(f"invalid_threshold_spec:{metric}")
                continue
            if threshold.get("direction") not in ALLOWED_THRESHOLD_DIRECTIONS:
                errors.append(f"invalid_threshold_direction:{metric}")
            limit = threshold.get("limit")
            if (
                isinstance(limit, bool)
                or not isinstance(limit, (int, float))
                or not math.isfinite(float(limit))
            ):
                errors.append(f"invalid_threshold_limit:{metric}")

    policy = manifest.get("execution_policy")
    required_policy = {
        "cpu_only": True,
        "gpu_required": False,
        "matrix_calculation": False,
        "backpropagation": False,
        "default_off": True,
        "production_mutation": False,
        "physical_energy_claim": False,
        "independent_evidence_required": True,
    }
    if not isinstance(policy, Mapping) or any(
        policy.get(key) is not expected
        for key, expected in required_policy.items()
    ):
        errors.append("execution_policy_does_not_match_phase33_boundaries")

    try:
        computed_fingerprint = preregistration_fingerprint(manifest)
    except (TypeError, ValueError):
        computed_fingerprint = None
        errors.append("preregistration_is_not_canonical_json")
    declared_fingerprint = manifest.get("protocol_fingerprint")
    if computed_fingerprint is None or declared_fingerprint != computed_fingerprint:
        errors.append("protocol_fingerprint_mismatch")
    return {
        "valid": not errors,
        "managed_path": managed_path,
        "computed_fingerprint": computed_fingerprint,
        "declared_fingerprint": declared_fingerprint,
        "errors": errors,
    }


def build_registered_manifest(
    draft: Mapping[str, Any],
    *,
    managed_path: bool,
) -> Dict[str, Any]:
    candidate = dict(draft)
    candidate.pop("protocol_fingerprint", None)
    try:
        candidate["protocol_fingerprint"] = preregistration_fingerprint(candidate)
    except (TypeError, ValueError):
        candidate["protocol_fingerprint"] = ""
        validation = validate_preregistration(
            candidate,
            managed_path=managed_path,
        )
        raise ValueError(
            "invalid Phase 33 preregistration: "
            + "; ".join(validation["errors"])
        )
    validation = validate_preregistration(candidate, managed_path=managed_path)
    if not validation["valid"]:
        raise ValueError(
            "invalid Phase 33 preregistration: "
            + "; ".join(validation["errors"])
        )
    return candidate


def compare_existing_registration(
    existing: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Tuple[bool, str]:
    if not existing:
        return True, "new_registration"
    if dict(existing) == dict(candidate):
        return True, "identical_registration_preserved"
    return False, "existing_registration_is_immutable"


__all__ = [
    "ALLOWED_THRESHOLD_DIRECTIONS",
    "MECHANISM_ARMS",
    "REQUIRED_BUDGETS",
    "REQUIRED_CASE_FAMILIES",
    "REQUIRED_METRICS",
    "SCHEMA",
    "SIMPLIFICATION_LEVELS",
    "build_registered_manifest",
    "compare_existing_registration",
    "is_managed_preregistration_path",
    "preregistration_fingerprint",
    "validate_preregistration",
]
