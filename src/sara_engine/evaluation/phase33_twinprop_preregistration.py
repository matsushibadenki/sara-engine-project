"""Immutable preregistration for the Phase 33 TwinProp-inspired ablation."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping

from sara_engine.evaluation.phase33_preregistration import (
    compare_existing_registration,
    is_managed_preregistration_path,
    preregistration_fingerprint,
)


SCHEMA = "sara-phase33-twinprop-ablation-preregistration-v1"
EXPERIMENT_ID = "phase33-twinprop-ablation-observed-v1"
ABLATION_ARMS = (
    "intact_bounded_branches",
    "passive_linear_branches",
    "topology_collapsed_aggregation",
    "no_slow_coincidence_state",
    "point_neuron_control",
)
CASE_FAMILIES = (
    "interaction_order_2",
    "interaction_order_3",
    "interaction_order_4",
    "slow_state_bridge",
    "slow_state_decay_control",
    "deterministic_contact_placement",
    "shuffled_contact_placement",
    "polarity_gated_subunit",
    "topology_collapse_control",
    "passive_linear_control",
    "fixed_readout_positive",
    "fixed_readout_negative",
    "missing_contact",
    "stale_source_revision",
)
INTERACTION_ORDERS = (2, 3, 4)
PLACEMENT_CONDITIONS = ("structured", "shuffled")
REQUIRED_BUDGETS = (
    "source_events_per_case",
    "max_total_state_bytes",
    "max_local_interactions_per_case",
    "max_latency_ms",
    "max_contacts_per_relation",
    "max_branch_slots_per_relation",
    "max_slow_state_slots_per_relation",
    "tuning_trials_per_arm",
    "restart_count_per_arm",
)
REQUIRED_METRICS = (
    "fixed_readout_quality",
    "branch_participation_monotonicity",
    "structured_over_shuffled_delta",
    "intact_over_passive_delta",
    "intact_over_collapsed_delta",
    "intact_over_no_slow_state_delta",
    "abstention_integrity",
    "state_bytes",
    "event_cost",
    "latency_ms",
    "deterministic_replay",
)
ALLOWED_DIRECTIONS = frozenset({"minimum", "maximum"})


def validate_preregistration(
    manifest: Mapping[str, Any],
    *,
    managed_path: bool,
) -> Dict[str, Any]:
    errors: List[str] = []
    if manifest.get("schema") != SCHEMA:
        errors.append("unsupported_twinprop_preregistration_schema")
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id_does_not_match_frozen_followup")
    if manifest.get("parent_experiment_id") != "phase33-structured-edge-observed-v1":
        errors.append("parent_experiment_identity_mismatch")
    if manifest.get("parent_protocol_fingerprint") != (
        "63168395ac7f5235d4173072fb52823712b89895e16610856ced77adf70d64ff"
    ):
        errors.append("parent_protocol_fingerprint_mismatch")
    if manifest.get("registered_before_execution") is not True:
        errors.append("not_registered_before_execution")
    if not managed_path:
        errors.append("preregistration_path_not_managed")
    for field in ("fixture_fingerprint", "environment_fingerprint"):
        value = manifest.get(field)
        if not (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        ):
            errors.append(f"invalid_{field}")

    if tuple(manifest.get("ablation_arms", ())) != ABLATION_ARMS:
        errors.append("ablation_arms_do_not_match_frozen_followup")
    if tuple(manifest.get("case_families", ())) != CASE_FAMILIES:
        errors.append("case_families_do_not_match_frozen_followup")
    if tuple(manifest.get("interaction_orders", ())) != INTERACTION_ORDERS:
        errors.append("interaction_orders_do_not_match_frozen_followup")
    if tuple(manifest.get("placement_conditions", ())) != PLACEMENT_CONDITIONS:
        errors.append("placement_conditions_do_not_match_frozen_followup")
    if manifest.get("replicates_per_condition") != 5:
        errors.append("replicates_per_condition_must_equal_five")
    seeds = manifest.get("replicate_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 5
        or len(set(seeds)) != 5
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        errors.append("five_unique_non_negative_replicate_seeds_required")

    readout = manifest.get("fixed_readout")
    required_readout = {
        "type": "spike_count_threshold",
        "decision_window_ticks": 4,
        "threshold": 2,
        "same_for_all_arms": True,
        "trainable": False,
        "deep_decoder_allowed": False,
    }
    if not isinstance(readout, Mapping) or any(
        readout.get(key) != value for key, value in required_readout.items()
    ):
        errors.append("fixed_readout_contract_mismatch")

    budgets = manifest.get("budgets")
    if not isinstance(budgets, Mapping):
        errors.append("budgets_must_be_mapping")
    else:
        keys = {key for key in budgets if isinstance(key, str)}
        if keys != set(REQUIRED_BUDGETS):
            errors.append("budgets_do_not_match_frozen_followup")
        for key in REQUIRED_BUDGETS:
            value = budgets.get(key)
            if type(value) is not int or value < 0:
                errors.append(f"invalid_budget:{key}")
        if budgets.get("tuning_trials_per_arm") != 1:
            errors.append("tuning_trials_per_arm_must_equal_one")
        if budgets.get("restart_count_per_arm") != 0:
            errors.append("restart_count_per_arm_must_equal_zero")

    accounting = manifest.get("resource_accounting")
    required_accounting = {
        "equal_input_events_across_arms": True,
        "equal_contact_budget_across_arms": True,
        "equal_state_budget_across_arms": True,
        "equal_tuning_allowance_across_arms": True,
        "same_readout_across_arms": True,
        "hidden_input_expansion_allowed": False,
        "gradient_selected_contact_locations": False,
    }
    if not isinstance(accounting, Mapping) or any(
        accounting.get(key) is not value
        for key, value in required_accounting.items()
    ):
        errors.append("resource_accounting_contract_mismatch")

    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, Mapping):
        errors.append("thresholds_must_be_mapping")
    else:
        keys = {key for key in thresholds if isinstance(key, str)}
        if keys != set(REQUIRED_METRICS):
            errors.append("thresholds_do_not_match_frozen_followup")
        for metric in REQUIRED_METRICS:
            spec = thresholds.get(metric)
            if not isinstance(spec, Mapping):
                errors.append(f"invalid_threshold_spec:{metric}")
                continue
            if spec.get("direction") not in ALLOWED_DIRECTIONS:
                errors.append(f"invalid_threshold_direction:{metric}")
            limit = spec.get("limit")
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
        "dense_digital_twin": False,
        "pca_runtime": False,
        "default_off": True,
        "production_mutation": False,
        "physical_energy_claim": False,
        "biological_learning_claim": False,
        "independent_evidence_required": True,
    }
    if not isinstance(policy, Mapping) or any(
        policy.get(key) is not value for key, value in required_policy.items()
    ):
        errors.append("execution_policy_does_not_match_sara_boundaries")

    try:
        computed = preregistration_fingerprint(manifest)
    except (TypeError, ValueError):
        computed = None
        errors.append("preregistration_is_not_canonical_json")
    if manifest.get("protocol_fingerprint") != computed:
        errors.append("protocol_fingerprint_mismatch")
    return {
        "valid": not errors,
        "managed_path": managed_path,
        "computed_fingerprint": computed,
        "declared_fingerprint": manifest.get("protocol_fingerprint"),
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
    validation = validate_preregistration(candidate, managed_path=managed_path)
    if not validation["valid"]:
        raise ValueError(
            "invalid Phase 33 TwinProp-inspired preregistration: "
            + "; ".join(validation["errors"])
        )
    return candidate


__all__ = [
    "ABLATION_ARMS",
    "CASE_FAMILIES",
    "EXPERIMENT_ID",
    "INTERACTION_ORDERS",
    "PLACEMENT_CONDITIONS",
    "REQUIRED_BUDGETS",
    "REQUIRED_METRICS",
    "SCHEMA",
    "build_registered_manifest",
    "compare_existing_registration",
    "is_managed_preregistration_path",
    "validate_preregistration",
]
