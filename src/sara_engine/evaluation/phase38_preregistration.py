"""Fail-closed Phase 38 canonical structural-delta preregistration."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from typing import Any, Dict, Mapping, Tuple


SCHEMA = "sara-phase38-structural-delta-preregistration-v1"
ARMS: Tuple[str, ...] = (
    "complete_canonical_snapshots",
    "base_plus_unshared_chronological_edit_log",
    "base_plus_unshared_canonical_typed_deltas",
    "shared_invariants_plus_instance_deltas_exceptions",
    "shared_invariants_plus_transformation_patterns",
    "intact_transformation_shuffled_binding_control",
)
OPERATORS: Tuple[str, ...] = (
    "ADD_NODE", "REMOVE_NODE", "ADD_RELATION", "REMOVE_RELATION",
    "CHANGE_ROLE", "CHANGE_VALUE", "GENERALIZE", "SPECIALIZE",
    "REORDER_TIME", "MERGE", "SPLIT",
)
CASE_FAMILIES: Tuple[str, ...] = (
    "throw_role_substitution", "bird_exception", "penguin_exception",
    "ostrich_exception", "emu_exception", "cross_domain_support_function",
    "add_remove_relation", "role_value_change", "generalize_specialize",
    "temporal_reorder", "merge_split", "repeated_transformation_family",
    "non_compressible_random", "ambiguous_base", "equivalent_cost_base",
    "long_delta_chain", "branch_merge_conflict", "duplicated_evidence",
    "stale_revision", "contradiction", "source_replacement", "missing_base",
    "corrupted_delta", "invalid_inverse", "cycle", "budget_exceeded",
)
REQUIRED_BUDGETS: Tuple[str, ...] = (
    "source_events_per_case", "max_nodes_per_structure", "max_relations_per_structure",
    "max_operations_per_delta", "max_base_candidates", "max_chain_depth",
    "max_branch_width", "materialization_interval", "max_patterns",
    "max_exemplars_per_pattern", "max_exceptions_per_pattern",
    "max_state_bytes", "max_event_cost", "max_cpu_latency_ms", "max_tuning_attempts",
)
REQUIRED_METRICS: Tuple[str, ...] = (
    "exact_reconstruction_rate", "digest_match_rate", "rollback_fidelity",
    "provenance_tombstone_preservation", "base_selection_stability",
    "verified_recall_preservation", "justified_abstention_accuracy",
    "exception_preservation", "revision_recovery", "evidence_traceability",
    "deterministic_replay", "withheld_transformation_precision",
    "withheld_transformation_recall",
)
REQUIRED_COSTS: Tuple[str, ...] = (
    "structure_node", "structure_relation", "delta_header", "precondition",
    "evidence_link", "exception", "codebook_entry", "decoder_byte",
    "index_byte", "checkpoint_byte", "materialization_state_byte",
)


def _digest(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("protocol_fingerprint", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode()).hexdigest()


def is_managed_preregistration_path(path: str) -> bool:
    return f"{os.sep}workspace{os.sep}" in os.path.realpath(path)


def validate_preregistration(manifest: Mapping[str, Any], *, managed_path: bool) -> Dict[str, Any]:
    errors = []
    if not managed_path:
        errors.append("preregistration_path_not_managed")
    if manifest.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    if manifest.get("registered_before_candidate_implementation") is not True:
        errors.append("implementation_boundary_not_frozen")
    if tuple(manifest.get("arms", ())) != ARMS:
        errors.append("arms_do_not_match_frozen_protocol")
    if tuple(manifest.get("operators", ())) != OPERATORS:
        errors.append("operators_do_not_match_frozen_vocabulary")
    if tuple(manifest.get("case_families", ())) != CASE_FAMILIES:
        errors.append("case_families_do_not_match_frozen_protocol")
    seeds = manifest.get("replicate_seeds", ())
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        errors.append("at_least_five_unique_seeds_required")
    budgets = manifest.get("budgets", {})
    missing = [key for key in REQUIRED_BUDGETS if key not in budgets]
    if missing:
        errors.append("missing_budgets:" + ",".join(missing))
    elif any(not isinstance(budgets[key], (int, float)) or budgets[key] <= 0 for key in REQUIRED_BUDGETS):
        errors.append("budgets_must_be_positive")
    thresholds = manifest.get("thresholds", {})
    missing = [key for key in REQUIRED_METRICS if key not in thresholds]
    if missing:
        errors.append("missing_thresholds:" + ",".join(missing))
    costs = manifest.get("description_cost_table", {})
    missing = [key for key in REQUIRED_COSTS if key not in costs]
    if missing:
        errors.append("missing_description_costs:" + ",".join(missing))
    elif any(not isinstance(costs[key], (int, float)) or costs[key] < 0 for key in REQUIRED_COSTS):
        errors.append("description_costs_must_be_nonnegative")
    canonical = manifest.get("canonical_schema", {})
    if canonical.get("roles") != ["role:source", "role:mediator", "role:target"] or canonical.get("entity_labels_define_invariant") is not False:
        errors.append("phase37_canonical_role_schema_not_preserved")
    delta = manifest.get("delta_contract", {})
    required_delta = ("base_digest_required", "target_digest_required", "ordered_operations", "preconditions_required", "evidence_required", "inverse_required", "remove_creates_tombstone", "failed_precondition_abstains")
    if any(delta.get(key) is not True for key in required_delta):
        errors.append("delta_contract_incomplete")
    leakage = manifest.get("leakage_policy", {})
    required_leakage = ("split_by_source_structure_and_transformation_family", "equivalent_delta_same_partition", "renamed_template_same_partition", "descendant_revision_same_partition", "target_hidden_until_materialization_frozen", "withheld_delta_hidden_until_proposal_frozen", "target_aware_base_selection_forbidden", "evaluator_labels_absent_from_candidate_trace")
    if any(leakage.get(key) is not True for key in required_leakage):
        errors.append("leakage_policy_incomplete")
    accounting = manifest.get("resource_accounting", {})
    required_accounting = ("codebook_counted", "decoder_counted", "index_counted", "checkpoint_counted", "exceptions_counted", "materialization_state_counted", "same_source_events_across_arms", "same_total_capacity_across_arms")
    if any(accounting.get(key) is not True for key in required_accounting):
        errors.append("resource_accounting_incomplete")
    execution = manifest.get("execution_policy", {})
    expected = {"cpu_only": True, "gpu_required": False, "matrix_calculation": False, "backpropagation": False, "external_model": False, "default_off": True, "production_mutation": False, "snapshot_format_mutation": False, "human_approval_required_for_integration": True}
    if any(execution.get(key) != value for key, value in expected.items()):
        errors.append("execution_policy_mismatch")
    prereq = manifest.get("prerequisites", {})
    if prereq.get("phase37_role_schema_frozen") is not True or not prereq.get("phase37_preregistration_sha256") or not prereq.get("phase37_negative_result_sha256"):
        errors.append("phase37_prerequisite_missing")
    fingerprint = manifest.get("protocol_fingerprint")
    if fingerprint is not None and fingerprint != _digest(manifest):
        errors.append("protocol_fingerprint_mismatch")
    return {"valid": not errors, "errors": errors}


def build_registered_manifest(draft: Mapping[str, Any], *, managed_path: bool) -> Dict[str, Any]:
    candidate = deepcopy(dict(draft))
    candidate.pop("protocol_fingerprint", None)
    validation = validate_preregistration(candidate, managed_path=managed_path)
    if not validation["valid"]:
        raise ValueError(";".join(validation["errors"]))
    candidate["protocol_fingerprint"] = _digest(candidate)
    return candidate


def compare_existing_registration(existing: Mapping[str, Any], candidate: Mapping[str, Any]) -> Tuple[bool, str]:
    if not existing:
        return True, "new_registration"
    if dict(existing) == dict(candidate):
        return True, "identical_registration_preserved"
    return False, "existing_registration_is_immutable"


__all__ = ["ARMS", "CASE_FAMILIES", "OPERATORS", "REQUIRED_BUDGETS", "REQUIRED_COSTS", "REQUIRED_METRICS", "SCHEMA", "build_registered_manifest", "compare_existing_registration", "is_managed_preregistration_path", "validate_preregistration"]
