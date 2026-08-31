"""Fail-closed Phase 37 structural-invariant preregistration contract."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from typing import Any, Dict, Mapping, Sequence, Tuple


SCHEMA = "sara-phase37-structural-invariant-preregistration-v1"
MECHANISM_ARMS: Tuple[str, ...] = (
    "exact_verified_edge_retrieval",
    "bounded_verified_path_composition",
    "relation_type_jaccard_analogy",
    "canonical_typed_motif_context_free",
    "canonical_typed_motif_context_exception_aware",
    "intact_candidate_shuffled_binding_control",
)
REQUIRED_SHUFFLES: Tuple[str, ...] = (
    "role_identity",
    "topology",
    "temporal_order",
    "causal_direction",
    "counterexample_binding",
    "evidence_binding",
)
REQUIRED_CASE_FAMILIES: Tuple[str, ...] = (
    "label_renamed_isomorph",
    "same_relations_different_topology",
    "unseen_nodes",
    "heldout_domain",
    "multi_edge_role_transfer",
    "temporal_order_reversal",
    "causal_direction_reversal",
    "context_change",
    "rare_exception",
    "revised_evidence",
    "contradiction",
    "missing_role",
    "adversarial_hub",
    "no_transfer",
)
REQUIRED_BUDGETS: Tuple[str, ...] = (
    "source_events_per_case",
    "max_active_nodes",
    "max_active_edges",
    "max_patterns",
    "max_roles_per_pattern",
    "max_edges_per_pattern",
    "max_exemplars_per_pattern",
    "max_counterexamples_per_pattern",
    "max_candidate_patterns_per_query",
    "max_proposals_per_query",
    "max_propagation_fanout",
    "max_state_bytes",
    "max_event_cost",
    "max_cpu_latency_ms",
    "max_tuning_attempts",
)
REQUIRED_METRICS: Tuple[str, ...] = (
    "verified_novel_relation_precision",
    "verified_novel_relation_recall",
    "justified_abstention_accuracy",
    "heldout_domain_transfer_accuracy",
    "rare_exception_preservation",
    "direction_order_sensitivity",
    "role_map_consistency",
    "evidence_chain_completeness",
    "revision_retraction_accuracy",
    "deterministic_replay",
)
REQUIRED_TRUE_POLICIES: Tuple[str, ...] = (
    "split_by_structural_family_and_source",
    "near_isomorphic_variants_same_partition",
    "node_aliases_same_partition",
    "source_revisions_same_partition",
    "answer_hidden_until_proposal_frozen",
    "candidate_query_blind_to_evaluator_labels",
    "equal_source_events_across_arms",
    "equal_total_capacity_across_arms",
    "same_seeds_across_arms",
    "all_pattern_state_counted",
    "all_match_work_counted",
)


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("protocol_fingerprint", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def is_managed_preregistration_path(path: str) -> bool:
    normalized = os.path.realpath(path)
    return f"{os.sep}workspace{os.sep}" in normalized


def validate_preregistration(manifest: Mapping[str, Any], *, managed_path: bool) -> Dict[str, Any]:
    errors = []
    if not managed_path:
        errors.append("preregistration_path_not_managed")
    if manifest.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    if manifest.get("registered_before_candidate_implementation") is not True:
        errors.append("candidate_implementation_boundary_not_frozen")
    if tuple(manifest.get("mechanism_arms", ())) != MECHANISM_ARMS:
        errors.append("mechanism_arms_do_not_match_frozen_protocol")
    if tuple(manifest.get("case_families", ())) != REQUIRED_CASE_FAMILIES:
        errors.append("case_families_do_not_match_frozen_protocol")
    if tuple(manifest.get("ablation_shuffles", ())) != REQUIRED_SHUFFLES:
        errors.append("ablation_shuffles_do_not_match_frozen_protocol")
    seeds = manifest.get("replicate_seeds", ())
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        errors.append("at_least_five_unique_seeds_required")
    budgets = manifest.get("budgets", {})
    missing_budgets = [key for key in REQUIRED_BUDGETS if key not in budgets]
    if missing_budgets:
        errors.append("missing_budgets:" + ",".join(missing_budgets))
    elif any(not isinstance(budgets[key], (int, float)) or budgets[key] <= 0 for key in REQUIRED_BUDGETS):
        errors.append("budgets_must_be_positive_numbers")
    metrics = manifest.get("thresholds", {})
    missing_metrics = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing_metrics:
        errors.append("missing_thresholds:" + ",".join(missing_metrics))
    split = manifest.get("leakage_and_resource_policy", {})
    if any(split.get(key) is not True for key in REQUIRED_TRUE_POLICIES):
        errors.append("leakage_or_resource_policy_incomplete")
    identity = manifest.get("canonical_identity", {})
    if identity.get("node_identity_in_fingerprint") is not False or identity.get("task_label_in_fingerprint") is not False:
        errors.append("canonical_identity_allows_label_leakage")
    if identity.get("roles") != ["role:source", "role:mediator", "role:target"]:
        errors.append("canonical_role_schema_not_frozen")
    execution = manifest.get("execution_policy", {})
    required_execution = {
        "cpu_only": True,
        "gpu_required": False,
        "matrix_calculation": False,
        "backpropagation": False,
        "external_model": False,
        "default_off": True,
        "production_mutation": False,
        "durable_graph_mutation": False,
        "human_approval_required_for_integration": True,
    }
    if any(execution.get(key) != value for key, value in required_execution.items()):
        errors.append("execution_policy_does_not_match_phase37_boundaries")
    prereqs = manifest.get("prerequisites", {})
    if prereqs.get("phase21_independent_gate_passed") is not True or not prereqs.get("phase21_report_sha256"):
        errors.append("phase21_independent_prerequisite_missing")
    fingerprint = manifest.get("protocol_fingerprint")
    if fingerprint is not None and fingerprint != _canonical_digest(manifest):
        errors.append("protocol_fingerprint_mismatch")
    return {"valid": not errors, "errors": errors}


def build_registered_manifest(draft: Mapping[str, Any], *, managed_path: bool) -> Dict[str, Any]:
    candidate = deepcopy(dict(draft))
    candidate.pop("protocol_fingerprint", None)
    validation = validate_preregistration(candidate, managed_path=managed_path)
    if not validation["valid"]:
        raise ValueError(";".join(validation["errors"]))
    candidate["protocol_fingerprint"] = _canonical_digest(candidate)
    return candidate


def compare_existing_registration(existing: Mapping[str, Any], candidate: Mapping[str, Any]) -> Tuple[bool, str]:
    if not existing:
        return True, "new_registration"
    if dict(existing) == dict(candidate):
        return True, "identical_registration_preserved"
    return False, "existing_registration_is_immutable"


__all__ = [
    "MECHANISM_ARMS", "REQUIRED_BUDGETS", "REQUIRED_CASE_FAMILIES",
    "REQUIRED_METRICS", "REQUIRED_SHUFFLES", "SCHEMA",
    "build_registered_manifest", "compare_existing_registration",
    "is_managed_preregistration_path", "validate_preregistration",
]
