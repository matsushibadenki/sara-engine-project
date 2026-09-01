"""Fail-closed Phase 39 anonymous local-reuse preregistration."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from typing import Any, Dict, Mapping, Tuple


SCHEMA = "sara-phase39-anonymous-local-reuse-preregistration-v1"
ARMS: Tuple[str, ...] = (
    "supervised_sparse_own_latent_reference",
    "phase37_explicit_typed_motif_control",
    "offline_global_clustering_diagnostic",
    "anonymous_local_reuse_no_homeostasis",
    "anonymous_local_reuse_intact",
    "anonymous_local_reuse_shuffled",
)
CASE_FAMILIES: Tuple[str, ...] = (
    "surface_variant_same_generator",
    "similar_surface_different_generator",
    "repeated_local_fragments",
    "unseen_fragment_composition",
    "overlapping_hidden_factors",
    "unnamed_multiscale_factor",
    "temporal_order_reversal",
    "interval_shift",
    "causal_direction_reversal",
    "spatial_role_interaction",
    "rare_exception",
    "abrupt_context_shift",
    "irrelevant_burst",
    "forced_hash_collision",
    "dominant_frequency_pressure",
    "random_noncompressible_stream",
    "all_new_no_reuse",
    "capacity_saturation",
    "dead_unit_recovery",
    "revision_contradiction_expiry",
    "source_replacement",
)
SHUFFLES: Tuple[str, ...] = ("event_order", "phase", "unit_identity", "neighborhood_assignment")
REQUIRED_BUDGETS: Tuple[str, ...] = (
    "source_events_per_case",
    "max_active_units",
    "max_candidate_units_per_fragment",
    "max_units_per_assembly",
    "max_active_assemblies",
    "max_assemblies_per_unit",
    "max_hierarchy_depth",
    "max_hierarchy_width",
    "max_support_refs_per_unit",
    "max_counterexample_refs_per_unit",
    "max_state_bytes",
    "max_event_cost",
    "max_cpu_latency_ms",
    "max_tuning_attempts",
)
REQUIRED_METRICS: Tuple[str, ...] = (
    "heldout_prediction_accuracy",
    "justified_abstention",
    "cross_context_transfer_delta",
    "posthoc_hidden_factor_recovery",
    "reuse_selectivity",
    "assembly_stability",
    "rare_exception_preservation",
    "ablation_prediction_delta",
    "random_stream_false_assembly_rate",
    "dominant_unit_rate",
    "dead_unit_rate",
    "always_active_unit_rate",
    "revision_retraction_accuracy",
    "evidence_chain_completeness",
    "deterministic_replay",
)
REQUIRED_TRUE_POLICIES: Tuple[str, ...] = (
    "split_by_source_and_hidden_generator",
    "same_generator_seed_same_partition",
    "surface_variants_same_partition",
    "source_revisions_same_partition",
    "hidden_factor_labels_evaluator_only",
    "answer_hidden_until_prediction_frozen",
    "candidate_trace_excludes_evaluator_labels",
    "equal_source_events_across_arms",
    "equal_total_capacity_across_runtime_arms",
    "same_seeds_across_arms",
    "offline_clustering_excluded_from_runtime_acceptance",
    "all_unit_assembly_hierarchy_state_counted",
    "all_candidate_neighborhood_work_counted",
    "single_registered_tuning_attempt",
)


def _digest(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("protocol_fingerprint", None)
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def is_managed_preregistration_path(path: str) -> bool:
    return f"{os.sep}workspace{os.sep}" in os.path.realpath(path)


def validate_preregistration(manifest: Mapping[str, Any], *, managed_path: bool) -> Dict[str, Any]:
    errors = []
    if not managed_path:
        errors.append("preregistration_path_not_managed")
    if manifest.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    if manifest.get("registered_before_candidate_implementation") is not True:
        errors.append("candidate_implementation_boundary_not_frozen")
    if tuple(manifest.get("arms", ())) != ARMS:
        errors.append("arms_do_not_match_frozen_protocol")
    if tuple(manifest.get("case_families", ())) != CASE_FAMILIES:
        errors.append("case_families_do_not_match_frozen_protocol")
    if tuple(manifest.get("ablation_shuffles", ())) != SHUFFLES:
        errors.append("shuffles_do_not_match_frozen_protocol")
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
    elif any(rule.get("direction") not in {"minimum", "maximum"} or not isinstance(rule.get("limit"), (int, float)) for rule in thresholds.values()):
        errors.append("threshold_rules_invalid")
    policy = manifest.get("leakage_and_resource_policy", {})
    if any(policy.get(key) is not True for key in REQUIRED_TRUE_POLICIES):
        errors.append("leakage_or_resource_policy_incomplete")
    learner = manifest.get("learner_visibility", {})
    forbidden = ("hidden_factor_id", "task_label", "human_concept_name", "offline_cluster_id", "expected_outcome", "source_partition_label")
    if any(name not in learner.get("forbidden_fields", ()) for name in forbidden):
        errors.append("learner_forbidden_fields_incomplete")
    if learner.get("global_all_pairs_search") is not False or learner.get("predeclared_hierarchy") is not False:
        errors.append("learner_global_or_predeclared_structure_allowed")
    prerequisites = manifest.get("prerequisites", {})
    required_prerequisites = (
        "phase30_protocol_fingerprint",
        "phase30_report_digest",
        "phase37_protocol_fingerprint",
        "phase37_report_sha256",
    )
    if any(not prerequisites.get(key) for key in required_prerequisites):
        errors.append("prerequisite_identity_missing")
    if prerequisites.get("phase30_mechanism_gate_passed") is not False or prerequisites.get("phase37_promotion_ready") is not False:
        errors.append("negative_prerequisite_status_not_bound")
    execution = manifest.get("execution_policy", {})
    expected = {
        "cpu_only": True,
        "gpu_required": False,
        "matrix_calculation": False,
        "global_gradient_backpropagation_required": False,
        "external_model": False,
        "default_off": True,
        "production_mutation": False,
        "durable_risa_mutation": False,
        "semantic_naming_changes_learning": False,
        "human_approval_required_for_integration": True,
    }
    if any(execution.get(key) != value for key, value in expected.items()):
        errors.append("execution_policy_mismatch")
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


__all__ = [
    "ARMS", "CASE_FAMILIES", "REQUIRED_BUDGETS", "REQUIRED_METRICS", "SCHEMA", "SHUFFLES",
    "build_registered_manifest", "compare_existing_registration", "is_managed_preregistration_path", "validate_preregistration",
]
