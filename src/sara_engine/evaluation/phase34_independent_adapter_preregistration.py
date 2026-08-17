"""Immutable contract for the Phase 34 independent source-identity adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from sara_engine.evaluation.phase33_preregistration import (
    compare_existing_registration,
    is_managed_preregistration_path,
    preregistration_fingerprint,
)
from sara_engine.evaluation.phase34_factorial_preregistration import (
    ARMS,
    REPLICATE_SEEDS,
)


SCHEMA = "sara-phase34-memory-cache-factorial-independent-adapter-preregistration-v2"
EXPERIMENT_ID = "phase34-memory-cache-factorial-independent-adapter-v2"
PARENT_PROTOCOL_FINGERPRINT = (
    "3eca1c4aec95c374be3c0ba9637df93e0df852bcd75114b3253a9d5494eb47ed"
)
SOURCE_DOMAINS = ("docs.python.org", "www.rfc-editor.org")
HORIZONS = (10, 30, 100)
CASE_FAMILIES = (
    "exact_identity_selection",
    "signature_decoy_selection",
    "old_identity_retention",
    "recent_identity_control",
    "missing_identity_control",
    "stale_digest_control",
    "contradiction_control",
)
CASE_COUNT = len(SOURCE_DOMAINS) * len(HORIZONS) * len(CASE_FAMILIES)


CASE_GENERATION = {
    "source_order": "migration_horizon_index_ascending",
    "source_prefix_inclusive_horizons": list(HORIZONS),
    "stream_width_rule": "min(16,horizon_plus_one)",
    "stream_position_rule": "floor(i*horizon/(stream_width_minus_one))",
    "exact_selection_target_position": "fourth_from_end",
    "old_retention_target_position": "first",
    "recent_control_target_position": "last",
    "signature_decoy_rule": "maximum_sparse_jaccard_then_material_hash",
    "query_representation": "exact_material_hash_identity",
    "missing_query_rule": "sha256(domain|horizon|missing_identity_control)",
    "case_families": list(CASE_FAMILIES),
    "cases_per_domain_horizon": len(CASE_FAMILIES),
    "case_count": CASE_COUNT,
    "query_visible_during_case_generation": False,
    "arm_results_visible_during_case_generation": False,
}

CLAIM_BOUNDARIES = {
    "exact_source_identity_recall_only": True,
    "semantic_accuracy_claim_allowed": False,
    "language_understanding_claim_allowed": False,
    "ann_parity_claim_allowed": False,
    "physical_energy_claim_allowed": False,
    "synthetic_negative_controls_are_independent_evidence": False,
}

EXECUTION_POLICY = {
    "reuse_parent_arms_unchanged": True,
    "reuse_parent_budgets_unchanged": True,
    "reuse_parent_thresholds_unchanged": True,
    "selector_retuning_allowed": False,
    "query_aware_retention_allowed": False,
    "learned_router_allowed": False,
    "backpropagation_allowed": False,
    "matrix_calculation_allowed": False,
    "gpu_required": False,
    "cpu_only": True,
    "default_off": True,
    "production_mutation": False,
    "durable_admission": False,
}

PARENT_BUDGETS = {
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
}

PARENT_THRESHOLDS = {
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
}


def _hex_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_preregistration(
    manifest: Mapping[str, Any], *, managed_path: bool
) -> Dict[str, Any]:
    errors: List[str] = []
    exact = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "parent_protocol_fingerprint": PARENT_PROTOCOL_FINGERPRINT,
        "registered_before_adapter_execution": True,
        "source_domains": list(SOURCE_DOMAINS),
        "required_horizons": list(HORIZONS),
        "arms": list(ARMS),
        "replicate_seeds": list(REPLICATE_SEEDS),
        "case_generation": CASE_GENERATION,
        "budgets": PARENT_BUDGETS,
        "thresholds": PARENT_THRESHOLDS,
        "claim_boundaries": CLAIM_BOUNDARIES,
        "execution_policy": EXECUTION_POLICY,
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            errors.append(f"frozen_independent_adapter_mismatch:{key}")
    if not managed_path:
        errors.append("preregistration_path_not_managed")
    for field in (
        "parent_factorial_report_fingerprint",
        "source_manifest_fingerprint",
        "external_gate_fingerprint",
        "readiness_gate_fingerprint",
        "case_plan_fingerprint",
        "environment_fingerprint",
    ):
        if not _hex_digest(manifest.get(field)):
            errors.append(f"invalid_{field}")
    snapshot = manifest.get("source_snapshot")
    if snapshot != {
        "record_count": 202,
        "records_per_domain": {domain: 101 for domain in SOURCE_DOMAINS},
        "horizon_span_per_domain": {domain: 100 for domain in SOURCE_DOMAINS},
        "unique_material_hash_count": 202,
        "unique_source_ref_count": 202,
        "observed_only": True,
        "compliance_level": "allow",
    }:
        errors.append("frozen_independent_adapter_mismatch:source_snapshot")
    if manifest.get("case_plan_count") != CASE_COUNT:
        errors.append("invalid_case_plan_count")
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
    draft: Mapping[str, Any], *, managed_path: bool
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
            "invalid Phase 34 independent adapter preregistration: "
            + "; ".join(validation["errors"])
        )
    return candidate


__all__ = [
    "CASE_COUNT",
    "CASE_FAMILIES",
    "CASE_GENERATION",
    "CLAIM_BOUNDARIES",
    "EXECUTION_POLICY",
    "EXPERIMENT_ID",
    "HORIZONS",
    "PARENT_PROTOCOL_FINGERPRINT",
    "PARENT_BUDGETS",
    "PARENT_THRESHOLDS",
    "SCHEMA",
    "SOURCE_DOMAINS",
    "build_registered_manifest",
    "compare_existing_registration",
    "is_managed_preregistration_path",
    "validate_preregistration",
]
