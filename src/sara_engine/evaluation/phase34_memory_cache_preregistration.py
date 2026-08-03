"""Immutable preregistration for bounded sparse memory checkpoint caching."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping

from sara_engine.evaluation.phase33_preregistration import (
    compare_existing_registration,
    is_managed_preregistration_path,
    preregistration_fingerprint,
)


SCHEMA = "sara-phase34-memory-checkpoint-cache-preregistration-v1"
EXPERIMENT_ID = "phase34-memory-checkpoint-cache-observed-v1"
ARMS = (
    "recurrent_event_memory_control",
    "equal_segment_retrieve_all",
    "logarithmic_segments_retrieve_all",
    "equal_segment_sparse_topk",
)
CASE_FAMILIES = (
    "delayed_key_value_recall",
    "long_irrelevant_interval",
    "revised_value",
    "contradiction",
    "source_replacement",
    "duplicate_segment",
    "near_duplicate_segment",
    "missing_segment",
    "stale_runtime_digest",
    "stale_schema_digest",
    "reordered_replay",
    "cache_overflow",
    "long_tail_pollution",
    "exact_verified_checkpoint",
    "irrelevant_high_recency",
    "topk_tie",
)
REQUIRED_BUDGETS = (
    "source_events_per_case",
    "max_checkpoints",
    "max_selected_checkpoints",
    "max_summary_ids_per_checkpoint",
    "max_total_state_bytes",
    "max_local_interactions_per_case",
    "max_latency_ms",
    "max_merges_per_event",
    "tuning_trials_per_arm",
    "restart_count_per_arm",
)
REQUIRED_METRICS = (
    "delayed_recall_quality",
    "revision_uptake",
    "contradiction_rejection",
    "abstention_integrity",
    "selection_precision",
    "selection_recall",
    "useful_checkpoint_rate",
    "retained_temporal_resolution",
    "state_bytes",
    "event_cost",
    "latency_ms",
    "deterministic_replay",
)


def validate_preregistration(
    manifest: Mapping[str, Any], *, managed_path: bool
) -> Dict[str, Any]:
    errors: List[str] = []
    if manifest.get("schema") != SCHEMA:
        errors.append("unsupported_memory_cache_preregistration_schema")
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id_does_not_match_frozen_phase34")
    if manifest.get("registered_before_execution") is not True:
        errors.append("not_registered_before_execution")
    if not managed_path:
        errors.append("preregistration_path_not_managed")
    for field in ("fixture_fingerprint", "environment_fingerprint"):
        value = manifest.get(field)
        if not (isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)):
            errors.append(f"invalid_{field}")
    if tuple(manifest.get("arms", ())) != ARMS:
        errors.append("arms_do_not_match_frozen_phase34")
    if tuple(manifest.get("case_families", ())) != CASE_FAMILIES:
        errors.append("case_families_do_not_match_frozen_phase34")

    segmentation = manifest.get("segmentation")
    required_segmentation = {
        "semantic_boundaries_required": True,
        "equal_segment_event_span": 4,
        "logarithmic_retention_tiers": [1, 2, 4, 8],
        "merge_order": "oldest_first",
        "preserve_provenance": True,
        "parameter_averaging": False,
    }
    if not isinstance(segmentation, Mapping) or dict(segmentation) != required_segmentation:
        errors.append("segmentation_contract_mismatch")

    selection = manifest.get("selection")
    required_selection = {
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
    }
    if not isinstance(selection, Mapping) or dict(selection) != required_selection:
        errors.append("selection_contract_mismatch")

    budgets = manifest.get("budgets")
    required_budget_values = {
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
    }
    if not isinstance(budgets, Mapping) or dict(budgets) != required_budget_values:
        errors.append("budgets_do_not_match_frozen_phase34")

    accounting = manifest.get("resource_accounting")
    required_accounting = {
        "equal_source_events_across_arms": True,
        "equal_state_byte_ceiling_across_arms": True,
        "equal_tuning_allowance_across_arms": True,
        "cache_count_bounded_independent_of_sequence_length": True,
        "hidden_dense_summary_allowed": False,
        "unbounded_checkpoint_scan_allowed": False,
    }
    if not isinstance(accounting, Mapping) or dict(accounting) != required_accounting:
        errors.append("resource_accounting_contract_mismatch")

    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(REQUIRED_METRICS):
        errors.append("thresholds_do_not_match_frozen_phase34")
    else:
        for metric in REQUIRED_METRICS:
            spec = thresholds.get(metric)
            if not isinstance(spec, Mapping) or spec.get("direction") not in {"minimum", "maximum"}:
                errors.append(f"invalid_threshold_spec:{metric}")
                continue
            limit = spec.get("limit")
            if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(float(limit)):
                errors.append(f"invalid_threshold_limit:{metric}")

    policy = manifest.get("execution_policy")
    required_policy = {
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
    }
    if not isinstance(policy, Mapping) or dict(policy) != required_policy:
        errors.append("execution_policy_does_not_match_sara_boundaries")

    try:
        computed = preregistration_fingerprint(manifest)
    except (TypeError, ValueError):
        computed = None
        errors.append("preregistration_is_not_canonical_json")
    if manifest.get("protocol_fingerprint") != computed:
        errors.append("protocol_fingerprint_mismatch")
    return {"valid": not errors, "managed_path": managed_path, "computed_fingerprint": computed, "declared_fingerprint": manifest.get("protocol_fingerprint"), "errors": errors}


def build_registered_manifest(draft: Mapping[str, Any], *, managed_path: bool) -> Dict[str, Any]:
    candidate = dict(draft)
    candidate.pop("protocol_fingerprint", None)
    try:
        candidate["protocol_fingerprint"] = preregistration_fingerprint(candidate)
    except (TypeError, ValueError):
        candidate["protocol_fingerprint"] = ""
    validation = validate_preregistration(candidate, managed_path=managed_path)
    if not validation["valid"]:
        raise ValueError("invalid Phase 34 memory-cache preregistration: " + "; ".join(validation["errors"]))
    return candidate


__all__ = ["ARMS", "CASE_FAMILIES", "EXPERIMENT_ID", "REQUIRED_BUDGETS", "REQUIRED_METRICS", "SCHEMA", "build_registered_manifest", "compare_existing_registration", "is_managed_preregistration_path", "validate_preregistration"]
