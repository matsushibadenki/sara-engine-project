"""Immutable preregistration for the Phase 34 retention-by-selection factorial."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping

from sara_engine.evaluation.phase33_preregistration import (
    compare_existing_registration,
    is_managed_preregistration_path,
    preregistration_fingerprint,
)


SCHEMA = "sara-phase34-memory-cache-factorial-preregistration-v1"
EXPERIMENT_ID = "phase34-memory-cache-factorial-observed-v1"
PARENT_EXPERIMENT_ID = "phase34-memory-cache-separation-observed-v1"
PARENT_PROTOCOL_FINGERPRINT = (
    "6b5f8394120936cab157c93c2917ea7bc3b75b98da74321e9be19f63aeaf8e06"
)
PARENT_REPORT_FINGERPRINT = (
    "185dca310a230e0612d5a78cc71f9eb4d9757732a92c66af31cf42995bca4897"
)
ARMS = (
    "recurrent_event_memory_control",
    "equal_retention_retrieve_all",
    "equal_retention_sparse_topk",
    "logarithmic_retention_retrieve_all",
    "logarithmic_retention_sparse_topk",
)
CASE_FAMILIES = (
    "retained_exact_target_pollution",
    "retained_shared_prefix_pollution",
    "retained_recency_decoy",
    "retained_topk_tie",
    "old_target_retention_pressure",
    "recent_target_retention_control",
    "logarithmic_merge_resolution",
    "revision_factorial_control",
    "contradiction_factorial_control",
    "stale_digest_factorial_control",
    "missing_target_factorial_control",
    "incompatible_group_factorial_control",
)
REPLICATE_SEEDS = (109, 227, 313, 431, 523)
REQUIRED_METRICS = (
    "selection_precision_main_effect",
    "selection_recall_noninferiority",
    "retention_old_recall_main_effect",
    "retention_recent_resolution_main_effect",
    "selection_retention_interaction_abs",
    "safety_integrity",
    "retained_set_identity",
    "state_bytes",
    "event_cost",
    "latency_ms",
    "deterministic_replay",
)


def validate_preregistration(
    manifest: Mapping[str, Any], *, managed_path: bool
) -> Dict[str, Any]:
    errors: List[str] = []
    exact = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": PARENT_EXPERIMENT_ID,
        "parent_protocol_fingerprint": PARENT_PROTOCOL_FINGERPRINT,
        "parent_report_fingerprint": PARENT_REPORT_FINGERPRINT,
        "registered_before_execution": True,
        "arms": list(ARMS),
        "case_families": list(CASE_FAMILIES),
        "replicate_seeds": list(REPLICATE_SEEDS),
        "replicates_per_condition": 5,
        "factorial_design": {
            "retention_factors": ["equal", "logarithmic"],
            "selection_factors": ["retrieve_all", "sparse_topk"],
            "control_arm_outside_factorial": True,
            "same_retained_set_within_retention_pair": True,
            "selection_runs_after_retention": True,
            "query_visible_during_retention": False,
            "query_visible_during_selection": True,
        },
        "budgets": {
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
        },
        "resource_accounting": {
            "same_generated_stream_across_arms_and_replays": True,
            "same_seed_across_arms": True,
            "retention_state_frozen_before_selection": True,
            "retained_set_digest_must_match_within_pair": True,
            "retention_bytes_reported_separately": True,
            "selection_bytes_reported_separately": True,
            "equal_total_state_ceiling_across_arms": True,
            "equal_tuning_allowance_across_arms": True,
            "query_aware_admission_allowed": False,
            "hidden_dense_summary_allowed": False,
            "unbounded_checkpoint_scan_allowed": False,
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
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            errors.append(f"frozen_factorial_mismatch:{key}")
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
    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(REQUIRED_METRICS):
        errors.append("thresholds_do_not_match_frozen_factorial")
    else:
        for metric in REQUIRED_METRICS:
            spec = thresholds.get(metric)
            if not isinstance(spec, Mapping) or spec.get("direction") not in {
                "minimum",
                "maximum",
            }:
                errors.append(f"invalid_threshold_spec:{metric}")
                continue
            limit = spec.get("limit")
            if (
                isinstance(limit, bool)
                or not isinstance(limit, (int, float))
                or not math.isfinite(float(limit))
            ):
                errors.append(f"invalid_threshold_limit:{metric}")
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
            "invalid Phase 34 factorial preregistration: "
            + "; ".join(validation["errors"])
        )
    return candidate


__all__ = [
    "ARMS",
    "CASE_FAMILIES",
    "EXPERIMENT_ID",
    "PARENT_EXPERIMENT_ID",
    "PARENT_PROTOCOL_FINGERPRINT",
    "PARENT_REPORT_FINGERPRINT",
    "REPLICATE_SEEDS",
    "REQUIRED_METRICS",
    "SCHEMA",
    "build_registered_manifest",
    "compare_existing_registration",
    "is_managed_preregistration_path",
    "validate_preregistration",
]
