"""Immutable preregistration for the Phase 34 cache-separation follow-up."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping

from sara_engine.evaluation.phase33_preregistration import (
    compare_existing_registration,
    is_managed_preregistration_path,
    preregistration_fingerprint,
)
from sara_engine.evaluation.phase34_memory_cache_preregistration import ARMS


SCHEMA = "sara-phase34-memory-cache-separation-preregistration-v1"
EXPERIMENT_ID = "phase34-memory-cache-separation-observed-v1"
PARENT_EXPERIMENT_ID = "phase34-memory-checkpoint-cache-observed-v1"
PARENT_PROTOCOL_FINGERPRINT = (
    "1e3d73dadd5d5ed49daf97617fc99403c8f6d2104143789afb9be142fe2b548e"
)
PARENT_REPORT_FINGERPRINT = (
    "996b974dc534b3c3ad8e4a68e5f7fd907f1840341d0b3f4a02d38351baed0429"
)
CASE_FAMILIES = (
    "old_target_after_overflow",
    "old_target_multi_resolution",
    "recent_target_fine_resolution",
    "boundary_burst_recent",
    "relevance_pollution",
    "recency_trap_topk",
    "deterministic_topk_tie",
    "revision_after_merge",
    "contradiction_after_merge",
    "incompatible_state_groups",
    "stale_digest_after_merge",
    "missing_target",
)
REPLICATE_SEEDS = (107, 223, 311, 419, 521)
EXPECTED_RELATIONS = (
    "logarithmic_over_equal",
    "equal_over_logarithmic",
    "topk_over_retrieve_all",
    "deterministic_tie",
    "safety_tie",
)
REQUIRED_METRICS = (
    "pairwise_separation_rate",
    "logarithmic_old_recall_delta",
    "topk_pollution_precision_delta",
    "equal_recent_resolution_delta",
    "safety_integrity",
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
        "expected_relations": list(EXPECTED_RELATIONS),
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
            "equal_source_events_across_arms": True,
            "equal_state_byte_ceiling_across_arms": True,
            "equal_tuning_allowance_across_arms": True,
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
            errors.append(f"frozen_followup_mismatch:{key}")
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
        errors.append("thresholds_do_not_match_frozen_followup")
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
            "invalid Phase 34 separation preregistration: "
            + "; ".join(validation["errors"])
        )
    return candidate


__all__ = [
    "CASE_FAMILIES",
    "EXPECTED_RELATIONS",
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
