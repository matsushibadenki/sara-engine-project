"""Immutable contract for the Phase 34 semantic delayed-recall workload."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping

from sara_engine.evaluation.phase33_preregistration import (
    compare_existing_registration,
    is_managed_preregistration_path,
    preregistration_fingerprint,
)
from sara_engine.evaluation.phase34_factorial_preregistration import ARMS


SCHEMA = "sara-phase34-semantic-delayed-recall-preregistration-v1"
EXPERIMENT_ID = "phase34-semantic-delayed-recall-v1"
PARENT_PROTOCOL_FINGERPRINT = (
    "7e4ce13ff7e0aded273a657133263ebf9c52e7d5285c3d2a341a87233bd44ec1"
)
PARENT_REPORT_FINGERPRINT = (
    "a6dacc13596b94d1bb2cf42780502c3bcb5bcc2b4f6ba05e854facfd4cbbfea4"
)
REVIEW_REQUEST_FINGERPRINT = (
    "e00730883aced609ed84947f7e6fb36a0bae91341ee2f760d3f0a245a6024e0c"
)
REVIEW_LEDGER_FINGERPRINT = (
    "ec3ffe44777f332bcc07da428a0385a9904bc98eb403ff627db8dc89711b124f"
)
REVIEW_GATE_REPORT_FINGERPRINT = (
    "687bb140373fc0d2e695b6b2a1aff403df2dc505b292f7063fb48e7a60a58aac"
)
COMPARISON_PACKET_FINGERPRINT = (
    "ae6fe43432f09e019d70ce12323c23dc9063c49ac6820ce16ee15b8bb81d8b3e"
)
REVIEW_SUPPORT_SNAPSHOT_FINGERPRINT = (
    "32b115d179bd495717ebabd139cfa5d1eb0dca5dd7fe2928c7f1b26dd224e495"
)

TARGET_IDS = (
    "arch-migration-ietf-001",
    "arch-migration-ietf-002",
    "arch-migration-ietf-003",
    "arch-migration-python-001",
    "arch-migration-python-002",
    "arch-migration-python-003",
)
LANGUAGES = ("en", "ja", "zh-Hans")
HORIZONS = (10, 30, 100)
CASE_FAMILIES = (
    "semantic_paraphrase_recall",
    "lexical_overlap_abstention",
    "revision_replacement",
    "contradiction_abstention",
    "missing_evidence_abstention",
)
REPLICATE_SEEDS = (127, 229, 317, 433, 541)
CASE_COUNT = (
    len(TARGET_IDS) * len(LANGUAGES) * len(HORIZONS) * len(CASE_FAMILIES)
)

BUDGETS = {
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

THRESHOLDS = {
    "semantic_paraphrase_macro_accuracy": {"direction": "minimum", "limit": 0.75},
    "best_checkpoint_minus_control": {"direction": "minimum", "limit": 0.10},
    "lexical_overlap_abstention": {"direction": "minimum", "limit": 0.90},
    "revision_uptake": {"direction": "minimum", "limit": 0.90},
    "contradiction_abstention": {"direction": "minimum", "limit": 1.0},
    "missing_evidence_abstention": {"direction": "minimum", "limit": 1.0},
    "worst_language_recall": {"direction": "minimum", "limit": 0.60},
    "source_traceability": {"direction": "minimum", "limit": 1.0},
    "retained_set_identity": {"direction": "minimum", "limit": 1.0},
    "state_bytes": {"direction": "maximum", "limit": 8192},
    "event_cost": {"direction": "maximum", "limit": 256},
    "latency_ms": {"direction": "maximum", "limit": 50},
    "deterministic_replay": {"direction": "minimum", "limit": 1.0},
}

CLAIM_BOUNDARIES = {
    "human_reviewed_source_alignment_only": True,
    "semantic_probe_wording_human_reviewed": False,
    "independent_semantic_scope": "six_source_bound_propositions",
    "synthetic_safety_controls_are_independent_evidence": False,
    "general_language_understanding_claim_allowed": False,
    "general_semantic_memory_claim_allowed": False,
    "ann_parity_claim_allowed": False,
    "physical_energy_claim_allowed": False,
}

EXECUTION_POLICY = {
    "registered_before_semantic_adapter_implementation": True,
    "reuse_parent_arms_unchanged": True,
    "selector_retuning_allowed": False,
    "query_aware_retention_allowed": False,
    "answer_or_expected_decision_visible_to_candidate": False,
    "external_pretrained_embedding_allowed": False,
    "external_model_allowed": False,
    "learned_router_allowed": False,
    "backpropagation_allowed": False,
    "matrix_calculation_allowed": False,
    "gpu_required": False,
    "cpu_only": True,
    "default_off": True,
    "production_mutation": False,
    "durable_admission": False,
}


def _is_hex_digest(value: Any) -> bool:
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
        "parent_report_fingerprint": PARENT_REPORT_FINGERPRINT,
        "review_request_fingerprint": REVIEW_REQUEST_FINGERPRINT,
        "review_ledger_fingerprint": REVIEW_LEDGER_FINGERPRINT,
        "review_gate_report_fingerprint": REVIEW_GATE_REPORT_FINGERPRINT,
        "comparison_packet_fingerprint": COMPARISON_PACKET_FINGERPRINT,
        "review_support_snapshot_fingerprint": REVIEW_SUPPORT_SNAPSHOT_FINGERPRINT,
        "registered_before_execution": True,
        "registered_before_semantic_adapter_implementation": True,
        "source_target_ids": list(TARGET_IDS),
        "languages": list(LANGUAGES),
        "horizons": list(HORIZONS),
        "case_families": list(CASE_FAMILIES),
        "case_count": CASE_COUNT,
        "arms": list(ARMS),
        "replicate_seeds": list(REPLICATE_SEEDS),
        "replicates_per_condition": len(REPLICATE_SEEDS),
        "budgets": BUDGETS,
        "thresholds": THRESHOLDS,
        "claim_boundaries": CLAIM_BOUNDARIES,
        "execution_policy": EXECUTION_POLICY,
        "evaluation_contract": {
            "candidate_visible_fields": [
                "case_id",
                "record_id",
                "language",
                "horizon",
                "family",
                "query_text",
                "control_mode",
            ],
            "evaluator_only_fields": [
                "expected_decision",
                "expected_proposition_id",
                "source_hash",
                "source_ref",
                "source_revision",
                "independent_semantic_evidence",
                "synthetic_control",
            ],
            "macro_average_axes": ["record_id", "language", "horizon"],
            "exact_identity_score_is_semantic_score": False,
            "token_overlap_is_semantic_score": False,
            "source_trace_required_for_non_abstaining_answer": True,
        },
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            errors.append(f"frozen_semantic_workload_mismatch:{key}")
    if not managed_path:
        errors.append("preregistration_path_not_managed")
    for field in ("fixture_fingerprint", "environment_fingerprint"):
        if not _is_hex_digest(manifest.get(field)):
            errors.append(f"invalid_{field}")
    for metric, spec in THRESHOLDS.items():
        actual = manifest.get("thresholds", {}).get(metric)
        if actual != spec:
            errors.append(f"invalid_threshold_spec:{metric}")
            continue
        limit = actual.get("limit")
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
            "invalid Phase 34 semantic delayed-recall preregistration: "
            + "; ".join(validation["errors"])
        )
    return candidate


__all__ = [
    "BUDGETS",
    "CASE_COUNT",
    "CASE_FAMILIES",
    "CLAIM_BOUNDARIES",
    "COMPARISON_PACKET_FINGERPRINT",
    "EXECUTION_POLICY",
    "EXPERIMENT_ID",
    "HORIZONS",
    "LANGUAGES",
    "PARENT_PROTOCOL_FINGERPRINT",
    "PARENT_REPORT_FINGERPRINT",
    "REPLICATE_SEEDS",
    "REVIEW_GATE_REPORT_FINGERPRINT",
    "REVIEW_LEDGER_FINGERPRINT",
    "REVIEW_REQUEST_FINGERPRINT",
    "REVIEW_SUPPORT_SNAPSHOT_FINGERPRINT",
    "SCHEMA",
    "TARGET_IDS",
    "THRESHOLDS",
    "build_registered_manifest",
    "compare_existing_registration",
    "is_managed_preregistration_path",
    "validate_preregistration",
]
