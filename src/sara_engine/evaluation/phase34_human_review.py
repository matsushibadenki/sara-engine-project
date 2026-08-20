"""Bound and evaluate human source-alignment decisions for Phase 34 excerpts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence


REQUEST_SCHEMA = "sara-phase34-transcribed-excerpt-human-review-request-v1"
LEDGER_SCHEMA = "sara-phase34-transcribed-excerpt-human-review-decisions-v1"
GATE_SCHEMA = "sara-phase34-transcribed-excerpt-human-review-gate-v1"
TARGET_COUNT = 6
ALIGNMENT_DECISIONS = ("aligned", "misaligned", "unresolved")


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_hex_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _has_timezone(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def validate_request(request: Mapping[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    targets = request.get("targets")
    if request.get("schema") != REQUEST_SCHEMA:
        errors.append("request_schema_mismatch")
    if request.get("observed_only") is not True:
        errors.append("request_must_remain_observed_only")
    if request.get("review_complete") is not False:
        errors.append("request_must_remain_pending")
    if request.get("promotion_ready") is not False:
        errors.append("request_cannot_be_promotion_ready")
    if request.get("target_count") != TARGET_COUNT:
        errors.append("request_target_count_mismatch")
    if not isinstance(targets, list) or len(targets) != TARGET_COUNT:
        errors.append("request_targets_mismatch")
        targets = []

    seen_ids = set()
    for target in targets:
        if not isinstance(target, Mapping):
            errors.append("request_target_not_object")
            continue
        record_id = str(target.get("record_id", ""))
        if not record_id or record_id in seen_ids:
            errors.append(f"request_record_id_invalid:{record_id}")
        seen_ids.add(record_id)
        if not _is_hex_digest(target.get("source_hash")):
            errors.append(f"request_source_hash_invalid:{record_id}")
        if not str(target.get("source_ref", "")):
            errors.append(f"request_source_ref_missing:{record_id}")
        if target.get("review_status") != "pending_human_review":
            errors.append(f"request_status_mutated:{record_id}")
        required = target.get("required_review")
        if not isinstance(required, Mapping):
            errors.append(f"request_required_review_missing:{record_id}")
        elif required.get("alignment_decision") != "pending":
            errors.append(f"request_decision_mutated:{record_id}")

    return {
        "valid": not errors,
        "errors": errors,
        "request_fingerprint": canonical_digest(dict(request)),
        "target_ids": sorted(seen_ids),
    }


def build_empty_ledger(request: Mapping[str, Any]) -> Dict[str, Any]:
    validation = validate_request(request)
    if not validation["valid"]:
        raise ValueError("invalid review request: " + "; ".join(validation["errors"]))
    return {
        "schema": LEDGER_SCHEMA,
        "observed_only": True,
        "request_fingerprint": validation["request_fingerprint"],
        "target_count": TARGET_COUNT,
        "decisions": [],
        "mutation_policy": {
            "historical_raw_rows_mutated": False,
            "executed_v2_fingerprint_mutated": False,
            "request_mutated": False,
            "automation_may_attest_human_review": False,
        },
    }


def validate_ledger(
    request: Mapping[str, Any], ledger: Mapping[str, Any]
) -> Dict[str, Any]:
    request_validation = validate_request(request)
    errors = list(request_validation["errors"])
    if ledger.get("schema") != LEDGER_SCHEMA:
        errors.append("ledger_schema_mismatch")
    if ledger.get("observed_only") is not True:
        errors.append("ledger_must_remain_observed_only")
    if ledger.get("request_fingerprint") != request_validation["request_fingerprint"]:
        errors.append("ledger_request_fingerprint_mismatch")
    if ledger.get("target_count") != TARGET_COUNT:
        errors.append("ledger_target_count_mismatch")
    expected_policy = {
        "historical_raw_rows_mutated": False,
        "executed_v2_fingerprint_mutated": False,
        "request_mutated": False,
        "automation_may_attest_human_review": False,
    }
    if ledger.get("mutation_policy") != expected_policy:
        errors.append("ledger_mutation_policy_mismatch")

    targets = {
        str(target.get("record_id", "")): target
        for target in request.get("targets", [])
        if isinstance(target, Mapping)
    }
    decisions = ledger.get("decisions")
    if not isinstance(decisions, list):
        errors.append("ledger_decisions_not_list")
        decisions = []
    seen_ids = set()
    for decision in decisions:
        if not isinstance(decision, Mapping):
            errors.append("ledger_decision_not_object")
            continue
        record_id = str(decision.get("record_id", ""))
        target = targets.get(record_id)
        if target is None or record_id in seen_ids:
            errors.append(f"ledger_record_id_invalid:{record_id}")
            continue
        seen_ids.add(record_id)
        if decision.get("request_fingerprint") != request_validation["request_fingerprint"]:
            errors.append(f"decision_request_fingerprint_mismatch:{record_id}")
        if decision.get("stored_excerpt_hash") != target.get("source_hash"):
            errors.append(f"decision_excerpt_hash_mismatch:{record_id}")
        if decision.get("source_ref") != target.get("source_ref"):
            errors.append(f"decision_source_ref_mismatch:{record_id}")
        if decision.get("alignment_decision") not in ALIGNMENT_DECISIONS:
            errors.append(f"decision_alignment_invalid:{record_id}")
        distortion = decision.get("semantic_omission_or_distortion_found")
        if distortion not in (True, False, None):
            errors.append(f"decision_distortion_invalid:{record_id}")
        if decision.get("alignment_decision") == "aligned" and distortion is not False:
            errors.append(f"aligned_decision_requires_no_distortion:{record_id}")
        if decision.get("alignment_decision") == "misaligned" and distortion is not True:
            errors.append(f"misaligned_decision_requires_distortion:{record_id}")
        if not str(decision.get("authoritative_section_locator", "")).strip():
            errors.append(f"decision_locator_missing:{record_id}")
        if not _is_hex_digest(decision.get("authoritative_text_hash")):
            errors.append(f"decision_authoritative_hash_invalid:{record_id}")
        if not str(decision.get("reviewer", "")).strip():
            errors.append(f"decision_reviewer_missing:{record_id}")
        if not _has_timezone(str(decision.get("reviewed_at", ""))):
            errors.append(f"decision_review_time_invalid:{record_id}")
        if decision.get("human_attestation") is not True:
            errors.append(f"decision_human_attestation_missing:{record_id}")
        digest_input = dict(decision)
        declared_digest = digest_input.pop("decision_digest", None)
        if declared_digest != canonical_digest(digest_input):
            errors.append(f"decision_digest_mismatch:{record_id}")

    return {
        "valid": not errors,
        "errors": errors,
        "decision_count": len(decisions),
        "decided_ids": sorted(seen_ids),
        "request_fingerprint": request_validation["request_fingerprint"],
    }


def record_decision(
    request: Mapping[str, Any],
    ledger: Optional[Mapping[str, Any]],
    *,
    record_id: str,
    authoritative_section_locator: str,
    authoritative_text_hash: str,
    alignment_decision: str,
    semantic_omission_or_distortion_found: Optional[bool],
    reviewer: str,
    reviewed_at: str,
    notes: str = "",
    human_attestation: bool,
    replace_existing: bool = False,
) -> Dict[str, Any]:
    base = dict(ledger) if ledger is not None else build_empty_ledger(request)
    validation = validate_ledger(request, base)
    if not validation["valid"]:
        raise ValueError("invalid review ledger: " + "; ".join(validation["errors"]))
    targets = {
        str(target.get("record_id", "")): target
        for target in request.get("targets", [])
        if isinstance(target, Mapping)
    }
    target = targets.get(str(record_id))
    if target is None:
        raise ValueError(f"unknown review target: {record_id}")

    decision_input: Dict[str, Any] = {
        "request_fingerprint": validation["request_fingerprint"],
        "record_id": str(record_id),
        "source_ref": str(target.get("source_ref", "")),
        "stored_excerpt_hash": str(target.get("source_hash", "")),
        "authoritative_section_locator": str(authoritative_section_locator).strip(),
        "authoritative_text_hash": str(authoritative_text_hash).strip().lower(),
        "alignment_decision": str(alignment_decision),
        "semantic_omission_or_distortion_found": semantic_omission_or_distortion_found,
        "reviewer": str(reviewer).strip(),
        "reviewed_at": str(reviewed_at).strip(),
        "notes": str(notes).strip(),
        "human_attestation": bool(human_attestation),
    }
    decision = dict(decision_input)
    decision["decision_digest"] = canonical_digest(decision_input)

    existing = {
        str(item.get("record_id", "")): dict(item)
        for item in base.get("decisions", [])
        if isinstance(item, Mapping)
    }
    previous = existing.get(str(record_id))
    if previous is not None and previous != decision and not replace_existing:
        raise ValueError(
            f"review decision already exists for {record_id}; use explicit replacement"
        )
    existing[str(record_id)] = decision
    updated = dict(base)
    updated["decisions"] = [existing[key] for key in sorted(existing)]
    updated_validation = validate_ledger(request, updated)
    if not updated_validation["valid"]:
        raise ValueError(
            "invalid review decision: " + "; ".join(updated_validation["errors"])
        )
    return updated


def evaluate_review_gate(
    request: Mapping[str, Any], ledger: Mapping[str, Any]
) -> Dict[str, Any]:
    validation = validate_ledger(request, ledger)
    decisions = [
        dict(item) for item in ledger.get("decisions", []) if isinstance(item, Mapping)
    ]
    aligned = [
        item
        for item in decisions
        if item.get("alignment_decision") == "aligned"
        and item.get("semantic_omission_or_distortion_found") is False
    ]
    misaligned = [
        item for item in decisions if item.get("alignment_decision") == "misaligned"
    ]
    unresolved = [
        item for item in decisions if item.get("alignment_decision") == "unresolved"
    ]
    final_count = len(aligned) + len(misaligned)
    review_complete = validation["valid"] and final_count == TARGET_COUNT
    review_gate_passed = review_complete and len(aligned) == TARGET_COUNT
    if not validation["valid"]:
        next_action = "repair_invalid_decision_ledger"
    elif len(decisions) < TARGET_COUNT or unresolved:
        next_action = "complete_remaining_human_source_alignment_reviews"
    elif misaligned:
        next_action = "resolve_misaligned_sources_under_new_preregistration"
    else:
        next_action = "preregister_semantic_delayed_recall_workload"
    report = {
        "schema": GATE_SCHEMA,
        "observed_only": True,
        "request_fingerprint": validation["request_fingerprint"],
        "ledger_fingerprint": canonical_digest(dict(ledger)),
        "target_count": TARGET_COUNT,
        "decision_count": len(decisions),
        "aligned_count": len(aligned),
        "misaligned_count": len(misaligned),
        "unresolved_count": len(unresolved),
        "pending_count": max(0, TARGET_COUNT - len(decisions)),
        "review_complete": review_complete,
        "review_gate_passed": review_gate_passed,
        "semantic_delayed_recall_preregistration_ready": review_gate_passed,
        "promotion_ready": False,
        "validation": validation,
        "aligned_ids": sorted(str(item.get("record_id", "")) for item in aligned),
        "misaligned_ids": sorted(str(item.get("record_id", "")) for item in misaligned),
        "unresolved_ids": sorted(str(item.get("record_id", "")) for item in unresolved),
        "next_action": next_action,
        "claim_boundary": (
            "Human source alignment can authorize semantic-workload preregistration only; "
            "it cannot promote checkpoint caching or rewrite executed evidence."
        ),
    }
    report["report_fingerprint"] = canonical_digest(report)
    return report


__all__ = [
    "ALIGNMENT_DECISIONS",
    "GATE_SCHEMA",
    "LEDGER_SCHEMA",
    "REQUEST_SCHEMA",
    "TARGET_COUNT",
    "build_empty_ledger",
    "canonical_digest",
    "evaluate_review_gate",
    "record_decision",
    "validate_ledger",
    "validate_request",
]
