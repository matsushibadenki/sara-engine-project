from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, Tuple


TRACE_SCHEMA = "sara-portable-decision-trace-v1"
SUPPORTED_SUBSYSTEMS = frozenset(
    {
        "event_memory",
        "event_memory_retrieval",
        "event_memory_eviction",
        "event_memory_revision",
        "risa_proposal",
        "predictive_feedback",
    }
)
MAX_DECISIONS = 10_000
MAX_EVIDENCE_IDS = 32


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f"{field} must be a non-empty string up to 256 characters")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _index(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def decide(record: Mapping[str, Any]) -> str:
    subsystem = str(record["subsystem"])
    verified = bool(record["verified"])
    contradiction = bool(record["contradiction"])
    stale = bool(record["stale"])
    capacity_available = bool(record["capacity_available"])
    prediction_match = bool(record["prediction_match"])
    support_count = int(record["support_count"])
    if subsystem == "event_memory":
        if not verified:
            return "reject_unverified"
        if contradiction:
            return "reject_contradiction"
        if stale:
            return "reject_stale"
        if not capacity_available:
            return "abstain_capacity"
        return "admit"
    if subsystem == "risa_proposal":
        if not verified:
            return "reject_unverified"
        if contradiction:
            return "freeze_contradiction"
        if support_count == 0:
            return "reject_missing_support"
        return "propose"
    if subsystem == "event_memory_retrieval":
        if not verified:
            return "abstain_unverified"
        if contradiction:
            return "reject_contradiction"
        if stale:
            return "reject_stale"
        if support_count == 0:
            return "abstain_missing_support"
        return "retrieve"
    if subsystem == "event_memory_eviction":
        if not verified:
            return "reject_unverified"
        if contradiction:
            return "retain_protected"
        if stale:
            return "evict_stale"
        if not capacity_available:
            return "evict_capacity"
        return "retain"
    if subsystem == "event_memory_revision":
        if not verified:
            return "reject_unverified"
        if contradiction:
            return "freeze_revision"
        if support_count == 0:
            return "abstain_missing_support"
        return "retain_revision" if prediction_match else "replace_revision"
    if subsystem == "predictive_feedback":
        if not verified:
            return "abstain_unverified"
        if contradiction:
            return "freeze_contradiction"
        if support_count == 0:
            return "abstain_missing_support"
        return "retain_prediction" if prediction_match else "emit_correction"
    raise ValueError(f"unsupported subsystem: {subsystem}")


def canonicalize_decisions(
    records: Iterable[Mapping[str, Any]], *, max_decisions: int = MAX_DECISIONS
) -> Tuple[Dict[str, Any], ...]:
    if isinstance(max_decisions, bool) or not isinstance(max_decisions, int) or max_decisions < 1:
        raise ValueError("max_decisions must be a positive integer")
    normalized = []
    seen = set()
    required = {
        "decision_id", "sequence", "subsystem", "subject_id", "evidence_ids",
        "verified", "contradiction", "stale", "capacity_available",
        "prediction_match", "support_count",
    }
    for index, record in enumerate(records):
        if index >= max_decisions:
            raise ValueError(f"decision count exceeds max_decisions={max_decisions}")
        if not isinstance(record, Mapping):
            raise ValueError(f"decision at index {index} must be a mapping")
        if set(record) != required:
            raise ValueError("decision fields do not match the portable schema")
        decision_id = _text(record["decision_id"], "decision_id")
        if decision_id in seen:
            raise ValueError(f"duplicate decision_id: {decision_id}")
        seen.add(decision_id)
        subsystem = _text(record["subsystem"], "subsystem")
        if subsystem not in SUPPORTED_SUBSYSTEMS:
            raise ValueError(f"unsupported subsystem: {subsystem}")
        evidence = record["evidence_ids"]
        if not isinstance(evidence, (list, tuple)) or len(evidence) > MAX_EVIDENCE_IDS:
            raise ValueError("evidence_ids must be a bounded list or tuple")
        row = {
            "capacity_available": _boolean(record["capacity_available"], "capacity_available"),
            "contradiction": _boolean(record["contradiction"], "contradiction"),
            "decision_id": decision_id,
            "evidence_ids": sorted({_text(item, "evidence_id") for item in evidence}),
            "prediction_match": _boolean(record["prediction_match"], "prediction_match"),
            "sequence": _index(record["sequence"], "sequence"),
            "stale": _boolean(record["stale"], "stale"),
            "subject_id": _text(record["subject_id"], "subject_id"),
            "subsystem": subsystem,
            "support_count": _index(record["support_count"], "support_count"),
            "verified": _boolean(record["verified"], "verified"),
        }
        row["decision"] = decide(row)
        normalized.append(row)
    return tuple(sorted(normalized, key=lambda row: (row["sequence"], row["decision_id"])))


def canonical_decision_json(records: Iterable[Mapping[str, Any]]) -> str:
    return json.dumps(
        canonicalize_decisions(records), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def decision_trace_digest(records: Iterable[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_decision_json(records).encode("utf-8")).hexdigest()


def adapt_event_memory_admission(result: Any, *, sequence: int) -> Dict[str, Any]:
    trace = dict(getattr(result, "trace", {}) or {})
    receipt = dict(trace.get("verification_receipt") or {})
    source_refs = tuple(str(item) for item in receipt.get("source_refs", ()) if str(item))
    source_decision = str(getattr(result, "decision", ""))
    verified = bool(
        trace.get("observed", False)
        and trace.get("source_backed", False)
        and trace.get("verified", False)
        and receipt.get("integrity_digest")
    )
    return {
        "decision_id": f"event-memory::{getattr(result, 'entry_id', '')}::{sequence}",
        "sequence": _index(sequence, "sequence"),
        "subsystem": "event_memory",
        "subject_id": _text(getattr(result, "entry_id", ""), "entry_id"),
        "evidence_ids": list(source_refs),
        "verified": verified,
        "contradiction": bool(trace.get("contradicted", False)),
        "stale": "stale" in source_decision or "expired" in source_decision,
        "capacity_available": source_decision not in {"evict_budget", "block_metabolic_budget"},
        "prediction_match": True,
        "support_count": max(len(source_refs), int(trace.get("sequence_support_count", 0) or 0)),
    }


def adapt_event_memory_retrieval(result: Any, *, subject_id: str, sequence: int) -> Dict[str, Any]:
    matches = tuple(dict(item) for item in getattr(result, "matches", ()) or ())
    evidence_ids = sorted(
        {
            str(item.get("source_ref", "") or item.get("entry_id", ""))
            for item in matches
            if item.get("source_ref") or item.get("entry_id")
        }
    )
    source_decision = str(getattr(result, "decision", ""))
    return {
        "decision_id": f"event-retrieval::{subject_id}::{sequence}",
        "sequence": _index(sequence, "sequence"),
        "subsystem": "event_memory_retrieval",
        "subject_id": _text(subject_id, "subject_id"),
        "evidence_ids": evidence_ids,
        "verified": True,
        "contradiction": "contradiction" in source_decision,
        "stale": "stale" in source_decision or "expired" in source_decision,
        "capacity_available": True,
        "prediction_match": bool(matches),
        "support_count": len(matches),
    }


def adapt_event_memory_evictions(result: Any, *, sequence_start: int) -> Tuple[Dict[str, Any], ...]:
    trace = dict(getattr(result, "trace", {}) or {})
    evicted = tuple(str(item) for item in trace.get("evicted_entry_ids", ()) if str(item))
    records = []
    for offset, entry_id in enumerate(evicted):
        sequence = _index(sequence_start + offset, "sequence")
        records.append(
            {
                "decision_id": f"event-eviction::{entry_id}::{sequence}",
                "sequence": sequence,
                "subsystem": "event_memory_eviction",
                "subject_id": _text(entry_id, "entry_id"),
                "evidence_ids": [str(getattr(result, "entry_id", ""))],
                "verified": True,
                "contradiction": False,
                "stale": False,
                "capacity_available": False,
                "prediction_match": False,
                "support_count": 1,
            }
        )
    return tuple(records)


def adapt_event_memory_revision(result: Any, *, sequence: int) -> Dict[str, Any]:
    trace = dict(getattr(result, "trace", {}) or {})
    receipt = dict(trace.get("verification_receipt") or {})
    evidence_ids = tuple(str(item) for item in receipt.get("source_refs", ()) if str(item))
    source_decision = str(getattr(result, "decision", ""))
    return {
        "decision_id": f"event-revision::{getattr(result, 'entry_id', '')}::{sequence}",
        "sequence": _index(sequence, "sequence"),
        "subsystem": "event_memory_revision",
        "subject_id": _text(getattr(result, "entry_id", ""), "entry_id"),
        "evidence_ids": list(evidence_ids),
        "verified": bool(
            trace.get("observed", False)
            and trace.get("source_backed", False)
            and trace.get("verified", False)
            and receipt.get("integrity_digest")
        ),
        "contradiction": bool(trace.get("contradicted", False)),
        "stale": False,
        "capacity_available": True,
        "prediction_match": source_decision != "replace_verified_revision",
        "support_count": len(set(evidence_ids)),
    }


def adapt_risa_proposal(proposal: Any, *, sequence: int) -> Dict[str, Any]:
    evidence_ids = tuple(
        str(item)
        for item in getattr(proposal, "source_refs", getattr(proposal, "evidence_ids", ()))
        if str(item)
    )
    contradiction_count = int(getattr(proposal, "contradiction_count", 0) or 0)
    frozen = bool(getattr(proposal, "frozen", False))
    return {
        "decision_id": f"risa::{getattr(proposal, 'proposal_id', '')}::{sequence}",
        "sequence": _index(sequence, "sequence"),
        "subsystem": "risa_proposal",
        "subject_id": _text(getattr(proposal, "proposal_id", ""), "proposal_id"),
        "evidence_ids": list(evidence_ids),
        "verified": bool(evidence_ids) or frozen,
        "contradiction": contradiction_count > 0 or frozen,
        "stale": False,
        "capacity_available": True,
        "prediction_match": True,
        "support_count": max(
            len(evidence_ids), int(getattr(proposal, "evidence_count", 0) or 0)
        ),
    }


def adapt_predictive_feedback(proposal: Any, *, sequence: int) -> Dict[str, Any]:
    evidence_ids = tuple(str(item) for item in getattr(proposal, "evidence_ids", ()) if str(item))
    edit_type = str(getattr(proposal, "edit_type", ""))
    frozen = bool(getattr(proposal, "frozen", False))
    return {
        "decision_id": f"predictive::{getattr(proposal, 'proposal_id', '')}::{sequence}",
        "sequence": _index(sequence, "sequence"),
        "subsystem": "predictive_feedback",
        "subject_id": _text(getattr(proposal, "proposal_id", ""), "proposal_id"),
        "evidence_ids": list(evidence_ids),
        "verified": True,
        "contradiction": frozen,
        "stale": False,
        "capacity_available": True,
        "prediction_match": edit_type == "request_more_evidence" and bool(evidence_ids),
        "support_count": len(set(evidence_ids)),
    }
