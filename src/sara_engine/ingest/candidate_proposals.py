from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _bounded_ints(values: Iterable[int], limit: int = 64) -> Tuple[int, ...]:
    return tuple(sorted({int(value) for value in values})[: max(1, int(limit))])


def _bounded_strings(values: Iterable[str], limit: int = 16) -> Tuple[str, ...]:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return tuple(cleaned[: max(1, int(limit))])


@dataclass(frozen=True)
class ProposalLineage:
    source_ref: str
    source_hash: str
    extractor_name: str
    extractor_version: str
    parent_ids: Tuple[str, ...] = ()
    proposal_model: str = ""
    proposal_config_hash: str = ""
    observed_anchor_ids: Tuple[str, ...] = ()
    schema: str = "sara-proposal-lineage-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "source_ref": self.source_ref,
            "source_hash": self.source_hash,
            "extractor_name": self.extractor_name,
            "extractor_version": self.extractor_version,
            "parent_ids": list(self.parent_ids),
            "proposal_model": self.proposal_model,
            "proposal_config_hash": self.proposal_config_hash,
            "observed_anchor_ids": list(self.observed_anchor_ids),
        }


@dataclass(frozen=True)
class ObservedEvent:
    record_id: str
    modality: str
    local_time_ms: int
    label: str = ""
    duration_ms: int = 0
    confidence: float = 1.0
    sparse_signature: Tuple[int, ...] = ()
    lineage: ProposalLineage = field(
        default_factory=lambda: ProposalLineage(
            source_ref="",
            source_hash="",
            extractor_name="",
            extractor_version="",
        )
    )
    record_type: str = "observed_event"
    verification: str = "observed"
    schema: str = "sara-observed-event-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "record_type": self.record_type,
            "record_id": self.record_id,
            "modality": self.modality,
            "local_time_ms": int(self.local_time_ms),
            "duration_ms": int(self.duration_ms),
            "label": self.label,
            "confidence": _clamp01(self.confidence),
            "sparse_signature": list(self.sparse_signature),
            "verification": self.verification,
            "lineage": self.lineage.to_dict(),
        }


@dataclass(frozen=True)
class CandidateEvent:
    record_id: str
    modality: str
    label: str
    local_time_ms: int
    confidence: float
    lineage: ProposalLineage
    duration_ms: int = 0
    sparse_signature: Tuple[int, ...] = ()
    evidence_count: int = 1
    counterexample_count: int = 0
    prediction_gain: float = 0.0
    verification: str = "unverified"
    record_type: str = "candidate_event"
    schema: str = "sara-candidate-event-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "record_type": self.record_type,
            "record_id": self.record_id,
            "modality": self.modality,
            "label": self.label,
            "local_time_ms": int(self.local_time_ms),
            "duration_ms": int(self.duration_ms),
            "confidence": _clamp01(self.confidence),
            "sparse_signature": list(self.sparse_signature),
            "evidence_count": max(0, int(self.evidence_count)),
            "counterexample_count": max(0, int(self.counterexample_count)),
            "prediction_gain": float(self.prediction_gain),
            "verification": self.verification,
            "lineage": self.lineage.to_dict(),
        }


@dataclass(frozen=True)
class CandidateRelation:
    record_id: str
    relation: str
    source_event_id: str
    target_event_id: str
    delay_lower_ms: int
    delay_upper_ms: int
    confidence: float
    lineage: ProposalLineage
    evidence_count: int = 1
    counterexample_count: int = 0
    prediction_gain: float = 0.0
    verification: str = "unverified"
    record_type: str = "candidate_relation"
    schema: str = "sara-candidate-relation-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "record_type": self.record_type,
            "record_id": self.record_id,
            "relation": self.relation,
            "source_event_id": self.source_event_id,
            "target_event_id": self.target_event_id,
            "delay_lower_ms": int(self.delay_lower_ms),
            "delay_upper_ms": int(self.delay_upper_ms),
            "confidence": _clamp01(self.confidence),
            "evidence_count": max(0, int(self.evidence_count)),
            "counterexample_count": max(0, int(self.counterexample_count)),
            "prediction_gain": float(self.prediction_gain),
            "verification": self.verification,
            "lineage": self.lineage.to_dict(),
        }


@dataclass(frozen=True)
class VerifiedRelation:
    record_id: str
    relation: str
    source_event_id: str
    target_event_id: str
    delay_lower_ms: int
    delay_upper_ms: int
    confidence: float
    lineage: ProposalLineage
    evidence_count: int
    counterexample_count: int
    prediction_gain: float
    verification: str = "provisional"
    record_type: str = "verified_relation"
    schema: str = "sara-verified-relation-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "record_type": self.record_type,
            "record_id": self.record_id,
            "relation": self.relation,
            "source_event_id": self.source_event_id,
            "target_event_id": self.target_event_id,
            "delay_lower_ms": int(self.delay_lower_ms),
            "delay_upper_ms": int(self.delay_upper_ms),
            "confidence": _clamp01(self.confidence),
            "evidence_count": max(0, int(self.evidence_count)),
            "counterexample_count": max(0, int(self.counterexample_count)),
            "prediction_gain": float(self.prediction_gain),
            "verification": self.verification,
            "lineage": self.lineage.to_dict(),
        }


@dataclass(frozen=True)
class ConceptCrystalCandidate:
    record_id: str
    concept_key: str
    supporting_relation_ids: Tuple[str, ...]
    confidence: float
    evidence_count: int
    counterexample_count: int
    prediction_gain: float
    lineage: ProposalLineage
    verification: str = "provisional"
    record_type: str = "concept_crystal_candidate"
    schema: str = "sara-concept-crystal-candidate-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "record_type": self.record_type,
            "record_id": self.record_id,
            "concept_key": self.concept_key,
            "supporting_relation_ids": list(self.supporting_relation_ids),
            "confidence": _clamp01(self.confidence),
            "evidence_count": max(0, int(self.evidence_count)),
            "counterexample_count": max(0, int(self.counterexample_count)),
            "prediction_gain": float(self.prediction_gain),
            "verification": self.verification,
            "lineage": self.lineage.to_dict(),
        }


def make_proposal_lineage(
    source_ref: str,
    source_hash: str,
    extractor_name: str,
    extractor_version: str,
    *,
    parent_ids: Iterable[str] = (),
    proposal_model: str = "",
    proposal_config_hash: str = "",
    observed_anchor_ids: Iterable[str] = (),
) -> ProposalLineage:
    return ProposalLineage(
        source_ref=str(source_ref),
        source_hash=str(source_hash),
        extractor_name=str(extractor_name),
        extractor_version=str(extractor_version),
        parent_ids=_bounded_strings(parent_ids),
        proposal_model=str(proposal_model),
        proposal_config_hash=str(proposal_config_hash),
        observed_anchor_ids=_bounded_strings(observed_anchor_ids),
    )


def make_observed_event(payload: Mapping[str, Any]) -> ObservedEvent:
    lineage_payload = payload.get("lineage", {})
    lineage = (
        lineage_payload
        if isinstance(lineage_payload, ProposalLineage)
        else make_proposal_lineage(
            source_ref=str(getattr(lineage_payload, "get", lambda *_: "")("source_ref", "") or ""),
            source_hash=str(getattr(lineage_payload, "get", lambda *_: "")("source_hash", "") or ""),
            extractor_name=str(getattr(lineage_payload, "get", lambda *_: "")("extractor_name", "") or ""),
            extractor_version=str(getattr(lineage_payload, "get", lambda *_: "")("extractor_version", "") or ""),
            parent_ids=getattr(lineage_payload, "get", lambda *_: ())("parent_ids", ()) or (),
            proposal_model=str(getattr(lineage_payload, "get", lambda *_: "")("proposal_model", "") or ""),
            proposal_config_hash=str(
                getattr(lineage_payload, "get", lambda *_: "")("proposal_config_hash", "") or ""
            ),
            observed_anchor_ids=getattr(lineage_payload, "get", lambda *_: ())("observed_anchor_ids", ()) or (),
        )
    )
    return ObservedEvent(
        record_id=str(payload.get("record_id", "")),
        modality=str(payload.get("modality", "")),
        local_time_ms=int(payload.get("local_time_ms", 0) or 0),
        label=str(payload.get("label", "") or ""),
        duration_ms=max(0, int(payload.get("duration_ms", 0) or 0)),
        confidence=_clamp01(float(payload.get("confidence", 1.0) or 0.0)),
        sparse_signature=_bounded_ints(payload.get("sparse_signature", ()) or ()),
        lineage=lineage,
    )


def make_candidate_event(payload: Mapping[str, Any]) -> CandidateEvent:
    lineage = make_proposal_lineage(
        source_ref=str(payload.get("source_ref", "") or ""),
        source_hash=str(payload.get("source_hash", "") or ""),
        extractor_name=str(payload.get("extractor_name", "") or "proposal"),
        extractor_version=str(payload.get("extractor_version", "") or "v1"),
        parent_ids=payload.get("parent_ids", ()) or (),
        proposal_model=str(payload.get("proposal_model", "") or ""),
        proposal_config_hash=str(payload.get("proposal_config_hash", "") or ""),
        observed_anchor_ids=payload.get("observed_anchor_ids", ()) or (),
    )
    return CandidateEvent(
        record_id=str(payload.get("record_id", "")),
        modality=str(payload.get("modality", "")),
        label=str(payload.get("label", "") or ""),
        local_time_ms=int(payload.get("local_time_ms", 0) or 0),
        confidence=_clamp01(float(payload.get("confidence", 0.0) or 0.0)),
        lineage=lineage,
        duration_ms=max(0, int(payload.get("duration_ms", 0) or 0)),
        sparse_signature=_bounded_ints(payload.get("sparse_signature", ()) or ()),
        evidence_count=max(0, int(payload.get("evidence_count", 1) or 0)),
        counterexample_count=max(0, int(payload.get("counterexample_count", 0) or 0)),
        prediction_gain=float(payload.get("prediction_gain", 0.0) or 0.0),
    )


def make_candidate_relation(payload: Mapping[str, Any]) -> CandidateRelation:
    lineage = make_proposal_lineage(
        source_ref=str(payload.get("source_ref", "") or ""),
        source_hash=str(payload.get("source_hash", "") or ""),
        extractor_name=str(payload.get("extractor_name", "") or "proposal"),
        extractor_version=str(payload.get("extractor_version", "") or "v1"),
        parent_ids=payload.get("parent_ids", ()) or (),
        proposal_model=str(payload.get("proposal_model", "") or ""),
        proposal_config_hash=str(payload.get("proposal_config_hash", "") or ""),
        observed_anchor_ids=payload.get("observed_anchor_ids", ()) or (),
    )
    lower = int(payload.get("delay_lower_ms", 0) or 0)
    upper = int(payload.get("delay_upper_ms", 0) or 0)
    if upper < lower:
        lower, upper = upper, lower
    return CandidateRelation(
        record_id=str(payload.get("record_id", "")),
        relation=str(payload.get("relation", "") or ""),
        source_event_id=str(payload.get("source_event_id", "") or ""),
        target_event_id=str(payload.get("target_event_id", "") or ""),
        delay_lower_ms=lower,
        delay_upper_ms=upper,
        confidence=_clamp01(float(payload.get("confidence", 0.0) or 0.0)),
        lineage=lineage,
        evidence_count=max(0, int(payload.get("evidence_count", 1) or 0)),
        counterexample_count=max(0, int(payload.get("counterexample_count", 0) or 0)),
        prediction_gain=float(payload.get("prediction_gain", 0.0) or 0.0),
    )


def make_verified_relation(payload: Mapping[str, Any]) -> VerifiedRelation:
    lineage = make_proposal_lineage(
        source_ref=str(payload.get("source_ref", "") or ""),
        source_hash=str(payload.get("source_hash", "") or ""),
        extractor_name=str(payload.get("extractor_name", "") or "proposal"),
        extractor_version=str(payload.get("extractor_version", "") or "v1"),
        parent_ids=payload.get("parent_ids", ()) or (),
        proposal_model=str(payload.get("proposal_model", "") or ""),
        proposal_config_hash=str(payload.get("proposal_config_hash", "") or ""),
        observed_anchor_ids=payload.get("observed_anchor_ids", ()) or (),
    )
    lower = int(payload.get("delay_lower_ms", 0) or 0)
    upper = int(payload.get("delay_upper_ms", 0) or 0)
    if upper < lower:
        lower, upper = upper, lower
    return VerifiedRelation(
        record_id=str(payload.get("record_id", "")),
        relation=str(payload.get("relation", "") or ""),
        source_event_id=str(payload.get("source_event_id", "") or ""),
        target_event_id=str(payload.get("target_event_id", "") or ""),
        delay_lower_ms=lower,
        delay_upper_ms=upper,
        confidence=_clamp01(float(payload.get("confidence", 0.0) or 0.0)),
        lineage=lineage,
        evidence_count=max(0, int(payload.get("evidence_count", 0) or 0)),
        counterexample_count=max(0, int(payload.get("counterexample_count", 0) or 0)),
        prediction_gain=float(payload.get("prediction_gain", 0.0) or 0.0),
        verification=str(payload.get("verification", "provisional") or "provisional"),
    )
