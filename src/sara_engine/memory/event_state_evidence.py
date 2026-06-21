from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from sara_engine.learning.resonance_credit import SparseResonanceCreditAssigner
from sara_engine.learning.resonance_evidence import build_resonance_evidence
from sara_engine.memory.event_state_cache import EventStateCandidate


@dataclass(frozen=True)
class EventStateEvidenceResult:
    candidate: EventStateCandidate
    promotion_allowed: bool
    promotion_decision: str
    event_cost: int
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.candidate.entry_id,
            "promotion_allowed": self.promotion_allowed,
            "promotion_decision": self.promotion_decision,
            "event_cost": self.event_cost,
            "trace": dict(self.trace),
        }


def build_event_state_candidate(
    material: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
    *,
    time_segment: int,
) -> EventStateEvidenceResult:
    signature_raw = material.get("sparse_signature", [])
    signature = (
        tuple(int(value) for value in signature_raw)
        if isinstance(signature_raw, list)
        else ()
    )
    source_ref = str(
        material.get("source_ref")
        or material.get("source_url")
        or material.get("source_path")
        or ""
    )
    observed = bool(material.get("observed_only", False))
    compliance_allowed = str(material.get("compliance_level", "")) == "allow"
    evidence = build_resonance_evidence(reports)
    assigner = SparseResonanceCreditAssigner(max_links=1)
    eligibility = (
        {(signature[0], signature[1] if len(signature) > 1 else signature[0]): 1.0}
        if signature
        else {}
    )
    credit = assigner.apply(eligibility, evidence.signals)
    material_source_backed = bool(source_ref and material.get("material_hash"))
    promotion_allowed = bool(
        credit.update_allowed
        and observed
        and compliance_allowed
        and material_source_backed
        and signature
    )
    promotion_decision = credit.decision
    if credit.update_allowed and not observed:
        promotion_decision = "freeze_predicted_material"
    elif credit.update_allowed and not compliance_allowed:
        promotion_decision = "freeze_compliance"
    elif credit.update_allowed and not material_source_backed:
        promotion_decision = "freeze_material_source"
    elif credit.update_allowed and not signature:
        promotion_decision = "freeze_empty_signature"

    candidate = EventStateCandidate(
        entry_id=str(
            material.get("manifest_id")
            or material.get("material_id")
            or material.get("material_hash")
            or ""
        ),
        signature=signature,
        source_ref=source_ref,
        source_revision=str(material.get("material_hash", "")),
        time_segment=int(time_segment),
        own_latent_id=str(material.get("latent_cluster_id", "")),
        confidence=float(material.get("quality_score", 0.0) or 0.0),
        uncertainty=max(
            0.0,
            1.0 - float(material.get("quality_score", 0.0) or 0.0),
        ),
        source_reliability=float(material.get("quality_score", 0.0) or 0.0),
        resonance_score=float(credit.resonance_score),
        sequence_support_score=float(material.get("sequence_support_score", 0.0) or 0.0),
        sequence_support_count=int(material.get("sequence_support_count", 0) or 0),
        metabolic_headroom=float(
            evidence.signals.get("metabolic_headroom", 0.0) or 0.0
        ),
        observed=observed,
        source_backed=bool(
            evidence.signals.get("source_backed", False)
            and material_source_backed
        ),
        verified=promotion_allowed,
        contradicted=bool(
            float(evidence.signals.get("contradiction", 0.0) or 0.0) >= 0.55
        ),
        abstained=bool(evidence.signals.get("abstained", False)),
        event_cost=int(material.get("event_cost", 0) or 0)
        + evidence.event_cost
        + credit.event_cost,
    )
    return EventStateEvidenceResult(
        candidate=candidate,
        promotion_allowed=promotion_allowed,
        promotion_decision=(
            "promote_verified_event_state"
            if promotion_allowed
            else promotion_decision
        ),
        event_cost=candidate.event_cost,
        trace={
            "material_schema": str(material.get("schema", "")),
            "material_hash": str(material.get("material_hash", "")),
            "source_ref": source_ref,
            "compliance_allowed": compliance_allowed,
            "observed": observed,
            "evidence": evidence.to_dict(),
            "credit": credit.to_dict(),
        },
    )
