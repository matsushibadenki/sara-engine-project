from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Dict, Iterable, List

from sara_engine.learning.adaptive_credit import summarize_event_memory_credit
from sara_engine.memory.event_state_cache import EventStateCandidate
from sara_engine.multimodal.synesthetic_binding import SparseEventBundle


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _stable_id(text: str, modulus: int = 4096) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % max(1, int(modulus))


@dataclass(frozen=True)
class MultimodalBundleAdmissionResult:
    candidate: EventStateCandidate
    promotion_allowed: bool
    promotion_decision: str
    event_cost: int
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.candidate.entry_id,
            "promotion_allowed": bool(self.promotion_allowed),
            "promotion_decision": self.promotion_decision,
            "event_cost": int(self.event_cost),
            "trace": dict(self.trace),
        }


def build_multimodal_event_state_candidate(
    bundle: SparseEventBundle,
    *,
    time_segment: int | None = None,
    signature_width: int = 64,
) -> MultimodalBundleAdmissionResult:
    child_records = tuple(bundle.child_records)
    audit = bundle.audit
    source_refs = tuple(sorted({record.source_ref for record in child_records if record.source_ref}))
    source_ref = source_refs[0] if len(source_refs) == 1 else f"bundle::{bundle.event_id}"
    source_revision = ""
    if source_refs:
        digest = hashlib.sha256("|".join(source_refs).encode("utf-8")).hexdigest()
        source_revision = f"bundle-rev:{digest[:16]}"
    signature_parts: List[int] = []
    for record in child_records:
        signature_parts.extend(int(value) for value in record.sparse_signature)
        signature_parts.append(_stable_id(f"modality:{record.modality}"))
    signature = tuple(sorted(set(signature_parts))[: max(4, int(signature_width))])
    route_states = [
        {
            "responsibility": _clamp01(record.confidence * (1.0 - record.uncertainty)),
            "confidence": _clamp01(record.confidence),
            "longevity": 1.0 if record.observed else 0.0,
        }
        for record in child_records
    ]
    credit_summary = summarize_event_memory_credit(route_states)
    observed = bool(child_records) and all(record.observed for record in child_records)
    source_backed = bool(source_refs) and "source_backed" in set(bundle.evidence_types)
    payload_separable = bool(audit.payload_separable) if audit is not None else False
    multi_modality = len(set(bundle.modality_ids)) > 1
    verified = bool(
        audit is not None
        and audit.admitted
        and payload_separable
        and multi_modality
        and observed
        and source_backed
    )
    decision = "promote_verified_multimodal_bundle"
    if not multi_modality:
        decision = "freeze_single_modality_bundle"
    elif audit is None or not audit.admitted:
        decision = "freeze_unverified_bundle"
    elif not payload_separable:
        decision = "freeze_payload_collapse_risk"
    elif not observed:
        decision = "freeze_predicted_bundle"
    elif not source_backed:
        decision = "freeze_material_source"
    sequence_support_score = 1.0 if "repeated_coactivation" in set(bundle.evidence_types) else 0.0
    sequence_support_count = 1 if sequence_support_score > 0.0 else 0
    source_reliability = _clamp01(
        sum(float(record.confidence) for record in child_records) / float(max(1, len(child_records)))
    )
    candidate = EventStateCandidate(
        entry_id=str(bundle.event_id),
        signature=signature,
        source_ref=source_ref,
        source_revision=source_revision,
        time_segment=int(bundle.time_chunk_id if time_segment is None else time_segment),
        own_latent_id=str(bundle.event_id),
        causal_predecessors=tuple(sorted(record.event_id for record in child_records)),
        confidence=_clamp01(bundle.binding_strength),
        uncertainty=_clamp01(bundle.uncertainty),
        source_reliability=source_reliability,
        resonance_score=_clamp01(bundle.binding_strength),
        sequence_support_score=sequence_support_score,
        sequence_support_count=sequence_support_count,
        credit_score=float(credit_summary.get("credit_score", 0.0) or 0.0),
        credit_responsibility=float(credit_summary.get("credit_responsibility", 0.0) or 0.0),
        credit_confidence=float(credit_summary.get("credit_confidence", 0.0) or 0.0),
        credit_longevity=float(credit_summary.get("credit_longevity", 0.0) or 0.0),
        metabolic_headroom=1.0,
        observed=observed,
        source_backed=source_backed,
        verified=verified,
        contradicted=False,
        abstained=False,
        event_cost=sum(int(record.event_cost) for record in child_records) + len(signature),
    )
    return MultimodalBundleAdmissionResult(
        candidate=candidate,
        promotion_allowed=verified,
        promotion_decision=decision,
        event_cost=int(candidate.event_cost),
        trace={
            "bundle_event_id": bundle.event_id,
            "modality_ids": list(bundle.modality_ids),
            "evidence_types": list(bundle.evidence_types),
            "payload_separable": payload_separable,
            "source_ref_count": len(source_refs),
            "observed": observed,
            "source_backed": source_backed,
            "route_trace_count": len(bundle.route_trace),
            "audit": audit.to_dict() if audit is not None else None,
            "credit_summary": credit_summary,
        },
    )
