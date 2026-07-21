from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

from sara_engine.memory.verification_receipt import (
    VerificationReceipt,
    evidence_digest,
    issue_verification_receipt,
)


@dataclass(frozen=True)
class ModalityEvidence:
    modality: str
    label: str
    timestamp_ms: float
    source_ref: str
    observed: bool = True
    confidence: float = 1.0
    claim_key: str = ""


@dataclass(frozen=True)
class StructuralFusionDecision:
    decision: str
    label: str
    confidence: float
    observed_modalities: Tuple[str, ...]
    missing_modalities: Tuple[str, ...]
    contradiction: bool
    durable_mutation_allowed: bool
    trace: Dict[str, Any]
    verification_receipt: VerificationReceipt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "label": self.label,
            "confidence": self.confidence,
            "observed_modalities": list(self.observed_modalities),
            "missing_modalities": list(self.missing_modalities),
            "contradiction": self.contradiction,
            "durable_mutation_allowed": self.durable_mutation_allowed,
            "trace": dict(self.trace),
            "verification_receipt": self.verification_receipt.to_dict(),
        }


def structural_evidence_payload(
    evidence: Iterable[ModalityEvidence],
    expected_modalities: Iterable[str],
) -> Dict[str, Any]:
    rows = tuple(evidence)
    return {
        "expected_modalities": sorted({str(item) for item in expected_modalities}),
        "evidence": [
            {
                "modality": str(row.modality),
                "label": str(row.label),
                "claim_key": str(row.claim_key),
                "timestamp_ms": float(row.timestamp_ms),
                "source_ref": str(row.source_ref),
                "observed": bool(row.observed),
                "confidence": max(0.0, min(1.0, float(row.confidence))),
            }
            for row in sorted(rows, key=lambda item: (item.modality, item.timestamp_ms, item.source_ref))
        ],
    }


class MultimodalStructuralVerifier:
    """Checks modality-local evidence before any cross-modal promotion."""

    def __init__(self, *, max_binding_delay_ms: float = 32.0) -> None:
        if max_binding_delay_ms <= 0:
            raise ValueError("max_binding_delay_ms must be positive")
        self.max_binding_delay_ms = float(max_binding_delay_ms)

    def verify(
        self,
        evidence: Iterable[ModalityEvidence],
        *,
        expected_modalities: Iterable[str],
    ) -> StructuralFusionDecision:
        rows = tuple(evidence)
        expected = tuple(sorted({str(item) for item in expected_modalities}))
        observed = tuple(sorted({str(row.modality) for row in rows if row.observed}))
        missing = tuple(sorted(set(expected) - set(observed)))
        labels = tuple(sorted({str(row.label) for row in rows if row.observed and row.label}))
        claims = tuple(
            sorted(
                {
                    str(row.claim_key or row.label)
                    for row in rows
                    if row.observed and (row.claim_key or row.label)
                }
            )
        )
        timestamps = [float(row.timestamp_ms) for row in rows if row.observed]
        delay = max(timestamps) - min(timestamps) if timestamps else 0.0
        contradiction = len(claims) > 1
        source_complete = all(bool(row.source_ref) for row in rows if row.observed)
        confidence = min((max(0.0, min(1.0, float(row.confidence))) for row in rows), default=0.0)
        if not rows or not source_complete:
            decision = "abstain_missing_source"
            label = ""
        elif contradiction:
            decision = "abstain_cross_modal_contradiction"
            label = ""
        elif delay > self.max_binding_delay_ms:
            decision = "abstain_temporal_misalignment"
            label = ""
        elif missing:
            decision = "provisional_missing_modality_prediction"
            label = claims[0] if claims else ""
        elif not claims:
            decision = "abstain_unlabeled_structure"
            label = ""
        else:
            decision = "verify_cross_modal_structure"
            label = claims[0]
        evidence_payload = structural_evidence_payload(rows, expected)
        source_refs = tuple(row.source_ref for row in rows if row.observed and row.source_ref)
        receipt = issue_verification_receipt(
            verifier_id="multimodal-structural-verifier",
            verifier_version="v2",
            decision=decision,
            evidence=evidence_payload,
            source_refs=source_refs,
            source_revision=evidence_digest(evidence_payload),
            observed=bool(rows) and all(row.observed for row in rows),
            source_backed=source_complete and bool(source_refs),
            verified=decision == "verify_cross_modal_structure",
            contradicted=contradiction,
            abstained=decision.startswith("abstain"),
        )
        return StructuralFusionDecision(
            decision=decision,
            label=label,
            confidence=confidence,
            observed_modalities=observed,
            missing_modalities=missing,
            contradiction=contradiction,
            durable_mutation_allowed=False,
            trace={
                "expected_modalities": list(expected),
                "source_complete": source_complete,
                "observed_event_count": len(rows),
                "binding_delay_ms": round(delay, 6),
                "max_binding_delay_ms": self.max_binding_delay_ms,
                "label_set": list(labels),
                "claim_set": list(claims),
            },
            verification_receipt=receipt,
        )


__all__ = [
    "ModalityEvidence",
    "MultimodalStructuralVerifier",
    "StructuralFusionDecision",
    "structural_evidence_payload",
]
