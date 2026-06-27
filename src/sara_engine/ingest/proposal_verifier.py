from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from .candidate_proposals import (
    CandidateEvent,
    CandidateRelation,
    VerifiedRelation,
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class ProposalVerificationResult:
    accepted: bool
    decision: str
    record_type: str
    promoted_record: Optional[Dict[str, Any]]
    event_cost: int
    state_budget_units: int
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "decision": self.decision,
            "record_type": self.record_type,
            "promoted_record": dict(self.promoted_record) if self.promoted_record else None,
            "event_cost": self.event_cost,
            "state_budget_units": self.state_budget_units,
            "trace": dict(self.trace),
        }


class ProposalVerifier:
    """Promotes proposal records only when bounded evidence and prediction criteria pass."""

    def __init__(
        self,
        *,
        min_confidence: float = 0.70,
        min_evidence_count: int = 3,
        min_prediction_gain: float = 0.05,
        max_counterexample_rate: float = 0.35,
        require_source_hash: bool = True,
    ) -> None:
        self.min_confidence = _clamp01(min_confidence)
        self.min_evidence_count = max(1, int(min_evidence_count))
        self.min_prediction_gain = float(min_prediction_gain)
        self.max_counterexample_rate = _clamp01(max_counterexample_rate)
        self.require_source_hash = bool(require_source_hash)
        self.accept_count = 0
        self.reject_count = 0

    def verify_event(self, candidate: CandidateEvent) -> ProposalVerificationResult:
        decision = self._event_decision(candidate)
        accepted = decision == "accept_candidate_event"
        if accepted:
            self.accept_count += 1
        else:
            self.reject_count += 1
        return ProposalVerificationResult(
            accepted=accepted,
            decision=decision,
            record_type=candidate.record_type,
            promoted_record=None,
            event_cost=max(1, len(candidate.sparse_signature)) + 1,
            state_budget_units=self.accept_count + self.reject_count,
            trace=self._trace_for_candidate(
                confidence=candidate.confidence,
                evidence_count=candidate.evidence_count,
                counterexample_count=candidate.counterexample_count,
                prediction_gain=candidate.prediction_gain,
                has_source_hash=bool(candidate.lineage.source_hash),
            ),
        )

    def verify_relation(self, candidate: CandidateRelation) -> ProposalVerificationResult:
        return self.verify_relation_with_self_state(candidate, self_state_alignment=0.0)

    def verify_relation_with_self_state(
        self,
        candidate: CandidateRelation,
        *,
        self_state_alignment: float = 0.0,
    ) -> ProposalVerificationResult:
        decision = self._relation_decision(candidate, self_state_alignment=self_state_alignment)
        accepted = decision == "promote_verified_relation"
        if accepted:
            self.accept_count += 1
            promoted = VerifiedRelation(
                record_id=candidate.record_id,
                relation=candidate.relation,
                source_event_id=candidate.source_event_id,
                target_event_id=candidate.target_event_id,
                delay_lower_ms=candidate.delay_lower_ms,
                delay_upper_ms=candidate.delay_upper_ms,
                confidence=candidate.confidence,
                lineage=candidate.lineage,
                evidence_count=candidate.evidence_count,
                counterexample_count=candidate.counterexample_count,
                prediction_gain=candidate.prediction_gain,
            )
            promoted_record = promoted.to_dict()
        else:
            self.reject_count += 1
            promoted_record = None
        return ProposalVerificationResult(
            accepted=accepted,
            decision=decision,
            record_type=candidate.record_type,
            promoted_record=promoted_record,
            event_cost=max(1, candidate.evidence_count + candidate.counterexample_count),
            state_budget_units=self.accept_count + self.reject_count,
            trace=self._trace_for_candidate(
                confidence=candidate.confidence,
                evidence_count=candidate.evidence_count,
                counterexample_count=candidate.counterexample_count,
                prediction_gain=candidate.prediction_gain,
                has_source_hash=bool(candidate.lineage.source_hash),
                self_state_alignment=self_state_alignment,
            ),
        )

    def verify(
        self,
        proposal: Union[CandidateEvent, CandidateRelation],
    ) -> ProposalVerificationResult:
        if isinstance(proposal, CandidateEvent):
            return self.verify_event(proposal)
        if isinstance(proposal, CandidateRelation):
            return self.verify_relation(proposal)
        raise TypeError("proposal must be CandidateEvent or CandidateRelation")

    def _event_decision(self, candidate: CandidateEvent) -> str:
        if self.require_source_hash and not candidate.lineage.source_hash:
            return "reject_missing_source_hash"
        if candidate.confidence < self.min_confidence:
            return "reject_low_confidence"
        if candidate.evidence_count < self.min_evidence_count:
            return "reject_insufficient_evidence"
        if self._counterexample_rate(candidate.evidence_count, candidate.counterexample_count) > self.max_counterexample_rate:
            return "reject_counterexample_pressure"
        return "accept_candidate_event"

    def _relation_decision(self, candidate: CandidateRelation, *, self_state_alignment: float = 0.0) -> str:
        if self.require_source_hash and not candidate.lineage.source_hash:
            return "reject_missing_source_hash"
        if candidate.confidence < self.min_confidence:
            return "reject_low_confidence"
        if candidate.evidence_count < self.min_evidence_count:
            return "reject_insufficient_evidence"
        effective_prediction_gain = float(candidate.prediction_gain) + (0.05 * _clamp01(self_state_alignment))
        if effective_prediction_gain < self.min_prediction_gain:
            return "reject_low_prediction_gain"
        if self._counterexample_rate(candidate.evidence_count, candidate.counterexample_count) > self.max_counterexample_rate:
            return "reject_counterexample_pressure"
        return "promote_verified_relation"

    def _counterexample_rate(self, evidence_count: int, counterexample_count: int) -> float:
        total = max(1, int(evidence_count) + int(counterexample_count))
        return float(counterexample_count) / float(total)

    def _trace_for_candidate(
        self,
        *,
        confidence: float,
        evidence_count: int,
        counterexample_count: int,
        prediction_gain: float,
        has_source_hash: bool,
        self_state_alignment: float = 0.0,
    ) -> Dict[str, Any]:
        effective_prediction_gain = float(prediction_gain) + (0.05 * _clamp01(self_state_alignment))
        return {
            "confidence": _clamp01(confidence),
            "evidence_count": int(evidence_count),
            "counterexample_count": int(counterexample_count),
            "prediction_gain": float(prediction_gain),
            "effective_prediction_gain": float(effective_prediction_gain),
            "self_state_alignment": _clamp01(self_state_alignment),
            "counterexample_rate": self._counterexample_rate(evidence_count, counterexample_count),
            "has_source_hash": bool(has_source_hash),
            "thresholds": {
                "min_confidence": self.min_confidence,
                "min_evidence_count": self.min_evidence_count,
                "min_prediction_gain": self.min_prediction_gain,
                "max_counterexample_rate": self.max_counterexample_rate,
            },
            "accepted_count": self.accept_count,
            "rejected_count": self.reject_count,
        }
