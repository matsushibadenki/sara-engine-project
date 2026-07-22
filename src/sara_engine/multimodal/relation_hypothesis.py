"""Bounded cross-modal relation hypotheses with receipt-bound evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Tuple

from sara_engine.memory.verification_receipt import evidence_digest

from .structural_verification import (
    ModalityEvidence,
    StructuralFusionDecision,
    structural_evidence_payload,
)


@dataclass(frozen=True)
class CrossModalHypothesisObservation:
    observation_id: str
    observation_source_id: str
    source_revision: str
    decision: str
    evidence_digest: str
    source_refs: Tuple[str, ...]
    observed_modalities: Tuple[str, ...]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["source_refs"] = list(self.source_refs)
        payload["observed_modalities"] = list(self.observed_modalities)
        return payload


@dataclass(frozen=True)
class CrossModalRelationHypothesis:
    hypothesis_id: str
    claim_key: str
    modality_relations: Tuple[str, ...]
    state: str
    support_count: int
    distinct_source_count: int
    contradiction_count: int
    provisional_count: int
    abstention_count: int
    confidence: float
    source_refs: Tuple[str, ...]
    source_revisions: Tuple[str, ...]
    observation_ids: Tuple[str, ...]
    eligible_for_review: bool
    frozen: bool
    durable_mutation_allowed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in (
            "modality_relations",
            "source_refs",
            "source_revisions",
            "observation_ids",
        ):
            payload[key] = list(payload[key])
        return payload


@dataclass(frozen=True)
class CrossModalHypothesisUpdate:
    accepted: bool
    reason: str
    hypothesis: CrossModalRelationHypothesis | None
    observation: CrossModalHypothesisObservation | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "sara-cross-modal-hypothesis-update-v1",
            "accepted": self.accepted,
            "reason": self.reason,
            "hypothesis": self.hypothesis.to_dict() if self.hypothesis else None,
            "observation": self.observation.to_dict() if self.observation else None,
            "durable_mutation_allowed": False,
        }


class BoundedCrossModalHypothesisLedger:
    """Accumulate cross-modal support without creating durable concepts."""

    def __init__(
        self,
        *,
        min_verified_observations: int = 2,
        min_distinct_sources: int = 2,
        max_hypotheses: int = 128,
        max_observations_per_hypothesis: int = 16,
    ) -> None:
        self.min_verified_observations = max(1, int(min_verified_observations))
        self.min_distinct_sources = max(1, int(min_distinct_sources))
        self.max_hypotheses = max(1, int(max_hypotheses))
        self.max_observations_per_hypothesis = max(
            1, int(max_observations_per_hypothesis)
        )
        self._observations: Dict[str, List[CrossModalHypothesisObservation]] = {}
        self._relations: Dict[str, Tuple[str, ...]] = {}

    @staticmethod
    def _relations_from(evidence: Iterable[ModalityEvidence]) -> Tuple[str, ...]:
        return tuple(
            sorted(
                {
                    f"{item.modality}:{item.label}"
                    for item in evidence
                    if item.observed and item.modality and item.label
                }
            )
        )

    def _view(self, claim_key: str) -> CrossModalRelationHypothesis:
        rows = tuple(self._observations.get(claim_key, ()))
        verified_rows = tuple(
            row for row in rows if row.decision == "verify_cross_modal_structure"
        )
        source_ids = {
            row.observation_source_id for row in verified_rows if row.observation_source_id
        }
        contradictions = sum(
            row.decision == "abstain_cross_modal_contradiction" for row in rows
        )
        provisional = sum(
            row.decision == "provisional_missing_modality_prediction" for row in rows
        )
        abstentions = sum(row.decision.startswith("abstain") for row in rows)
        frozen = contradictions > 0
        eligible = bool(
            not frozen
            and len(verified_rows) >= self.min_verified_observations
            and len(source_ids) >= self.min_distinct_sources
        )
        state = "frozen_contradiction" if frozen else (
            "eligible_for_review" if eligible else "provisional_hypothesis"
        )
        digest = sha256(claim_key.encode("utf-8")).hexdigest()[:16]
        return CrossModalRelationHypothesis(
            hypothesis_id=f"cross-modal-hypothesis::{digest}",
            claim_key=claim_key,
            modality_relations=self._relations.get(claim_key, ()),
            state=state,
            support_count=len(verified_rows),
            distinct_source_count=len(source_ids),
            contradiction_count=contradictions,
            provisional_count=provisional,
            abstention_count=abstentions,
            confidence=round(
                sum(row.confidence for row in verified_rows)
                / float(max(1, len(verified_rows))),
                6,
            ),
            source_refs=tuple(
                sorted({value for row in rows for value in row.source_refs})
            ),
            source_revisions=tuple(sorted({row.source_revision for row in rows})),
            observation_ids=tuple(row.observation_id for row in rows),
            eligible_for_review=eligible,
            frozen=frozen,
        )

    def observe(
        self,
        *,
        claim_key: str,
        decision: StructuralFusionDecision,
        evidence: Iterable[ModalityEvidence],
        expected_modalities: Iterable[str],
        observation_source_id: str,
        source_revision: str,
    ) -> CrossModalHypothesisUpdate:
        claim = str(claim_key).strip()
        source_id = str(observation_source_id).strip()
        revision = str(source_revision).strip()
        evidence_rows = tuple(evidence)
        if not claim or not source_id or not revision:
            return CrossModalHypothesisUpdate(False, "missing_identity_or_revision", None, None)
        payload = structural_evidence_payload(evidence_rows, expected_modalities)
        digest = evidence_digest(payload)
        receipt = decision.verification_receipt
        source_refs = tuple(
            sorted({item.source_ref for item in evidence_rows if item.observed and item.source_ref})
        )
        receipt_valid = bool(
            receipt.is_valid()
            and receipt.verifier_id == "multimodal-structural-verifier"
            and receipt.evidence_digest == digest
            and set(source_refs).issubset(set(receipt.source_refs))
            and decision.decision == receipt.decision
            and receipt.observed
            and receipt.source_backed
        )
        if decision.decision == "verify_cross_modal_structure":
            receipt_valid = bool(
                receipt_valid
                and receipt.verified
                and not receipt.abstained
                and not receipt.contradicted
            )
        elif decision.decision.startswith("abstain"):
            receipt_valid = bool(receipt_valid and receipt.abstained)
        if decision.decision == "abstain_cross_modal_contradiction":
            receipt_valid = bool(receipt_valid and receipt.contradicted)
        evidence_claims = {
            str(item.claim_key or item.label)
            for item in evidence_rows
            if item.observed and (item.claim_key or item.label)
        }
        receipt_valid = bool(receipt_valid and claim in evidence_claims)
        if not receipt_valid:
            return CrossModalHypothesisUpdate(False, "invalid_or_stale_verification_receipt", None, None)
        if claim not in self._observations and len(self._observations) >= self.max_hypotheses:
            return CrossModalHypothesisUpdate(False, "hypothesis_budget_exceeded", None, None)
        rows = self._observations.setdefault(claim, [])
        if len(rows) >= self.max_observations_per_hypothesis:
            return CrossModalHypothesisUpdate(
                False, "observation_budget_exceeded", self._view(claim), None
            )
        if decision.decision == "verify_cross_modal_structure":
            reused_refs = {
                value
                for row in rows
                if row.decision == "verify_cross_modal_structure"
                and row.observation_source_id != source_id
                for value in set(row.source_refs) & set(source_refs)
            }
            if reused_refs:
                return CrossModalHypothesisUpdate(
                    False,
                    "source_ref_reuse_across_independent_observations",
                    self._view(claim),
                    None,
                )
        observation_digest = sha256(
            f"{claim}|{source_id}|{revision}|{digest}".encode("utf-8")
        ).hexdigest()[:20]
        observation_id = f"cross-modal-observation::{observation_digest}"
        if any(row.observation_id == observation_id for row in rows):
            return CrossModalHypothesisUpdate(
                False, "duplicate_observation", self._view(claim), None
            )
        observation = CrossModalHypothesisObservation(
            observation_id=observation_id,
            observation_source_id=source_id,
            source_revision=revision,
            decision=decision.decision,
            evidence_digest=digest,
            source_refs=source_refs,
            observed_modalities=tuple(decision.observed_modalities),
            confidence=max(0.0, min(1.0, float(decision.confidence))),
        )
        rows.append(observation)
        relations = self._relations_from(evidence_rows)
        if decision.decision == "verify_cross_modal_structure" and relations:
            self._relations[claim] = tuple(
                sorted(set(self._relations.get(claim, ())) | set(relations))
            )
        return CrossModalHypothesisUpdate(True, "observation_recorded", self._view(claim), observation)

    def get(self, claim_key: str) -> CrossModalRelationHypothesis | None:
        claim = str(claim_key).strip()
        return self._view(claim) if claim in self._observations else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "sara-bounded-cross-modal-hypothesis-ledger-v1",
            "hypotheses": [self._view(key).to_dict() for key in sorted(self._observations)],
            "limits": {
                "min_verified_observations": self.min_verified_observations,
                "min_distinct_sources": self.min_distinct_sources,
                "max_hypotheses": self.max_hypotheses,
                "max_observations_per_hypothesis": self.max_observations_per_hypothesis,
            },
            "durable_mutation_allowed": False,
        }


__all__ = [
    "BoundedCrossModalHypothesisLedger",
    "CrossModalHypothesisObservation",
    "CrossModalHypothesisUpdate",
    "CrossModalRelationHypothesis",
]
