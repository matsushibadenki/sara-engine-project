from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from typing import Any, Dict, Iterable, Tuple

from sara_engine.memory.event_state_cache import EventStateCandidate
from sara_engine.memory.verification_receipt import evidence_digest, issue_verification_receipt


@dataclass(frozen=True)
class CausalEvidence:
    source: str
    target: str
    relation_type: str
    source_ref: str
    intervention_count: int = 0
    contrastive_count: int = 0
    confidence: float = 0.0
    verified: bool = True
    observed: bool = True
    source_revision: str = ""


@dataclass(frozen=True)
class CausalInference:
    relation_type: str
    source: str
    target: str
    confidence: float
    abstained: bool
    durable_mutation_allowed: bool
    supporting_paths: Tuple[str, ...]
    alternatives: Tuple[str, ...]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["supporting_paths"] = list(self.supporting_paths)
        payload["alternatives"] = list(self.alternatives)
        return payload


class BoundedCausalReasoner:
    """Separates temporal correlation, causal candidates, and verified causes."""

    def __init__(self, *, min_confidence: float = 0.65, max_branch_depth: int = 3) -> None:
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.max_branch_depth = max(1, int(max_branch_depth))

    def infer(self, evidence: Iterable[CausalEvidence]) -> CausalInference:
        rows = tuple(
            item for item in evidence if item.verified and item.observed and item.source_ref
        )
        if not rows:
            return self._abstain("no_verified_source_backed_evidence")
        sources = {item.source for item in rows}
        targets = {item.target for item in rows}
        relation_types = {item.relation_type for item in rows}
        if len(sources) != 1 or len(targets) != 1:
            return self._abstain("multiple_candidate_paths_without_resolution")
        source, target = next(iter(sources)), next(iter(targets))
        if "contradicts" in relation_types:
            return CausalInference(
                "abstain", source, target, 0.0, True, False,
                tuple(sorted(item.source_ref for item in rows)),
                ("conflicting_relation",), "source_conflict_freeze",
            )
        intervention = sum(max(0, int(item.intervention_count)) for item in rows)
        contrastive = sum(max(0, int(item.contrastive_count)) for item in rows)
        confidence = min(float(item.confidence) for item in rows)
        causal_hypothesis = bool(
            {"causes_candidate", "causes_verified"} & relation_types
        )
        if causal_hypothesis and intervention > 0 and contrastive > 0 and confidence >= self.min_confidence:
            relation = "causes_verified"
            reason = "intervention_and_contrastive_support"
        elif {"precedes", "correlates_with", "causes_candidate", "causes_verified"} & relation_types:
            relation = "causes_candidate"
            reason = "insufficient_verified_causal_support"
        else:
            return self._abstain("unsupported_causal_relation")
        return CausalInference(
            relation, source, target, round(max(0.0, min(1.0, confidence)), 6),
            False, False, tuple(sorted(item.source_ref for item in rows)), (), reason,
        )

    def counterfactual(
        self,
        inference: CausalInference,
        *,
        intervention: str,
        predicted_outcome: str,
        alternative_outcome: str,
        depth: int = 1,
    ) -> Dict[str, Any]:
        bounded_depth = max(1, int(depth))
        if bounded_depth > self.max_branch_depth or inference.relation_type != "causes_verified":
            return {
                "abstained": True,
                "reason": "causal_support_or_branch_budget_insufficient",
                "branch_count": 0,
                "durable_mutation_allowed": False,
            }
        return {
            "abstained": False,
            "reason": "verified_causal_branch",
            "branch_count": 2,
            "depth": bounded_depth,
            "intervention": intervention,
            "predicted_outcome": predicted_outcome,
            "alternative_outcome": alternative_outcome,
            "supporting_paths": list(inference.supporting_paths),
            "durable_mutation_allowed": False,
        }

    @staticmethod
    def _abstain(reason: str) -> CausalInference:
        return CausalInference("abstain", "", "", 0.0, True, False, (), (), reason)


def causal_event_state_candidate(
    inference: CausalInference,
    *,
    source_ref: str,
    time_segment: int = 0,
) -> EventStateCandidate:
    """Map causal output to Event Memory without promoting candidate-only claims."""
    source_backed = bool(source_ref and inference.supporting_paths)
    verified = bool(
        inference.relation_type == "causes_verified"
        and not inference.abstained
        and source_backed
    )
    source_revision = evidence_digest(
        {
            "source": inference.source,
            "target": inference.target,
            "supporting_paths": list(inference.supporting_paths),
        }
    )
    source_id = int.from_bytes(sha256(inference.source.encode("utf-8")).digest()[:2], "big") % 4096
    target_id = int.from_bytes(sha256(inference.target.encode("utf-8")).digest()[:2], "big") % 4096
    receipt = issue_verification_receipt(
        verifier_id="bounded-causal-reasoner",
        verifier_version="v2",
        decision=inference.relation_type,
        evidence=inference.to_dict(),
        source_refs=(source_ref, *inference.supporting_paths),
        source_revision=source_revision,
        observed=source_backed,
        source_backed=source_backed,
        verified=verified,
        contradicted=inference.abstained or inference.relation_type == "abstain",
        abstained=inference.abstained,
    )
    return EventStateCandidate(
        entry_id=f"causal:{inference.source}->{inference.target}",
        signature=tuple(sorted({source_id, target_id})),
        source_ref=str(source_ref),
        source_revision=source_revision,
        time_segment=int(time_segment),
        own_latent_id=f"causal:{inference.source}->{inference.target}",
        causal_predecessors=tuple(inference.supporting_paths),
        confidence=float(inference.confidence),
        uncertainty=1.0 - float(inference.confidence),
        source_reliability=1.0,
        resonance_score=float(inference.confidence),
        sequence_support_score=1.0 if verified else 0.0,
        sequence_support_count=len(inference.supporting_paths),
        credit_score=float(inference.confidence),
        credit_responsibility=float(inference.confidence),
        credit_confidence=float(inference.confidence),
        credit_longevity=float(inference.confidence),
        metabolic_headroom=1.0,
        observed=source_backed,
        source_backed=source_backed,
        verified=verified,
        contradicted=inference.abstained or inference.relation_type == "abstain",
        abstained=inference.abstained,
        event_cost=4,
        verification_receipt=receipt,
    )
