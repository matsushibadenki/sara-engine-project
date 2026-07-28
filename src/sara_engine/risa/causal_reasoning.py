from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from typing import Any, Dict, Iterable, Sequence, Tuple

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
    event_path: Tuple[str, ...] = ()
    context_tags: Tuple[str, ...] = ()
    feedback_stable: bool = True


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
    context_tags: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["supporting_paths"] = list(self.supporting_paths)
        payload["alternatives"] = list(self.alternatives)
        payload["context_tags"] = list(self.context_tags)
        return payload


@dataclass(frozen=True)
class CounterfactualBranchRecord:
    branch_id: str
    transaction_id: str
    branch_type: str
    depth: int
    intervention: str
    outcome: str
    supporting_paths: Tuple[str, ...]
    alternatives: Tuple[str, ...]
    context_tags: Tuple[str, ...]
    status: str = "staged"
    rollback_action: str = "discard_counterfactual_branch"
    durable_mutation_allowed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["supporting_paths"] = list(self.supporting_paths)
        payload["alternatives"] = list(self.alternatives)
        payload["context_tags"] = list(self.context_tags)
        return payload


@dataclass(frozen=True)
class CounterfactualBranchResult:
    transaction_id: str
    branches: Tuple[CounterfactualBranchRecord, ...]
    abstained: bool
    reason: str
    event_cost: int
    serialized_state_bytes: int
    rolled_back: bool = False
    rollback_reason: str = ""
    durable_mutation_allowed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "sara-counterfactual-branch-result-v1",
            "transaction_id": self.transaction_id,
            "branches": [item.to_dict() for item in self.branches],
            "branch_count": len(self.branches),
            "depth": max((item.depth for item in self.branches), default=0),
            "abstained": self.abstained,
            "reason": self.reason,
            "event_cost": self.event_cost,
            "serialized_state_bytes": self.serialized_state_bytes,
            "rolled_back": self.rolled_back,
            "rollback_reason": self.rollback_reason,
            "durable_mutation_allowed": False,
        }


class BoundedCausalReasoner:
    """Separates temporal correlation, causal candidates, and verified causes."""

    def __init__(
        self,
        *,
        min_confidence: float = 0.65,
        max_branch_depth: int = 3,
        max_branch_count: int = 2,
        max_branch_event_cost: int = 16,
        max_branch_state_bytes: int = 4096,
    ) -> None:
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.max_branch_depth = max(1, int(max_branch_depth))
        self.max_branch_count = max(1, int(max_branch_count))
        self.max_branch_event_cost = max(1, int(max_branch_event_cost))
        self.max_branch_state_bytes = max(256, int(max_branch_state_bytes))

    def infer(self, evidence: Iterable[CausalEvidence]) -> CausalInference:
        rows = tuple(
            item for item in evidence if item.verified and item.observed and item.source_ref
        )
        if not rows:
            return self._abstain(
                "no_verified_source_backed_evidence",
                alternatives=("collect_verified_source_backed_evidence",),
            )
        sources = {item.source for item in rows}
        targets = {item.target for item in rows}
        relation_types = {item.relation_type for item in rows}
        if len(sources) != 1 or len(targets) != 1:
            return self._abstain(
                "multiple_candidate_paths_without_resolution",
                supporting_paths=self._supporting_paths(rows),
                alternatives=("resolve_candidate_path_identity",),
            )
        source, target = next(iter(sources)), next(iter(targets))
        supporting_paths = self._supporting_paths(rows)
        context_tags = tuple(
            sorted(
                {
                    str(tag)
                    for item in rows
                    for tag in item.context_tags
                    if str(tag).strip()
                }
            )
        )
        if any(not item.feedback_stable for item in rows):
            return CausalInference(
                "abstain",
                source,
                target,
                0.0,
                True,
                False,
                supporting_paths,
                ("unstable_feedback", "collect_stable_feedback_revision"),
                "unstable_feedback_freeze",
                context_tags,
            )
        if "contradicts" in relation_types:
            return CausalInference(
                "abstain", source, target, 0.0, True, False,
                supporting_paths,
                ("conflicting_relation", "collect_independent_resolution"),
                "source_conflict_freeze",
                context_tags,
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
            alternatives = ("unobserved_confounder", "context_specific_effect")
        elif {"precedes", "correlates_with", "causes_candidate", "causes_verified"} & relation_types:
            relation = "causes_candidate"
            reason = "insufficient_verified_causal_support"
            alternatives = ("temporal_correlation", "unobserved_confounder")
        else:
            return self._abstain(
                "unsupported_causal_relation",
                supporting_paths=supporting_paths,
                alternatives=("non_causal_relation",),
            )
        return CausalInference(
            relation, source, target, round(max(0.0, min(1.0, confidence)), 6),
            False, False, supporting_paths, alternatives, reason, context_tags,
        )

    def branch_counterfactual(
        self,
        inference: CausalInference,
        *,
        intervention: str,
        predicted_outcome: str,
        alternative_outcome: str,
        depth: int = 1,
        context_tags: Sequence[str] = (),
    ) -> CounterfactualBranchResult:
        bounded_depth = max(1, int(depth))
        transaction_id = self._transaction_id(
            inference,
            intervention=intervention,
            predicted_outcome=predicted_outcome,
            alternative_outcome=alternative_outcome,
            depth=bounded_depth,
        )
        valid_text = all(
            str(value).strip()
            for value in (intervention, predicted_outcome, alternative_outcome)
        )
        if (
            bounded_depth > self.max_branch_depth
            or inference.relation_type != "causes_verified"
            or inference.abstained
            or not valid_text
        ):
            return CounterfactualBranchResult(
                transaction_id=transaction_id,
                branches=(),
                abstained=True,
                reason="causal_support_or_branch_budget_insufficient",
                event_cost=0,
                serialized_state_bytes=0,
            )
        if self.max_branch_count < 2:
            return CounterfactualBranchResult(
                transaction_id=transaction_id,
                branches=(),
                abstained=True,
                reason="branch_count_budget_exceeded",
                event_cost=0,
                serialized_state_bytes=0,
            )
        tags = tuple(
            sorted(
                set(inference.context_tags)
                | {str(item) for item in context_tags if str(item).strip()}
            )
        )
        outcomes = (
            ("predicted", str(predicted_outcome)),
            ("alternative", str(alternative_outcome)),
        )
        branches = tuple(
            CounterfactualBranchRecord(
                branch_id=f"{transaction_id}:{branch_type}",
                transaction_id=transaction_id,
                branch_type=branch_type,
                depth=bounded_depth,
                intervention=str(intervention),
                outcome=outcome,
                supporting_paths=tuple(inference.supporting_paths),
                alternatives=tuple(inference.alternatives),
                context_tags=tags,
            )
            for branch_type, outcome in outcomes
        )
        event_cost = (
            len(branches) * bounded_depth
            + len(inference.supporting_paths)
            + len(branches)
        )
        staged_state_bytes = len(
            json.dumps(
                [item.to_dict() for item in branches],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        rolled_back_state_bytes = len(
            json.dumps(
                [replace(item, status="rolled_back").to_dict() for item in branches],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        serialized_state_bytes = max(staged_state_bytes, rolled_back_state_bytes)
        if (
            event_cost > self.max_branch_event_cost
            or serialized_state_bytes > self.max_branch_state_bytes
        ):
            return CounterfactualBranchResult(
                transaction_id=transaction_id,
                branches=(),
                abstained=True,
                reason="branch_event_or_state_budget_exceeded",
                event_cost=event_cost,
                serialized_state_bytes=serialized_state_bytes,
            )
        return CounterfactualBranchResult(
            transaction_id=transaction_id,
            branches=branches,
            abstained=False,
            reason="verified_causal_branch_staged",
            event_cost=event_cost,
            serialized_state_bytes=serialized_state_bytes,
        )

    @staticmethod
    def rollback_counterfactual(
        result: CounterfactualBranchResult,
        *,
        reason: str,
    ) -> CounterfactualBranchResult:
        rollback_reason = str(reason).strip()
        if result.abstained or not result.branches or not rollback_reason:
            return result
        return CounterfactualBranchResult(
            transaction_id=result.transaction_id,
            branches=tuple(replace(item, status="rolled_back") for item in result.branches),
            abstained=False,
            reason="explicit_counterfactual_rollback",
            event_cost=result.event_cost,
            serialized_state_bytes=result.serialized_state_bytes,
            rolled_back=True,
            rollback_reason=rollback_reason,
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
        result = self.branch_counterfactual(
            inference,
            intervention=intervention,
            predicted_outcome=predicted_outcome,
            alternative_outcome=alternative_outcome,
            depth=depth,
        )
        payload = result.to_dict()
        payload.update(
            {
                "intervention": intervention,
                "predicted_outcome": predicted_outcome,
                "alternative_outcome": alternative_outcome,
                "supporting_paths": list(inference.supporting_paths),
                "alternatives": list(inference.alternatives),
            }
        )
        return payload

    @staticmethod
    def _supporting_paths(rows: Iterable[CausalEvidence]) -> Tuple[str, ...]:
        paths = []
        for item in rows:
            path = "->".join(str(value) for value in item.event_path if str(value).strip())
            paths.append(f"{item.source_ref}::{path}" if path else item.source_ref)
        return tuple(sorted(set(paths)))

    @staticmethod
    def _transaction_id(
        inference: CausalInference,
        *,
        intervention: str,
        predicted_outcome: str,
        alternative_outcome: str,
        depth: int,
    ) -> str:
        digest = evidence_digest(
            {
                "inference": inference.to_dict(),
                "intervention": intervention,
                "predicted_outcome": predicted_outcome,
                "alternative_outcome": alternative_outcome,
                "depth": depth,
            }
        )[:20]
        return f"counterfactual::{digest}"

    @staticmethod
    def _abstain(
        reason: str,
        *,
        supporting_paths: Tuple[str, ...] = (),
        alternatives: Tuple[str, ...] = (),
    ) -> CausalInference:
        return CausalInference(
            "abstain",
            "",
            "",
            0.0,
            True,
            False,
            supporting_paths,
            alternatives,
            reason,
        )


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
