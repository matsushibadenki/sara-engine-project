"""Observed-only structural interpolation proposals for RISA.

The engine aggregates repeated, source-backed relation evidence without
mutating the durable graph.  It is intentionally a proposal layer: Concept
Review and Event Memory admission remain the only paths to durable knowledge.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class StructuralEvidence:
    source_node: str
    target_node: str
    relation_type: str
    confidence: float
    source_ref: str
    source_hash: str
    source_revision: str
    context_tags: Tuple[str, ...] = ()
    acquired_at: int = 0
    contradiction_count: int = 0
    expiry_segment: Optional[int] = None
    metabolic_cost: int = 0
    verified: bool = True


@dataclass(frozen=True)
class StructuralInterpolationProposal:
    proposal_id: str
    action: str
    source_node: str
    target_node: str
    relation_type: str
    context_tags: Tuple[str, ...]
    confidence_before: float
    confidence_after: float
    confidence_delta: float
    evidence_count: int
    distinct_source_count: int
    source_refs: Tuple[str, ...]
    source_hashes: Tuple[str, ...]
    source_revisions: Tuple[str, ...]
    acquired_at_min: int
    acquired_at_max: int
    expiry_segment: Optional[int]
    contradiction_count: int
    metabolic_cost: int
    durable_mutation_allowed: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in ("context_tags", "source_refs", "source_hashes", "source_revisions"):
            payload[key] = list(payload[key])
        return payload


@dataclass(frozen=True)
class StructuralInterpolationResult:
    proposals: Tuple[StructuralInterpolationProposal, ...]
    rejected_count: int
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "sara-structural-interpolation-result-v1",
            "observed_only": True,
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "rejected_count": self.rejected_count,
            "trace": dict(self.trace),
        }


@dataclass(frozen=True)
class StructuralFeedbackSignal:
    """Typed mismatch between an upper hypothesis and lower evidence."""

    predicting_concept: str
    source_node: str
    target_node: str
    relation_type: str
    predicted_confidence: float
    observed_confidence: float
    evidence_ids: Tuple[str, ...] = ()
    context_tags: Tuple[str, ...] = ()
    contradiction_count: int = 0
    recent_actions: Tuple[str, ...] = ()
    eligible: bool = True
    rollback_state: str = "verified_snapshot"
    target_exists: bool = True
    provisional_node_kind: str = "provisional_concept"
    provisional_node_label: str = ""


@dataclass(frozen=True)
class StructuralEditProposal:
    proposal_id: str
    edit_type: str
    predicting_concept: str
    source_node: str
    target_node: str
    relation_type: str
    confidence: float
    prediction_error: float
    evidence_ids: Tuple[str, ...]
    context_tags: Tuple[str, ...]
    rollback_state: str
    frozen: bool
    provisional_node_kind: str = ""
    provisional_node_label: str = ""
    durable_mutation_allowed: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["evidence_ids"] = list(self.evidence_ids)
        payload["context_tags"] = list(self.context_tags)
        return payload


class PredictiveStructuralFeedbackEngine:
    """Convert upper/lower mismatches into bounded, non-mutating edit hints."""

    def __init__(self, *, mismatch_threshold: float = 0.25, max_proposals: int = 32) -> None:
        self.mismatch_threshold = _clamp01(mismatch_threshold)
        self.max_proposals = max(1, int(max_proposals))

    @staticmethod
    def _is_oscillating(actions: Tuple[str, ...]) -> bool:
        meaningful = tuple(item for item in actions if item in {"strengthen_relation", "cut_relation"})
        return len(meaningful) >= 4 and any(
            meaningful[index] != meaningful[index - 1] and meaningful[index] == meaningful[index - 2]
            for index in range(2, len(meaningful))
        )

    def propose(self, signals: Iterable[StructuralFeedbackSignal]) -> Tuple[StructuralEditProposal, ...]:
        proposals: List[StructuralEditProposal] = []
        for signal in signals:
            if len(proposals) >= self.max_proposals:
                break
            predicted = _clamp01(signal.predicted_confidence)
            observed = _clamp01(signal.observed_confidence)
            error = round(observed - predicted, 6)
            oscillating = self._is_oscillating(tuple(signal.recent_actions))
            if oscillating or signal.contradiction_count > 0:
                edit_type = "freeze_subgraph"
                reason = "oscillating_feedback" if oscillating else "verified_contradiction"
                frozen = True
            elif not signal.eligible or not signal.evidence_ids:
                edit_type = "request_more_evidence"
                reason = "ineligible_or_missing_evidence"
                frozen = False
            elif not signal.target_exists:
                edit_type = "create_provisional_node"
                reason = "source_backed_unknown_target"
                frozen = False
            elif error >= self.mismatch_threshold:
                edit_type = "strengthen_relation"
                reason = "observed_support_exceeds_prediction"
                frozen = False
            elif error <= -self.mismatch_threshold:
                edit_type = "cut_relation"
                reason = "observed_evidence_disconfirms_prediction"
                frozen = False
            else:
                edit_type = "request_more_evidence"
                reason = "mismatch_below_edit_threshold"
                frozen = False
            digest = sha256(
                "|".join(
                    (
                        signal.predicting_concept,
                        signal.source_node,
                        signal.target_node,
                        signal.relation_type,
                        edit_type,
                    )
                ).encode("utf-8")
            ).hexdigest()[:16]
            proposals.append(
                StructuralEditProposal(
                    proposal_id=f"structural-feedback::{digest}",
                    edit_type=edit_type,
                    predicting_concept=signal.predicting_concept,
                    source_node=signal.source_node,
                    target_node=signal.target_node,
                    relation_type=signal.relation_type,
                    confidence=round(max(predicted, observed), 6),
                    prediction_error=error,
                    evidence_ids=tuple(signal.evidence_ids),
                    context_tags=tuple(sorted(set(signal.context_tags))),
                    rollback_state=signal.rollback_state,
                    frozen=frozen,
                    provisional_node_kind=(
                        signal.provisional_node_kind
                        if edit_type == "create_provisional_node"
                        else ""
                    ),
                    provisional_node_label=(
                        signal.provisional_node_label
                        if edit_type == "create_provisional_node"
                        else ""
                    ),
                    reason=reason,
                )
            )
        return tuple(proposals)


class StructuralInterpolationEngine:
    """Aggregate independent relation evidence into bounded edit proposals."""

    def __init__(self, *, max_proposals: int = 64, max_evidence_per_relation: int = 8) -> None:
        self.max_proposals = max(1, int(max_proposals))
        self.max_evidence_per_relation = max(1, int(max_evidence_per_relation))

    def propose(
        self,
        evidence: Iterable[StructuralEvidence],
        *,
        current_segment: Optional[int] = None,
    ) -> StructuralInterpolationResult:
        grouped: Dict[Tuple[str, str, str, Tuple[str, ...]], List[StructuralEvidence]] = defaultdict(list)
        rejected = 0
        for item in evidence:
            if not item.verified or not item.source_hash or not item.source_revision:
                rejected += 1
                continue
            if current_segment is not None and item.expiry_segment is not None and item.expiry_segment < current_segment:
                rejected += 1
                continue
            key = (item.source_node, item.target_node, item.relation_type, tuple(sorted(set(item.context_tags))))
            bucket = grouped[key]
            if len(bucket) < self.max_evidence_per_relation:
                bucket.append(item)
            else:
                rejected += 1

        proposals: List[StructuralInterpolationProposal] = []
        for key, items in sorted(grouped.items()):
            source_hashes = tuple(sorted({item.source_hash for item in items}))
            contradictions = sum(max(0, int(item.contradiction_count)) for item in items)
            if len(source_hashes) < 2 or contradictions:
                rejected += 1
                continue
            confidence_before = min(_clamp01(item.confidence) for item in items)
            mean_confidence = sum(_clamp01(item.confidence) for item in items) / len(items)
            confidence_after = _clamp01(mean_confidence + min(0.15, 0.05 * (len(source_hashes) - 1)))
            source_node, target_node, relation_type, context_tags = key
            digest = sha256("|".join(key[:3] + (";".join(context_tags),)).encode("utf-8")).hexdigest()[:16]
            proposals.append(
                StructuralInterpolationProposal(
                    proposal_id=f"structural-interpolation::{digest}",
                    action="merge_candidate" if len(items) > 1 else "strengthen_relation",
                    source_node=source_node,
                    target_node=target_node,
                    relation_type=relation_type,
                    context_tags=context_tags,
                    confidence_before=round(confidence_before, 6),
                    confidence_after=round(confidence_after, 6),
                    confidence_delta=round(confidence_after - confidence_before, 6),
                    evidence_count=len(items),
                    distinct_source_count=len(source_hashes),
                    source_refs=tuple(sorted({item.source_ref for item in items if item.source_ref})),
                    source_hashes=source_hashes,
                    source_revisions=tuple(sorted({item.source_revision for item in items})),
                    acquired_at_min=min(item.acquired_at for item in items),
                    acquired_at_max=max(item.acquired_at for item in items),
                    expiry_segment=min(
                        (item.expiry_segment for item in items if item.expiry_segment is not None),
                        default=None,
                    ),
                    contradiction_count=contradictions,
                    metabolic_cost=sum(max(0, int(item.metabolic_cost)) for item in items),
                    reason="independent_verified_evidence_agrees",
                )
            )
            if len(proposals) >= self.max_proposals:
                rejected += sum(len(grouped[item_key]) for item_key in list(sorted(grouped))[len(proposals):])
                break

        return StructuralInterpolationResult(
            proposals=tuple(proposals),
            rejected_count=rejected,
            trace={
                "group_count": len(grouped),
                "proposal_count": len(proposals),
                "max_proposals": self.max_proposals,
                "max_evidence_per_relation": self.max_evidence_per_relation,
                "current_segment": current_segment,
                "durable_mutation_allowed": False,
            },
        )


__all__ = [
    "StructuralEvidence",
    "StructuralEditProposal",
    "StructuralFeedbackSignal",
    "StructuralInterpolationEngine",
    "StructuralInterpolationProposal",
    "StructuralInterpolationResult",
    "PredictiveStructuralFeedbackEngine",
]
