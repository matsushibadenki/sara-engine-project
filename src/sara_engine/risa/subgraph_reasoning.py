"""Bounded, observed-only composition and analogy over verified RISA edges."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class SubgraphEdge:
    source: str
    target: str
    relation_type: str
    confidence: float = 0.0
    evidence_count: int = 0
    context_tags: Tuple[str, ...] = ()
    verified: bool = True


@dataclass(frozen=True)
class ComposedRelationProposal:
    proposal_id: str
    source: str
    target: str
    relation_type: str
    path: Tuple[Tuple[str, str, str], ...]
    confidence: float
    evidence_count: int
    context_tags: Tuple[str, ...]
    durable_mutation_allowed: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["path"] = [list(item) for item in self.path]
        payload["context_tags"] = list(self.context_tags)
        return payload


@dataclass(frozen=True)
class SubgraphCompositionResult:
    proposals: Tuple[ComposedRelationProposal, ...]
    abstained: bool
    reason: str
    event_cost: int
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "sara-bounded-subgraph-composition-v1",
            "observed_only": True,
            "proposals": [item.to_dict() for item in self.proposals],
            "abstained": self.abstained,
            "reason": self.reason,
            "event_cost": self.event_cost,
            "trace": dict(self.trace),
        }


@dataclass(frozen=True)
class StructuralAnalogyResult:
    score: float
    matched_relation_types: Tuple[str, ...]
    compared_edge_count: int
    abstained: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["matched_relation_types"] = list(self.matched_relation_types)
        return payload


class BoundedSubgraphComposer:
    """Compose only recent, verified, bounded paths into provisional proposals."""

    def __init__(self, *, max_hops: int = 3, max_paths: int = 8, min_confidence: float = 0.5) -> None:
        self.max_hops = max(1, int(max_hops))
        self.max_paths = max(1, int(max_paths))
        self.min_confidence = _clamp01(min_confidence)

    def compose(
        self,
        edges: Iterable[SubgraphEdge],
        *,
        source: str,
        target: str,
        context_tags: Sequence[str] = (),
    ) -> SubgraphCompositionResult:
        edge_list = tuple(edges)
        outgoing: Dict[str, List[SubgraphEdge]] = {}
        for edge in edge_list:
            if not edge.verified or _clamp01(edge.confidence) < self.min_confidence:
                continue
            if context_tags and not set(context_tags).issubset(set(edge.context_tags)):
                continue
            outgoing.setdefault(edge.source, []).append(edge)
        paths: List[Tuple[SubgraphEdge, ...]] = []

        def walk(node: str, path: Tuple[SubgraphEdge, ...], visited: Tuple[str, ...]) -> None:
            if len(paths) >= self.max_paths or len(path) >= self.max_hops:
                return
            for edge in sorted(outgoing.get(node, ()), key=lambda item: (item.target, item.relation_type)):
                if edge.target in visited:
                    continue
                next_path = path + (edge,)
                if edge.target == target:
                    paths.append(next_path)
                    continue
                walk(edge.target, next_path, visited + (edge.target,))

        walk(source, (), (source,))
        proposals: List[ComposedRelationProposal] = []
        for path in paths:
            relation_chain = tuple(edge.relation_type for edge in path)
            digest = sha256("|".join((source, target, *relation_chain)).encode("utf-8")).hexdigest()[:16]
            proposals.append(
                ComposedRelationProposal(
                    proposal_id=f"subgraph-composition::{digest}",
                    source=source,
                    target=target,
                    relation_type="composed::" + "+".join(relation_chain),
                    path=tuple((edge.source, edge.target, edge.relation_type) for edge in path),
                    confidence=round(min(_clamp01(edge.confidence) for edge in path), 6),
                    evidence_count=min(max(1, int(edge.evidence_count)) for edge in path),
                    context_tags=tuple(sorted(set.intersection(*(set(edge.context_tags) for edge in path)))) if path else (),
                    reason="verified_bounded_path",
                )
            )
        event_cost = len(edge_list) + sum(len(path) for path in paths)
        return SubgraphCompositionResult(
            proposals=tuple(proposals),
            abstained=not proposals,
            reason="verified_path_not_found" if not proposals else "verified_path_found",
            event_cost=event_cost,
            trace={
                "input_edge_count": len(edge_list),
                "eligible_edge_count": sum(len(items) for items in outgoing.values()),
                "path_count": len(paths),
                "max_hops": self.max_hops,
                "max_paths": self.max_paths,
                "durable_mutation_allowed": False,
            },
        )


class StructuralAnalogyEngine:
    """Compare relation signatures without comparing dense node embeddings."""

    def __init__(self, *, min_score: float = 0.5, max_edges: int = 32) -> None:
        self.min_score = _clamp01(min_score)
        self.max_edges = max(1, int(max_edges))

    def compare(self, left: Iterable[SubgraphEdge], right: Iterable[SubgraphEdge]) -> StructuralAnalogyResult:
        left_edges = tuple(edge for edge in left if edge.verified)[: self.max_edges]
        right_edges = tuple(edge for edge in right if edge.verified)[: self.max_edges]
        left_types = {edge.relation_type for edge in left_edges}
        right_types = {edge.relation_type for edge in right_edges}
        union = left_types | right_types
        intersection = left_types & right_types
        score = len(intersection) / float(max(1, len(union)))
        abstained = not left_edges or not right_edges or score < self.min_score
        return StructuralAnalogyResult(
            score=round(score, 6),
            matched_relation_types=tuple(sorted(intersection)),
            compared_edge_count=len(left_edges) + len(right_edges),
            abstained=abstained,
            reason="insufficient_structural_overlap" if abstained else "relation_signature_overlap",
        )


__all__ = [
    "BoundedSubgraphComposer",
    "ComposedRelationProposal",
    "StructuralAnalogyEngine",
    "StructuralAnalogyResult",
    "SubgraphCompositionResult",
    "SubgraphEdge",
]
