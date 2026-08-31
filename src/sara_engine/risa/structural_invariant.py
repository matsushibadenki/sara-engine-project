"""Bounded, evidence-linked canonical motif sharing for Phase 37 research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class MotifEdge:
    source: str
    relation_type: str
    target: str
    evidence_id: str
    verified: bool = True


@dataclass(frozen=True)
class TypedMotifPattern:
    pattern_id: str
    topology_signature: Tuple[Tuple[int, int, int, int], ...]
    relation_signature: Tuple[str, ...]
    evidence_ids: Tuple[str, ...]
    exemplar_ids: Tuple[str, ...]
    context: str


@dataclass(frozen=True)
class MotifProposal:
    relation_type: str
    source_role: str
    target_role: str
    confidence: float
    pattern_id: str
    evidence_ids: Tuple[str, ...]
    durable_mutation_allowed: bool = False


@dataclass(frozen=True)
class MotifMatchResult:
    proposals: Tuple[MotifProposal, ...]
    abstained: bool
    reason: str
    candidate_pattern_count: int
    event_cost: int
    state_bytes: int
    trace: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "sara-phase37-motif-match-v1",
            "proposals": [asdict(item) | {"evidence_ids": list(item.evidence_ids)} for item in self.proposals],
            "abstained": self.abstained,
            "reason": self.reason,
            "candidate_pattern_count": self.candidate_pattern_count,
            "event_cost": self.event_cost,
            "state_bytes": self.state_bytes,
            "trace": dict(self.trace),
        }


def _topology(edges: Sequence[MotifEdge]) -> Tuple[Tuple[int, int, int, int], ...]:
    outgoing: Dict[str, int] = {}
    incoming: Dict[str, int] = {}
    for edge in edges:
        outgoing[edge.source] = outgoing.get(edge.source, 0) + 1
        incoming[edge.target] = incoming.get(edge.target, 0) + 1
    return tuple(sorted((outgoing.get(e.source, 0), incoming.get(e.source, 0), outgoing.get(e.target, 0), incoming.get(e.target, 0)) for e in edges))


class CanonicalTypedMotifStore:
    """Research-only sparse store; never mutates a durable RISA graph."""

    def __init__(self, *, max_patterns: int = 64, max_edges_per_pattern: int = 6, max_candidate_patterns: int = 8) -> None:
        self.max_patterns = max(1, int(max_patterns))
        self.max_edges_per_pattern = max(1, int(max_edges_per_pattern))
        self.max_candidate_patterns = max(1, int(max_candidate_patterns))
        self._patterns: List[TypedMotifPattern] = []

    @property
    def patterns(self) -> Tuple[TypedMotifPattern, ...]:
        return tuple(self._patterns)

    def observe(self, exemplar_id: str, edges: Iterable[MotifEdge], *, context: str) -> bool:
        verified = tuple(edge for edge in edges if edge.verified)[: self.max_edges_per_pattern]
        if not verified or len(self._patterns) >= self.max_patterns:
            return False
        topology = _topology(verified)
        relations = tuple(sorted(edge.relation_type for edge in verified))
        evidence = tuple(sorted({edge.evidence_id for edge in verified if edge.evidence_id}))
        digest = sha256(repr((topology, relations, context)).encode()).hexdigest()[:16]
        self._patterns.append(TypedMotifPattern(f"motif::{digest}", topology, relations, evidence, (exemplar_id,), context))
        return True

    def propose(self, edges: Iterable[MotifEdge], *, context: str, context_aware: bool, shuffled_binding: bool = False, max_proposals: int = 4) -> MotifMatchResult:
        visible = tuple(edge for edge in edges if edge.verified)[: self.max_edges_per_pattern]
        if not visible:
            return MotifMatchResult((), True, "missing_roles", 0, 0, self._state_bytes(), {"durable_mutation_allowed": False})
        visible_topology = _topology(visible)
        candidates = []
        for pattern in self._patterns[: self.max_candidate_patterns]:
            event_overlap = sum(1 for descriptor in visible_topology if descriptor in pattern.topology_signature)
            topology_score = event_overlap / max(1, len(visible_topology))
            relation_overlap = len(set(e.relation_type for e in visible) & set(pattern.relation_signature)) / max(1, len(set(e.relation_type for e in visible) | set(pattern.relation_signature)))
            score = topology_score if not context_aware else 0.6 * topology_score + 0.4 * relation_overlap
            if context_aware and context != pattern.context:
                score *= 0.25
            candidates.append((score, pattern))
        candidates.sort(key=lambda item: (-item[0], item[1].pattern_id))
        proposals: List[MotifProposal] = []
        if candidates and candidates[0][0] >= 0.8:
            score, pattern = candidates[0]
            known = {edge.relation_type for edge in visible}
            missing = [relation for relation in pattern.relation_signature if relation not in known]
            if missing:
                relation = missing[-1] if shuffled_binding else missing[0]
                proposals.append(MotifProposal(relation, "role:source", "role:target", round(score, 6), pattern.pattern_id, pattern.evidence_ids))
        cost = len(visible) * min(len(self._patterns), self.max_candidate_patterns)
        return MotifMatchResult(tuple(proposals[:max_proposals]), not proposals, "no_bounded_pattern_match" if not proposals else "provisional_pattern_transfer", len(candidates), cost, self._state_bytes(), {"context_aware": context_aware, "shuffled_binding": shuffled_binding, "durable_mutation_allowed": False})

    def _state_bytes(self) -> int:
        return sum(32 + 16 * len(p.topology_signature) + sum(len(x) for x in p.relation_signature + p.evidence_ids) for p in self._patterns)


__all__ = ["CanonicalTypedMotifStore", "MotifEdge", "MotifMatchResult", "MotifProposal", "TypedMotifPattern"]
