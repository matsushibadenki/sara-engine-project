from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from .models import ConceptCell, ConceptRelation


class RisaGraphStore:
    def __init__(self) -> None:
        self.nodes_by_id: Dict[str, ConceptCell] = {}
        self.edges_by_key: Dict[Tuple[str, str, str], ConceptRelation] = {}
        self.adjacency_out: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        self.adjacency_in: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

    def add_or_update_node(self, node: ConceptCell) -> ConceptCell:
        existing = self.nodes_by_id.get(node.cell_id)
        if existing is None:
            self.nodes_by_id[node.cell_id] = node
            return node
        existing.usage_count += 1
        existing.stability = max(existing.stability, node.stability)
        existing.recent_activity = max(existing.recent_activity, node.recent_activity)
        existing.energy = max(existing.energy, node.energy)
        existing.dormant = existing.dormant and node.dormant
        if node.attributes:
            existing.attributes.update(node.attributes)
        return existing

    def add_or_update_edge(self, edge: ConceptRelation) -> ConceptRelation:
        key = (edge.source, edge.target, edge.relation_type)
        existing = self.edges_by_key.get(key)
        if existing is None:
            edge.evidence_count = max(1, int(edge.evidence_count))
            self.edges_by_key[key] = edge
            self.adjacency_out[edge.source].add((edge.target, edge.relation_type))
            self.adjacency_in[edge.target].add((edge.source, edge.relation_type))
            return edge
        existing.evidence_count += max(1, int(edge.evidence_count))
        existing.last_updated = max(existing.last_updated, edge.last_updated)
        existing.reliability = max(existing.reliability, edge.reliability)
        existing.context_tags = tuple(sorted(set(existing.context_tags) | set(edge.context_tags)))
        return existing

    def get_node(self, node_id: str) -> ConceptCell | None:
        return self.nodes_by_id.get(node_id)

    def degree_in(self, node_id: str) -> int:
        return len(self.adjacency_in.get(node_id, set()))

    def degree_out(self, node_id: str) -> int:
        return len(self.adjacency_out.get(node_id, set()))

    def outgoing(self, node_id: str) -> List[ConceptRelation]:
        edges: List[ConceptRelation] = []
        for target, relation_type in self.adjacency_out.get(node_id, set()):
            edge = self.edges_by_key.get((node_id, target, relation_type))
            if edge is not None:
                edges.append(edge)
        return edges

    def to_dict(self) -> dict:
        return {
            "nodes": [node.to_dict() for node in self.nodes_by_id.values()],
            "edges": [edge.to_dict() for edge in self.edges_by_key.values()],
        }
