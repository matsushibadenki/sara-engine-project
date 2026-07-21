"""Bounded, non-durable structural edit staging with atomic rollback."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any, Dict, Iterable, List, Tuple

from .graph_store import RisaGraphStore
from .models import ConceptCell, ConceptRelation
from .structural_interpolation import StructuralEditProposal


def _canonical_graph(store: RisaGraphStore) -> Dict[str, Any]:
    return {
        "nodes": [
            store.nodes_by_id[key].to_dict() for key in sorted(store.nodes_by_id)
        ],
        "edges": [
            store.edges_by_key[key].to_dict() for key in sorted(store.edges_by_key)
        ],
    }


def graph_digest(store: RisaGraphStore) -> str:
    payload = json.dumps(
        _canonical_graph(store), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StructuralGraphSnapshot:
    digest: str
    node_count: int
    edge_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructuralEditBatchResult:
    batch_id: str
    accepted_for_review: bool
    rolled_back: bool
    rollback_verified: bool
    staged_edit_count: int
    reason: str
    snapshot: StructuralGraphSnapshot
    staged_digest: str
    final_digest: str
    staged_graph: Dict[str, Any] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)
    durable_mutation_allowed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "sara-structural-edit-batch-result-v1",
            "batch_id": self.batch_id,
            "accepted_for_review": self.accepted_for_review,
            "rolled_back": self.rolled_back,
            "rollback_verified": self.rollback_verified,
            "staged_edit_count": self.staged_edit_count,
            "reason": self.reason,
            "snapshot": self.snapshot.to_dict(),
            "staged_digest": self.staged_digest,
            "final_digest": self.final_digest,
            "staged_graph": dict(self.staged_graph),
            "trace": dict(self.trace),
            "durable_mutation_allowed": False,
        }


class BoundedStructuralEditTransaction:
    """Apply proposals to an isolated graph copy and atomically reject failures."""

    def __init__(
        self,
        *,
        max_edits: int = 8,
        max_nodes: int = 128,
        max_edges: int = 256,
    ) -> None:
        self.max_edits = max(1, int(max_edits))
        self.max_nodes = max(1, int(max_nodes))
        self.max_edges = max(1, int(max_edges))

    @staticmethod
    def _remove_edge(store: RisaGraphStore, proposal: StructuralEditProposal) -> bool:
        key = (proposal.source_node, proposal.target_node, proposal.relation_type)
        if key not in store.edges_by_key:
            return False
        del store.edges_by_key[key]
        store.adjacency_out[proposal.source_node].discard(
            (proposal.target_node, proposal.relation_type)
        )
        store.adjacency_in[proposal.target_node].discard(
            (proposal.source_node, proposal.relation_type)
        )
        return True

    @staticmethod
    def _apply(
        store: RisaGraphStore,
        proposal: StructuralEditProposal,
        snapshot_digest: str,
    ) -> str:
        if proposal.frozen or proposal.edit_type in {
            "freeze_subgraph",
            "request_more_evidence",
        }:
            return "non_editing_or_frozen_proposal"
        if not proposal.evidence_ids:
            return "missing_evidence"
        if proposal.rollback_state not in {"verified_snapshot", snapshot_digest}:
            return "stale_rollback_snapshot"
        if proposal.edit_type == "create_provisional_node":
            if not proposal.target_node or store.get_node(proposal.target_node) is not None:
                return "provisional_target_invalid_or_exists"
            store.add_or_update_node(
                ConceptCell(
                    cell_id=proposal.target_node,
                    kind=proposal.provisional_node_kind or "provisional_concept",
                    label=proposal.provisional_node_label or proposal.target_node,
                    attributes={
                        "admission_state": "provisional_review_only",
                        "proposal_id": proposal.proposal_id,
                        "evidence_ids": "|".join(sorted(set(proposal.evidence_ids))),
                    },
                    stability=0.0,
                    energy=0.0,
                )
            )
            return ""
        if proposal.edit_type == "strengthen_relation":
            if (
                store.get_node(proposal.source_node) is None
                or store.get_node(proposal.target_node) is None
            ):
                return "relation_endpoint_missing"
            store.add_or_update_edge(
                ConceptRelation(
                    source=proposal.source_node,
                    target=proposal.target_node,
                    relation_type=proposal.relation_type,
                    context_tags=proposal.context_tags,
                    evidence_count=max(1, len(proposal.evidence_ids)),
                    reliability=max(0.0, min(1.0, proposal.confidence)),
                )
            )
            return ""
        if proposal.edit_type == "cut_relation":
            return "" if BoundedStructuralEditTransaction._remove_edge(store, proposal) else "relation_not_found"
        return "unsupported_edit_type"

    def stage(
        self,
        store: RisaGraphStore,
        proposals: Iterable[StructuralEditProposal],
    ) -> StructuralEditBatchResult:
        proposal_rows: Tuple[StructuralEditProposal, ...] = tuple(proposals)
        snapshot_digest = graph_digest(store)
        snapshot = StructuralGraphSnapshot(
            digest=snapshot_digest,
            node_count=len(store.nodes_by_id),
            edge_count=len(store.edges_by_key),
        )
        batch_digest = sha256(
            "|".join(item.proposal_id for item in proposal_rows).encode("utf-8")
        ).hexdigest()[:16]
        working = deepcopy(store)
        staged_count = 0
        reason = ""
        seen_ids = set()
        if not proposal_rows:
            reason = "empty_edit_batch"
        elif len(proposal_rows) > self.max_edits:
            reason = "edit_budget_exceeded"
        for proposal in proposal_rows if not reason else ():
            if proposal.proposal_id in seen_ids:
                reason = "duplicate_proposal_id"
                break
            seen_ids.add(proposal.proposal_id)
            reason = self._apply(working, proposal, snapshot_digest)
            if reason:
                break
            staged_count += 1
            if len(working.nodes_by_id) > self.max_nodes:
                reason = "node_budget_exceeded"
                break
            if len(working.edges_by_key) > self.max_edges:
                reason = "edge_budget_exceeded"
                break

        accepted = not reason and staged_count == len(proposal_rows)
        staged_digest = graph_digest(working)
        rolled_back = not accepted
        if rolled_back:
            working = deepcopy(store)
        final_digest = graph_digest(working)
        original_unchanged = graph_digest(store) == snapshot_digest
        rollback_verified = original_unchanged and (
            not rolled_back or final_digest == snapshot_digest
        )
        return StructuralEditBatchResult(
            batch_id=f"structural-edit-batch::{batch_digest}",
            accepted_for_review=accepted,
            rolled_back=rolled_back,
            rollback_verified=rollback_verified,
            staged_edit_count=staged_count,
            reason=reason or "bounded_batch_staged_for_review",
            snapshot=snapshot,
            staged_digest=staged_digest,
            final_digest=final_digest,
            staged_graph=_canonical_graph(working) if accepted else {},
            trace={
                "proposal_count": len(proposal_rows),
                "max_edits": self.max_edits,
                "max_nodes": self.max_nodes,
                "max_edges": self.max_edges,
                "original_graph_unchanged": original_unchanged,
                "review_required": True,
            },
        )


__all__ = [
    "BoundedStructuralEditTransaction",
    "StructuralEditBatchResult",
    "StructuralGraphSnapshot",
    "graph_digest",
]
