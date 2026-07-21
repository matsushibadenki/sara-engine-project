from __future__ import annotations

from dataclasses import replace

from sara_engine.risa.graph_store import RisaGraphStore
from sara_engine.risa.models import ConceptCell
from sara_engine.risa.structural_edit_transaction import (
    BoundedStructuralEditTransaction,
    graph_digest,
)
from sara_engine.risa.structural_interpolation import StructuralEditProposal


def _proposal(proposal_id: str, edit_type: str, **overrides):
    values = {
        "proposal_id": proposal_id,
        "edit_type": edit_type,
        "predicting_concept": "concept:animal",
        "source_node": "concept:animal",
        "target_node": "concept:unknown",
        "relation_type": "predicts",
        "confidence": 0.8,
        "prediction_error": 0.3,
        "evidence_ids": ("evidence-1", "evidence-2"),
        "context_tags": ("biology",),
        "rollback_state": "verified_snapshot",
        "frozen": False,
        "provisional_node_kind": "animal_candidate",
        "provisional_node_label": "unknown animal",
    }
    values.update(overrides)
    return StructuralEditProposal(**values)


def _store():
    store = RisaGraphStore()
    store.add_or_update_node(
        ConceptCell(cell_id="concept:animal", kind="concept", label="animal")
    )
    return store


def test_structural_edit_batch_stages_multiple_edits_without_mutating_durable_graph():
    store = _store()
    before = graph_digest(store)
    proposals = (
        _proposal(
            "proposal:create", "create_provisional_node", rollback_state=before
        ),
        _proposal("proposal:link", "strengthen_relation", rollback_state=before),
    )

    result = BoundedStructuralEditTransaction().stage(store, proposals)

    assert result.accepted_for_review is True
    assert result.rolled_back is False
    assert result.rollback_verified is True
    assert result.staged_edit_count == 2
    assert result.staged_graph["nodes"][1]["attributes"]["admission_state"] == "provisional_review_only"
    assert graph_digest(store) == before
    assert store.get_node("concept:unknown") is None
    assert result.durable_mutation_allowed is False


def test_structural_edit_batch_rolls_back_all_prior_edits_after_late_failure():
    store = _store()
    before = graph_digest(store)
    invalid_link = _proposal(
        "proposal:invalid-link",
        "strengthen_relation",
        source_node="concept:missing-source",
    )

    result = BoundedStructuralEditTransaction().stage(
        store,
        (_proposal("proposal:create", "create_provisional_node"), invalid_link),
    )

    assert result.accepted_for_review is False
    assert result.rolled_back is True
    assert result.rollback_verified is True
    assert result.staged_edit_count == 1
    assert result.reason == "relation_endpoint_missing"
    assert result.final_digest == before
    assert graph_digest(store) == before


def test_structural_edit_batch_rejects_budget_and_stale_snapshot():
    store = _store()
    proposal = _proposal("proposal:create", "create_provisional_node")
    budget_result = BoundedStructuralEditTransaction(max_edits=1).stage(
        store, (proposal, replace(proposal, proposal_id="proposal:second"))
    )
    stale_result = BoundedStructuralEditTransaction().stage(
        store, (replace(proposal, rollback_state="stale-digest"),)
    )

    assert budget_result.reason == "edit_budget_exceeded"
    assert budget_result.rollback_verified is True
    assert stale_result.reason == "stale_rollback_snapshot"
    assert stale_result.rollback_verified is True
