from __future__ import annotations

import json

import pytest

from sara_engine.edge.portable_decision_trace import (
    adapt_event_memory_admission,
    adapt_predictive_feedback,
    adapt_risa_proposal,
    canonical_decision_json,
    decide,
    decision_trace_digest,
)
from sara_engine.memory.event_state_cache import (
    EventStateCandidate,
    VerifiedHierarchicalEventStateCache,
)
from sara_engine.memory.verification_receipt import issue_verification_receipt
from sara_engine.risa.structural_interpolation import (
    PredictiveStructuralFeedbackEngine,
    StructuralEvidence,
    StructuralFeedbackSignal,
    StructuralInterpolationEngine,
)


def _candidate(entry_id: str, *, contradicted: bool = False) -> EventStateCandidate:
    receipt = issue_verification_receipt(
        verifier_id="portable-adapter-test",
        verifier_version="v1",
        decision="verified_fixture",
        evidence={"entry_id": entry_id},
        source_refs=(f"source:{entry_id}",),
        source_revision="revision-1",
        observed=True,
        source_backed=True,
        verified=True,
        contradicted=contradicted,
    )
    return EventStateCandidate(
        entry_id=entry_id,
        signature=(1, 3, 5),
        source_ref=f"source:{entry_id}",
        source_revision="revision-1",
        time_segment=1,
        own_latent_id=f"latent:{entry_id}",
        confidence=0.9,
        uncertainty=0.1,
        source_reliability=0.9,
        resonance_score=0.9,
        metabolic_headroom=0.9,
        observed=True,
        source_backed=True,
        verified=True,
        contradicted=contradicted,
        verification_receipt=receipt,
    )


def test_real_subsystem_outputs_adapt_to_portable_decisions():
    cache = VerifiedHierarchicalEventStateCache()
    admitted = cache.admit(_candidate("accepted"))
    contradicted = cache.admit(_candidate("contradicted", contradicted=True))

    interpolation = StructuralInterpolationEngine().propose(
        (
            StructuralEvidence(
                source_node="column",
                target_node="roof",
                relation_type="supports",
                confidence=0.9,
                source_ref="observation:1",
                source_hash="hash-1",
                source_revision="revision-1",
            ),
            StructuralEvidence(
                source_node="column",
                target_node="roof",
                relation_type="supports",
                confidence=0.85,
                source_ref="observation:2",
                source_hash="hash-2",
                source_revision="revision-1",
            ),
        )
    ).proposals[0]
    feedback_engine = PredictiveStructuralFeedbackEngine()
    retained = feedback_engine.propose(
        (
            StructuralFeedbackSignal(
                predicting_concept="support",
                source_node="column",
                target_node="roof",
                relation_type="supports",
                predicted_confidence=0.8,
                observed_confidence=0.85,
                evidence_ids=("observation:1",),
            ),
        )
    )[0]
    corrected = feedback_engine.propose(
        (
            StructuralFeedbackSignal(
                predicting_concept="support",
                source_node="column",
                target_node="roof",
                relation_type="supports",
                predicted_confidence=0.2,
                observed_confidence=0.9,
                evidence_ids=("observation:2",),
            ),
        )
    )[0]

    records = [
        adapt_event_memory_admission(admitted, sequence=0),
        adapt_event_memory_admission(contradicted, sequence=1),
        adapt_risa_proposal(interpolation, sequence=2),
        adapt_predictive_feedback(retained, sequence=3),
        adapt_predictive_feedback(corrected, sequence=4),
    ]
    assert [decide(record) for record in records] == [
        "admit",
        "reject_contradiction",
        "propose",
        "retain_prediction",
        "emit_correction",
    ]

    rust = pytest.importorskip("sara_engine.sara_rust_core")
    source = json.dumps(records, ensure_ascii=True, separators=(",", ":"))
    assert rust.canonical_portable_decision_trace_json(source) == canonical_decision_json(records)
    assert rust.portable_decision_trace_digest(source) == decision_trace_digest(records)
