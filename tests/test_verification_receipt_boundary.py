from __future__ import annotations

from dataclasses import replace

from sara_engine.memory.event_state_cache import EventStateCandidate, VerifiedHierarchicalEventStateCache
from sara_engine.memory.verification_receipt import issue_verification_receipt


def _candidate(*, receipt=True):
    source_ref = "fixture:receipt"
    source_revision = "revision-v1"
    verification_receipt = issue_verification_receipt(
        verifier_id="receipt-boundary-test",
        verifier_version="v1",
        decision="verified_fixture",
        evidence={"event": "observed"},
        source_refs=(source_ref,),
        source_revision=source_revision,
        observed=True,
        source_backed=True,
        verified=True,
    ) if receipt else None
    return EventStateCandidate(
        entry_id="receipt-test",
        signature=(1, 2),
        source_ref=source_ref,
        source_revision=source_revision,
        time_segment=1,
        resonance_score=0.9,
        observed=True,
        source_backed=True,
        verified=True,
        verification_receipt=verification_receipt,
    )


def test_event_memory_rejects_verified_boolean_without_receipt():
    result = VerifiedHierarchicalEventStateCache().admit(_candidate(receipt=False))

    assert result.accepted is False
    assert result.decision == "block_missing_verification_receipt"


def test_event_memory_rejects_tampered_receipt():
    candidate = _candidate()
    tampered = replace(
        candidate,
        verification_receipt=replace(
            candidate.verification_receipt,
            source_revision="tampered",
        ),
    )

    result = VerifiedHierarchicalEventStateCache().admit(tampered)

    assert result.accepted is False
    assert result.decision == "block_invalid_verification_receipt"
