from __future__ import annotations

from sara_engine.multimodal.relation_hypothesis import (
    BoundedCrossModalHypothesisLedger,
)
from sara_engine.multimodal.structural_verification import (
    ModalityEvidence,
    MultimodalStructuralVerifier,
)
from sara_engine.risa.adapters import observation_from_cross_modal_hypothesis


EXPECTED = ("audio", "vision")


def _evidence(source: str, *, contradiction: bool = False, claim: str = "impact"):
    return (
        ModalityEvidence(
            "vision", "contact_motion", 10.0, f"{source}:vision", claim_key=claim
        ),
        ModalityEvidence(
            "audio",
            "impact_sound" if not contradiction else "silence",
            18.0,
            f"{source}:audio",
            claim_key=claim if not contradiction else f"no-{claim}",
        ),
    )


def _observe(ledger, source, revision="v1", *, contradiction=False):
    evidence = _evidence(source, contradiction=contradiction)
    decision = MultimodalStructuralVerifier().verify(
        evidence, expected_modalities=EXPECTED
    )
    return ledger.observe(
        claim_key="impact",
        decision=decision,
        evidence=evidence,
        expected_modalities=EXPECTED,
        observation_source_id=source,
        source_revision=revision,
    )


def test_cross_modal_hypothesis_requires_repeated_independent_sources():
    ledger = BoundedCrossModalHypothesisLedger()

    first = _observe(ledger, "session-a")
    same_source_revision = _observe(ledger, "session-a", revision="v2")
    second_source = _observe(ledger, "session-b")

    assert first.hypothesis is not None
    assert first.hypothesis.state == "provisional_hypothesis"
    assert same_source_revision.hypothesis is not None
    assert same_source_revision.hypothesis.distinct_source_count == 1
    assert same_source_revision.hypothesis.eligible_for_review is False
    assert second_source.hypothesis is not None
    assert second_source.hypothesis.state == "eligible_for_review"
    assert second_source.hypothesis.distinct_source_count == 2
    assert second_source.hypothesis.durable_mutation_allowed is False


def test_cross_modal_hypothesis_rejects_duplicates_and_stale_receipts():
    ledger = BoundedCrossModalHypothesisLedger()
    accepted = _observe(ledger, "session-a")
    duplicate = _observe(ledger, "session-a")
    assert accepted.accepted is True
    assert duplicate.accepted is False
    assert duplicate.reason == "duplicate_observation"

    evidence = _evidence("session-b")
    decision = MultimodalStructuralVerifier().verify(
        evidence, expected_modalities=EXPECTED
    )
    altered = evidence + (
        ModalityEvidence("tactile", "contact", 12.0, "session-b:tactile", claim_key="impact"),
    )
    stale = ledger.observe(
        claim_key="impact",
        decision=decision,
        evidence=altered,
        expected_modalities=EXPECTED,
        observation_source_id="session-b",
        source_revision="v1",
    )

    assert stale.accepted is False
    assert stale.reason == "invalid_or_stale_verification_receipt"


def test_cross_modal_hypothesis_rejects_relabelled_source_identity():
    ledger = BoundedCrossModalHypothesisLedger()
    _observe(ledger, "session-a")
    evidence = _evidence("session-a")
    decision = MultimodalStructuralVerifier().verify(
        evidence, expected_modalities=EXPECTED
    )

    reused = ledger.observe(
        claim_key="impact",
        decision=decision,
        evidence=evidence,
        expected_modalities=EXPECTED,
        observation_source_id="claimed-session-b",
        source_revision="v1",
    )

    assert reused.accepted is False
    assert reused.reason == "source_ref_reuse_across_independent_observations"


def test_cross_modal_contradiction_freezes_hypothesis():
    ledger = BoundedCrossModalHypothesisLedger()
    _observe(ledger, "session-a")
    result = _observe(ledger, "session-b", contradiction=True)

    assert result.hypothesis is not None
    assert result.hypothesis.frozen is True
    assert result.hypothesis.state == "frozen_contradiction"
    assert result.hypothesis.eligible_for_review is False


def test_risa_adapter_preserves_hypothesis_as_unverified():
    ledger = BoundedCrossModalHypothesisLedger()
    _observe(ledger, "session-a")
    hypothesis = _observe(ledger, "session-b").hypothesis
    assert hypothesis is not None

    observation = observation_from_cross_modal_hypothesis(hypothesis)

    assert hypothesis.eligible_for_review is True
    assert observation.action == "hypothesize_cross_modal_relation"
    assert observation.verified is False
    assert "durable_mutation_allowed:false" in observation.context_tags


def test_cross_modal_hypothesis_ledger_enforces_state_budgets():
    ledger = BoundedCrossModalHypothesisLedger(
        max_hypotheses=1, max_observations_per_hypothesis=1
    )
    _observe(ledger, "session-a")
    observation_limit = _observe(ledger, "session-b")

    evidence = _evidence("session-c", claim="different-claim")
    decision = MultimodalStructuralVerifier().verify(
        evidence, expected_modalities=EXPECTED
    )
    hypothesis_limit = ledger.observe(
        claim_key="different-claim",
        decision=decision,
        evidence=evidence,
        expected_modalities=EXPECTED,
        observation_source_id="session-c",
        source_revision="v1",
    )

    assert observation_limit.reason == "observation_budget_exceeded"
    assert hypothesis_limit.reason == "hypothesis_budget_exceeded"
