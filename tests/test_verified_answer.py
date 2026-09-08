import json
from dataclasses import replace
from pathlib import Path

import pytest

from sara_engine.memory.event_state_cache import EventStateCandidate, VerifiedHierarchicalEventStateCache
from sara_engine.memory.verified_answer import VerifiedAnswer, resolve_verified_answer
from sara_engine.memory.verification_receipt import issue_verification_receipt


FIXTURE = Path(__file__).resolve().parents[1] / "data/processed/benchmark_fixtures/verified_answer_compatibility.json"
PROTOCOL = json.loads(FIXTURE.read_text(encoding="utf-8"))


def binding(case):
    candidate = EventStateCandidate.from_verified_evidence(
        verifier_id="fixture-state-verifier", evidence=case,
        entry_id=case["id"], signature=(1, 3, 5), source_ref=case["source_ref"],
        source_revision=case["source_revision"], time_segment=1, expires_at=5,
        observed=True, source_backed=True, verified=True, resonance_score=0.95,
    )
    cache = VerifiedHierarchicalEventStateCache()
    assert cache.admit(candidate).accepted
    payload = {
        "schema": "sara-verified-answer-binding-v1", "entry_id": case["id"],
        "text": case["text"], "language": case["language"],
        "source_ref": case["source_ref"], "source_revision": case["source_revision"],
    }
    receipt = issue_verification_receipt(
        verifier_id="fixture-answer-verifier", verifier_version="v1",
        decision="verified_answer_binding", evidence=payload,
        source_refs=(case["source_ref"],), source_revision=case["source_revision"],
        observed=True, source_backed=True, verified=True,
    )
    return cache.entries[case["id"]], VerifiedAnswer(
        case["id"], case["text"], case["language"],
        case["source_ref"], case["source_revision"], receipt,
    )


@pytest.mark.parametrize("case", PROTOCOL["cases"], ids=lambda case: case["id"])
@pytest.mark.parametrize("control", PROTOCOL["required_controls"])
def test_frozen_answer_compatibility(case, control):
    entry, answer = binding(case)
    before = entry.to_dict()
    now = 2
    language = case["language"]
    selected = entry
    if control == "missing_evidence":
        answer = None
    elif control == "tampered_text":
        answer = replace(answer, text=answer.text + " changed")
    elif control == "wrong_source":
        answer = replace(answer, source_ref="fixture:unrelated")
    elif control == "stale_revision":
        answer = replace(answer, source_revision="r0")
    elif control == "expired":
        now = 5
    elif control == "future":
        now = 0
    elif control == "unverified_entry":
        selected = replace(entry, verified=False)
    elif control == "contradicted_receipt":
        receipt = issue_verification_receipt(
            verifier_id="fixture-answer-verifier", verifier_version="v1",
            decision="verified_answer_binding", evidence=answer.evidence_payload(),
            source_refs=(answer.source_ref,), source_revision=answer.source_revision,
            observed=True, source_backed=True, verified=True, contradicted=True,
        )
        answer = replace(answer, receipt=receipt)
    elif control == "language_mismatch":
        language = "ja" if language != "ja" else "en"
    elif control == "oversized_answer":
        answer = replace(answer, text="x" * 4097)
    result = resolve_verified_answer(selected, answer, now_segment=now, language=language)
    if control == "valid_binding":
        assert result.decision == "answer"
        assert result.text == case["text"]
        assert result.source_ref == case["source_ref"]
        assert result.source_revision == case["source_revision"]
    else:
        assert result.decision != "answer"
        assert result.text == result.source_ref == result.source_revision == ""
    assert entry.to_dict() == before
