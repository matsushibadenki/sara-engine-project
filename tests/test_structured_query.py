import copy
import importlib.util
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from sara_engine.memory.structured_query import TopicQuery, resolve_topic_query
from sara_engine.memory.verification_receipt import issue_verification_receipt


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("structured_query_eval", ROOT / "scripts/eval/structured_query_contract.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RAW = (ROOT / "data/processed/benchmark_fixtures/structured_query_contract_v1.json").read_bytes()
PROTOCOL = json.loads(RAW)


@pytest.fixture
def evidence():
    return MODULE.build_evidence(PROTOCOL["languages"][0], PROTOCOL["topics"])


def test_frozen_multilingual_acceptance():
    assert sha256(RAW).hexdigest() == "ae8dad11371aa4e3078f19c83032657832fb848a89e4a89aab41524bb7fac554"
    report = MODULE.evaluate(PROTOCOL)
    assert report["case_count"] == report["correct"] == 36
    assert report["contract_passed"]
    assert not report["normal_chat_promotion"]


@pytest.mark.parametrize("mutation,decision", [
    ("topic", "invalid_topic_binding"),
    ("text", "answer_digest_mismatch"),
    ("expired", "expired_evidence"),
    ("future", "future_evidence"),
    ("revision", "entry_binding_mismatch"),
    ("receipt", "receipt_invalid"),
    ("topic_receipt", "invalid_topic_binding"),
])
def test_invalid_evidence_never_returns_partial_answers(evidence, mutation, decision):
    first, second = evidence
    if mutation == "topic":
        second = replace(second, topic_id="other")
    elif mutation == "text":
        second = replace(second, answer=replace(second.answer, text="Tampered answer"))
    elif mutation == "expired":
        second = replace(second, entry=replace(second.entry, expires_at=3))
    elif mutation == "future":
        second = replace(second, entry=replace(second.entry, time_segment=4))
    elif mutation == "revision":
        second = replace(second, entry=replace(second.entry, source_revision="r2"))
    elif mutation == "receipt":
        second = replace(second, answer=replace(second.answer, receipt=replace(second.answer.receipt, source_refs=None)))
    else:
        second = replace(second, receipt=replace(second.receipt, source_refs=None))
    result = resolve_topic_query(TopicQuery((first.topic_id, second.topic_id)), (first, second), now_segment=3)
    assert result.decision == decision
    assert result.items == ()


def alternate_source(item, *, text):
    entry = replace(item.entry, entry_id="alternative", source_ref="fixture:alternative")
    answer = replace(item.answer, entry_id=entry.entry_id, source_ref=entry.source_ref, text=text)
    receipt = issue_verification_receipt(
        verifier_id="fixture-answer-verifier", verifier_version="v1",
        decision="verified_answer_binding", evidence=answer.evidence_payload(),
        source_refs=(entry.source_ref,), source_revision=entry.source_revision,
        observed=True, source_backed=True, verified=True,
    )
    return MODULE.bind_fixture_topic(item.topic_id, entry, replace(answer, receipt=receipt))


def test_conflicting_sources_abstain_in_both_orders(evidence):
    first = evidence[0]
    conflicting = alternate_source(first, text="The west vent is open.")
    for snapshot in ((first, conflicting), (conflicting, first)):
        result = resolve_topic_query(TopicQuery((first.topic_id,)), snapshot, now_segment=3)
        assert result.decision == "conflicting_evidence"
        assert not result.items


def test_agreeing_sources_preserve_provenance_and_ignore_order(evidence):
    first = evidence[0]
    agreeing = alternate_source(first, text=first.answer.text)
    query = TopicQuery((first.topic_id,))
    forward = resolve_topic_query(query, (first, agreeing), now_segment=3)
    reverse = resolve_topic_query(query, (agreeing, first), now_segment=3)
    assert forward == reverse
    assert forward.decision == "answer"
    assert {answer.source_ref for answer in forward.items[0].evidence} == {first.answer.source_ref, "fixture:alternative"}


def test_read_only_and_exclusion_does_not_require_excluded_answer(evidence):
    before = copy.deepcopy(evidence)
    first, second = evidence
    invalid_excluded = replace(first, answer=None)
    result = resolve_topic_query(TopicQuery((second.topic_id,), (first.topic_id,)), (invalid_excluded, second), now_segment=3)
    assert result.decision == "answer"
    assert tuple(item.topic_id for item in result.items) == (second.topic_id,)
    assert evidence == before


@pytest.mark.parametrize("query", [
    TopicQuery(("x" * 129,)), TopicQuery((" x",)), TopicQuery((None,)),
    TopicQuery(["x"]), TopicQuery(("x",), excluded=("a", "a")),
    TopicQuery(("x",), language=[]), TopicQuery(("x",), unresolved="false"),
])
def test_invalid_query_is_bounded(query, evidence):
    assert resolve_topic_query(query, evidence, now_segment=3).decision == "invalid_query"


def test_evidence_budget_rejected_before_inspecting_records(evidence):
    query = TopicQuery((evidence[0].topic_id,))
    assert resolve_topic_query(query, (None,) * 13, now_segment=3).decision == "evidence_limit"
    assert resolve_topic_query(query, (evidence[0],) * 12, now_segment=3).decision == "answer"


def test_language_and_time_fail_closed(evidence):
    query = TopicQuery((evidence[0].topic_id,), language="ja")
    result = resolve_topic_query(query, evidence, now_segment=3)
    assert result.decision == "language_mismatch"
    assert not result.items
    assert resolve_topic_query(query, evidence, now_segment=True).decision == "invalid_time"
