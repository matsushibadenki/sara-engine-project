import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from sara_engine.memory.structured_query import TopicQuery
from sara_engine.memory.topic_evidence_store import TopicEvidenceStore
from sara_engine.memory.verification_receipt import issue_verification_receipt


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("store_fixtures", ROOT / "scripts/eval/structured_query_contract.py")
FIXTURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURES)
PROTOCOL = json.loads((ROOT / "data/processed/benchmark_fixtures/structured_query_contract_v1.json").read_text())


@pytest.fixture
def records():
    return FIXTURES.build_evidence(PROTOCOL["languages"][0], PROTOCOL["topics"])


def revised(item, *, revision="r2", source=None, text="Updated state"):
    entry = replace(item.entry, source_revision=revision, source_ref=source or item.entry.source_ref)
    answer = replace(item.answer, source_revision=revision, source_ref=entry.source_ref, text=text)
    receipt = issue_verification_receipt(
        verifier_id="fixture-answer-verifier", verifier_version="v1", decision="verified_answer_binding",
        evidence=answer.evidence_payload(), source_refs=(entry.source_ref,), source_revision=revision,
        observed=True, source_backed=True, verified=True,
    )
    return FIXTURES.bind_fixture_topic(item.topic_id, entry, replace(answer, receipt=receipt))


def test_atomic_revision_replacement_and_stale_writer(records):
    store = TopicEvidenceStore()
    query = TopicQuery((records[0].topic_id,))
    assert store.answer(query, now_segment=3).result.decision == "evidence_unavailable"
    assert store.publish(records, expected_generation=0, now_segment=3).decision == "published"
    new = revised(records[0])
    assert store.publish((new, records[1]), expected_generation=1, now_segment=3).generation == 2
    assert store.publish(records, expected_generation=1, now_segment=3).decision == "stale_generation"
    answer = store.answer(query, now_segment=3)
    assert answer.generation == 2
    assert answer.result.items[0].evidence[0].source_revision == "r2"
    assert answer.result.items[0].evidence[0].text == "Updated state"


@pytest.mark.parametrize("failure", ["overflow", "bad_binding", "duplicate_source"])
def test_failed_refresh_invalidates_old_answers_and_can_recover(records, failure):
    store = TopicEvidenceStore()
    store.publish(records, expected_generation=0, now_segment=3)
    if failure == "overflow":
        invalid = (None,) * 13
    elif failure == "bad_binding":
        invalid = (replace(records[0], topic_id="relabelled"),)
    else:
        invalid = (records[0], revised(records[0]))
    assert store.publish(invalid, expected_generation=1, now_segment=3).decision != "published"
    result = store.answer(TopicQuery((records[0].topic_id,)), now_segment=3)
    assert result.generation == 2 and not result.result.items
    assert result.result.decision == "evidence_unavailable"
    assert store.publish(records, expected_generation=2, now_segment=3).decision == "published"


def test_conflicting_sources_are_preserved_not_ranked_away(records):
    store = TopicEvidenceStore()
    conflict = revised(records[0], source="fixture:independent")
    assert store.publish((*records, conflict), expected_generation=0, now_segment=3).decision == "published"
    assert store.answer(TopicQuery((records[0].topic_id,)), now_segment=3).result.decision == "conflicting_evidence"


def test_expiry_is_checked_at_query_time(records):
    store = TopicEvidenceStore()
    expiring = replace(records[0], entry=replace(records[0].entry, expires_at=4))
    store.publish((expiring,), expected_generation=0, now_segment=3)
    answer = store.answer(TopicQuery((expiring.topic_id,)), now_segment=4)
    assert answer.result.decision == "expired_evidence" and not answer.result.items


@pytest.mark.parametrize("row", PROTOCOL["languages"])
def test_real_store_question_path_in_three_languages(row):
    questions = json.loads((ROOT / "data/processed/benchmark_fixtures/topic_question_v1.json").read_text())
    catalog = next(item for item in questions["languages"] if item["language"] == row["language"])
    store = TopicEvidenceStore()
    records = FIXTURES.build_evidence(row, PROTOCOL["topics"])
    store.publish(records, expected_generation=0, now_segment=3)
    for text, requested, excluded in catalog["cases"]:
        answer = store.answer_question(text, language=row["language"], aliases=tuple(map(tuple, catalog["aliases"])), now_segment=3)
        if requested is None:
            assert answer.result.decision == "unsupported_question" and not answer.result.items
        else:
            assert answer.result.decision == "answer"
            assert tuple(item.topic_id for item in answer.result.items) == tuple(requested)
            for item in answer.result.items:
                assert item.evidence[0].text == row["texts"][PROTOCOL["topics"].index(item.topic_id)]


def test_full_scope_withdrawal_and_empty_snapshot(records):
    store = TopicEvidenceStore()
    store.publish(records, expected_generation=0, now_segment=3)
    assert store.publish((), expected_generation=1, now_segment=3).decision == "published"
    assert store.answer(TopicQuery((records[0].topic_id,)), now_segment=3).result.decision == "incomplete_coverage"


def test_snapshot_deadline_even_when_records_have_no_expiry(records):
    store = TopicEvidenceStore(max_age_segments=2)
    store.publish(records, expected_generation=0, now_segment=3)
    query = TopicQuery((records[0].topic_id,))
    assert store.answer(query, now_segment=4).result.decision == "answer"
    assert store.answer(query, now_segment=5).result.decision == "snapshot_expired"
    assert store.answer(query, now_segment=4).result.decision == "time_regression"
    assert store.answer(query, now_segment=6).result.decision == "evidence_unavailable"
    assert store.publish(records, expected_generation=1, now_segment=6).decision == "published"
    assert store.answer(query, now_segment=6).result.decision == "answer"


def test_refresh_failure_revokes_only_matching_generation(records):
    store = TopicEvidenceStore()
    store.publish(records, expected_generation=0, now_segment=3)
    assert store.refresh_failed(expected_generation=0).decision == "stale_generation"
    query = TopicQuery((records[0].topic_id,))
    assert store.answer(query, now_segment=3).result.decision == "answer"
    assert store.refresh_failed(expected_generation=1).generation == 2
    assert store.answer(query, now_segment=3).result.decision == "evidence_unavailable"
    assert store.publish(records, expected_generation=1, now_segment=3).decision == "stale_generation"


def test_regressive_publication_cannot_extend_freshness(records):
    store = TopicEvidenceStore()
    store.publish(records, expected_generation=0, now_segment=3)
    query = TopicQuery((records[0].topic_id,))
    store.answer(query, now_segment=4)
    assert store.publish(records, expected_generation=1, now_segment=3).decision == "time_regression"
    assert store.answer(query, now_segment=4).result.decision == "evidence_unavailable"


@pytest.mark.parametrize("age", [0, -1, True, 65537, 1.5])
def test_snapshot_age_validation(age):
    with pytest.raises(ValueError):
        TopicEvidenceStore(max_age_segments=age)
