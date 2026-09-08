import json
from dataclasses import replace
from pathlib import Path

import pytest

from sara_engine.evaluation.verified_retrieval_compatibility import (
    build_components, build_inputs, encode, evaluate, select_answer,
)
from sara_engine.memory.verified_answer import resolve_verified_answer


FIXTURE = Path(__file__).resolve().parents[1] / "data/processed/benchmark_fixtures/verified_retrieval_compatibility.json"
PROTOCOL = json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_development_compatibility_passes_without_chat_promotion():
    report = evaluate(PROTOCOL)
    assert len(report["cases"]) == 24
    assert report["component_contract_passed"] is True
    assert report["normal_chat_promotion"] is False
    assert report["metrics"]["verified"]["required_abstention_rate"] == 1.0
    assert report == evaluate(PROTOCOL)


@pytest.mark.parametrize("language", PROTOCOL["languages"])
@pytest.mark.parametrize("scenario", ["conflicting_sources", "conflicting_sources_reversed"])
def test_conflicting_sources_never_choose_an_answer_by_insertion_order(language, scenario):
    records, question, _ = build_inputs(language, scenario)
    cache, answers, _, _ = build_components(records)
    assert len(cache.entries) == 2
    result = select_answer(cache, answers, question, language["language"])
    assert result["text"] == ""


def test_revision_binds_current_answer_to_stable_entry_id():
    records, question, _ = build_inputs(PROTOCOL["languages"][0], "revision")
    cache, answers, _, admissions = build_components(records)
    assert admissions[-1] == "replace_verified_revision"
    result = select_answer(cache, answers, question, "en")
    assert result["decision"] == "answer"
    assert result["text"] == PROTOCOL["languages"][0]["current"]
    assert result["source_revision"] == "r2"


def test_query_encoding_is_bounded_and_deterministic():
    assert encode("Door state?") == encode("DOOR STATE")
    with pytest.raises(ValueError, match="limit"):
        encode("x" * 257)


def test_evaluator_expectations_do_not_enter_components():
    records, question, expected = build_inputs(PROTOCOL["languages"][0], "single")
    cache, answers, _, _ = build_components(records)
    before = select_answer(cache, answers, question, "en")
    expected["text"] = "Evaluator-only replacement"
    assert select_answer(cache, answers, question, "en") == before


@pytest.mark.parametrize("language", PROTOCOL["languages"])
def test_old_revision_cannot_overwrite_current_answer(language):
    records, question, _ = build_inputs(language, "revision_reversed")
    cache, answers, _, _ = build_components(records)
    result = select_answer(cache, answers, question, language["language"])
    assert result["text"] == language["current"]
    assert result["source_revision"] == "r2"


def test_stable_id_binding_requires_new_valid_receipt():
    records, _, _ = build_inputs(PROTOCOL["languages"][0], "revision")
    cache, answers, _, _ = build_components(records)
    entry_id = next(iter(cache.entries))
    answer = answers[entry_id]
    forged = replace(answer, text="Tampered revised answer")
    assert resolve_verified_answer(
        cache.entries[entry_id], forged, now_segment=3, language="en"
    ).decision == "answer_digest_mismatch"


def test_containment_rejects_partial_topic_and_invalid_mode():
    records, _, _ = build_inputs(PROTOCOL["languages"][0], "single")
    cache, answers, _, _ = build_components(records)
    assert select_answer(cache, answers, "door", "en")["text"] == ""
    with pytest.raises(ValueError, match="match mode"):
        cache.retrieve((1,), signature_match="unknown")


@pytest.mark.parametrize("question", [
    "Please report the north door state now",
    "今、北側の扉の状態を確認してください",
    "请报告北门状态",
])
def test_complete_topic_survives_question_framing(question):
    index = 0 if question.startswith("Please") else (1 if question.startswith("今") else 2)
    language = PROTOCOL["languages"][index]
    records, _, _ = build_inputs(language, "single")
    cache, answers, _, _ = build_components(records)
    assert select_answer(cache, answers, question, language["language"])["text"] == language["current"]
