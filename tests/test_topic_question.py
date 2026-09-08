import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import pytest

from sara_engine.memory.topic_question import parse_topic_question
from sara_engine.memory.structured_query import resolve_complete_topic_query


ROOT = Path(__file__).resolve().parents[1]
RAW = (ROOT / "data/processed/benchmark_fixtures/topic_question_v1.json").read_bytes()
PROTOCOL = json.loads(RAW)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/eval/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_acceptance():
    assert sha256(RAW).hexdigest() == "7c0424b0d11e933c7b3fac5c6d00ad436a177991a6e4e52c38cbde3039c6ab94"
    report = load_script("topic_question").evaluate(PROTOCOL)
    assert report["correct"] == report["case_count"] == 24
    assert not report["normal_chat_promotion"]


@pytest.mark.parametrize("row", PROTOCOL["languages"])
def test_parse_to_bounded_verified_resolution(row):
    fixture = json.loads((ROOT / "data/processed/benchmark_fixtures/structured_query_contract_v1.json").read_text())
    language = next(value for value in fixture["languages"] if value["language"] == row["language"])
    evidence = load_script("structured_query_contract").build_evidence(language, fixture["topics"])
    aliases = tuple(tuple(pair) for pair in row["aliases"])
    for text, requested, excluded in row["cases"]:
        parsed = parse_topic_question(text, language=row["language"], aliases=aliases)
        if requested is None:
            assert parsed.query is None
            continue
        result = resolve_complete_topic_query(parsed.query, iter(evidence), now_segment=3)
        assert result.decision == "answer"
        assert tuple(item.topic_id for item in result.items) == tuple(requested)
        for item in result.items:
            index = fixture["topics"].index(item.topic_id)
            assert item.evidence[0].text == language["texts"][index]
            assert item.evidence[0].source_ref == f"fixture:sensor:{index}"
        missing = resolve_complete_topic_query(parsed.query, iter(()), now_segment=3)
        assert missing.decision == "incomplete_coverage" and not missing.items


@pytest.mark.parametrize("text", [
    "Report a and b.", "Exclude a; report a.", "Report a and a.",
    "Report a or b.", "Report not a.", "Report a. Report b.",
    "Report a\nand b.", "Report a and c and d and e and f.",
])
def test_unsupported_scopes_and_duplicate_alias_targets_return_no_query(text):
    aliases = (("a", "same"), ("b", "same"), ("c", "c"), ("d", "d"), ("e", "e"), ("f", "f"))
    result = parse_topic_question(text, language="en", aliases=aliases)
    assert result.decision == "unsupported_question"
    assert result.query is None


@pytest.mark.parametrize("aliases", [
    (), (("a", "x"),) * 33, (("a", "x"), ("A", "y")),
    (("a and b", "x"),), (("a; report b", "x"),),
    (("a" * 129, "x"),), (("a", "x" * 129),), ((" a", "x"),),
    ((None, "x"),), (["a", "x"],),
])
def test_invalid_catalog_abstains(aliases):
    result = parse_topic_question("Report a.", language="en", aliases=aliases)
    assert result.decision == "invalid_catalog" and result.query is None


@pytest.mark.parametrize("text,language", [(None, "en"), ("x" * 257, "en"), ("", "en"), ("Report a.", "fr"), ("Report a.", [])])
def test_input_limits(text, language):
    result = parse_topic_question(text, language=language, aliases=(("a", "x"),))
    assert result.decision == "invalid_input" and result.query is None


def test_declared_aliases_are_generic_and_only_explicit_normalization_applies():
    result = parse_topic_question("  REPORT BATTERY LEVEL.  ", language="en", aliases=(("battery level", "device:charge"),))
    assert result.query.requested == ("device:charge",)
    assert parse_topic_question("Report battery  level.", language="en", aliases=(("battery level", "device:charge"),)).query is None
