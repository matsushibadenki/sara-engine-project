import importlib.util
import json
from pathlib import Path

import pytest

from sara_engine.agent.sara_agent import SaraAgent
from sara_engine.memory.topic_evidence_store import TopicEvidenceStore

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("agent_evidence_fixture", ROOT / "scripts/eval/structured_query_contract.py")
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)
PROTOCOL = json.loads((ROOT / "data/processed/benchmark_fixtures/structured_query_contract_v1.json").read_text())


@pytest.mark.parametrize("language", PROTOCOL["languages"])
def test_agent_explicit_evidence_path_preserves_results_without_chat(language):
    # Isolate the actual method from constructor model loading/training.
    agent = object.__new__(SaraAgent)
    agent.chat = lambda *args, **kwargs: pytest.fail("Evidence mode must not generate")
    agent.last_response_trace = {"kind": "sentinel"}
    store = TopicEvidenceStore()
    store.publish(FIXTURE.build_evidence(language, PROTOCOL["topics"]), expected_generation=0, now_segment=3)
    questions = json.loads((ROOT / "data/processed/benchmark_fixtures/topic_question_v1.json").read_text())
    row = next(row for row in questions["languages"] if row["language"] == language["language"])
    aliases = tuple(map(tuple, row["aliases"]))
    for text, requested, excluded in row["cases"]:
        result = agent.answer_verified_question(text, evidence_store=store, language=language["language"], aliases=aliases, now_segment=3)
        if requested is None:
            assert result.result.decision == "unsupported_question" and not result.result.items
        else:
            assert result.result.decision == "answer"
            assert tuple(item.topic_id for item in result.result.items) == tuple(requested)
            for item in result.result.items:
                index = PROTOCOL["topics"].index(item.topic_id)
                assert item.evidence[0].text == language["texts"][index]
                assert item.evidence[0].source_ref == f"fixture:sensor:{index}"
                assert item.evidence[0].source_revision == "r1"
        assert result.generation == 1
    store.refresh_failed(expected_generation=1)
    failed = agent.answer_verified_question(row["cases"][0][0], evidence_store=store, language=language["language"], aliases=aliases, now_segment=3)
    assert failed.result.decision == "evidence_unavailable" and not failed.result.items
    assert agent.last_response_trace == {"kind": "sentinel"}


def test_explicit_evidence_requires_store():
    agent = object.__new__(SaraAgent)
    with pytest.raises(TypeError):
        agent.answer_verified_question("Report a.", evidence_store=None, language="en", aliases=(), now_segment=3)


def test_initialized_agent_preserves_dialogue_across_verified_answers():
    import copy
    agent = SaraAgent(input_size=256, hidden_size=256, compartments=["general"])
    calls = []
    agent.register_tool("<PING>", lambda _: calls.append(1) or "pong")
    assert "pong" in agent.chat("Run <PING>")
    history = copy.deepcopy(agent.dialogue_history)
    state = copy.deepcopy(agent.dialogue_state)
    trace = agent.get_last_response_trace()
    store = TopicEvidenceStore()
    language = PROTOCOL["languages"][0]
    store.publish(FIXTURE.build_evidence(language, PROTOCOL["topics"]), expected_generation=0, now_segment=3)
    result = agent.answer_verified_question("Report tank temperature.", evidence_store=store,
        language="en", aliases=(("tank temperature", "tank:temperature"),), now_segment=3)
    assert result.result.decision == "answer"
    assert result.result.items[0].evidence[0].text == language["texts"][1]
    assert agent.dialogue_history == history
    assert agent.dialogue_state == state
    assert agent.get_last_response_trace() == trace
    assert len(calls) == 1
    assert "pong" in agent.chat("Run <PING>")
    assert len(calls) == 2


@pytest.mark.parametrize("language", PROTOCOL["languages"])
def test_chat_opt_in_has_no_fallback_or_history_mutation(language):
    import copy
    agent = SaraAgent(input_size=256, hidden_size=256, compartments=["general"])
    store = TopicEvidenceStore()
    store.publish(FIXTURE.build_evidence(language, PROTOCOL["topics"]), expected_generation=0, now_segment=3)
    questions = json.loads((ROOT / "data/processed/benchmark_fixtures/topic_question_v1.json").read_text())
    row = next(row for row in questions["languages"] if row["language"] == language["language"])
    kwargs = dict(evidence_store=store, evidence_language=language["language"],
                  evidence_aliases=tuple(map(tuple, row["aliases"])), evidence_now_segment=3)
    before = copy.deepcopy(agent.dialogue_history)
    assert agent.chat(row["cases"][0][0], **kwargs) == language["texts"][0]
    trace = agent.get_last_response_trace()
    assert trace["kind"] == "verified_answer" and trace["evidence_generation"] == 1
    assert trace["owners"] == ["verified_topic_store"] and not trace["generated_continuation"]
    assert trace["source_refs"] == ["fixture:sensor:0"]
    agent.register_tool("<DO>", lambda _: pytest.fail("Evidence mode invoked a tool"))
    refusal = agent.chat("<DO>", **kwargs)
    assert refusal and agent.get_last_response_trace()["kind"] == "verified_abstention"
    store.refresh_failed(expected_generation=1)
    assert agent.chat(row["cases"][0][0], **kwargs) == refusal
    assert agent.get_last_response_trace()["status"] == "evidence_unavailable"
    assert agent.dialogue_history == before


@pytest.mark.parametrize("option", ["stream", "teaching_mode"])
def test_chat_evidence_mode_rejects_mutating_or_streaming_modes(option):
    agent = object.__new__(SaraAgent)
    with pytest.raises(ValueError):
        agent.chat("Report a.", evidence_store=TopicEvidenceStore(), **{option: True})


def test_chat_evidence_mode_retains_input_guard():
    from types import SimpleNamespace
    agent = SaraAgent(input_size=256, hidden_size=256, compartments=["general"])
    agent.safety_guard = SimpleNamespace(check_input=lambda _: SimpleNamespace(is_safe=False))
    assert "rejected" in agent.chat("Report a.", evidence_store=TopicEvidenceStore())
    assert agent.get_last_response_trace()["owners"] == ["safety_guard"]
    agent.safety_guard = SimpleNamespace(check_input=lambda _: SimpleNamespace(is_safe=True, sanitized_text="changed"))
    assert "not executed" in agent.chat("Report a.", evidence_store=TopicEvidenceStore())
    assert agent.get_last_response_trace()["status"] == "modified_evidence_query"
    agent.safety_guard = None
    assert agent.chat("Report a.", evidence_store=TopicEvidenceStore(), evidence_language=[])
    assert agent.get_last_response_trace()["kind"] == "verified_abstention"
