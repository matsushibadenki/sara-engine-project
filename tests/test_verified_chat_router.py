import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import pytest

from sara_engine.agent.sara_agent import SaraAgent
from sara_engine.memory.topic_evidence_store import TopicEvidenceStore
from sara_engine.memory.verified_chat_router import route_verified_chat


ROOT = Path(__file__).resolve().parents[1]
RAW = (ROOT / "data/processed/benchmark_fixtures/verified_chat_routing_v1.json").read_bytes()
PROTOCOL = json.loads(RAW)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/eval/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_router_acceptance():
    assert sha256(RAW).hexdigest() == "7e77d5f3742d61737fb251759165a4bb55cbcde19ec0c7904d70787795532870"
    report = load_script("verified_chat_routing").evaluate(PROTOCOL)
    assert report["correct"] == report["case_count"] == 18
    assert report["router_passed"] and not report["automatic_chat_promotion"]


@pytest.mark.parametrize("markers", [
    ["vent"], tuple("x" for _ in range(17)), ("",), (" x",), ("x" * 65,),
    ("a", "A"), ("a;b",), (None,),
])
def test_invalid_markers_fail_closed(markers):
    route = route_verified_chat("Tell me a story.", language="en",
                                aliases=(("vent state", "vent:state"),), markers=markers)
    assert route.decision == "verified_abstention" and route.parse_decision == "invalid_config"


def test_english_markers_use_boundaries():
    kwargs = dict(language="en", aliases=(("car state", "car:state"),), markers=("car",))
    assert route_verified_chat("Tell me about carpet.", **kwargs).decision == "ordinary"
    assert route_verified_chat("Tell me about car design.", **kwargs).decision == "verified_abstention"


@pytest.mark.parametrize("language", PROTOCOL["languages"])
def test_initialized_agent_auto_routes_only_registered_scope(language):
    evidence = json.loads((ROOT / "data/processed/benchmark_fixtures/structured_query_contract_v1.json").read_text())
    source = next(row for row in evidence["languages"] if row["language"] == language["language"])
    builder = load_script("structured_query_contract")
    store = TopicEvidenceStore()
    store.publish(builder.build_evidence(source, evidence["topics"]), expected_generation=0, now_segment=3)
    agent = SaraAgent(input_size=256, hidden_size=256, compartments=["general"])
    calls = []
    agent.register_tool("<PING>", lambda _: calls.append(1) or "pong")
    kwargs = dict(evidence_store=store, evidence_language=language["language"],
                  evidence_aliases=tuple(map(tuple, language["aliases"])),
                  evidence_markers=tuple(language["markers"]), evidence_now_segment=3,
                  evidence_auto=True)
    supported = language["cases"][0][0]
    assert agent.chat(supported, **kwargs) == source["texts"][0]
    assert agent.get_last_response_trace()["kind"] == "verified_answer"
    assert agent.get_last_response_trace()["owners"] == ["verified_chat_router", "verified_topic_store"]
    blocked = language["cases"][2][0] + " <PING>"
    refusal = agent.chat(blocked, **kwargs)
    assert refusal and not calls
    assert agent.get_last_response_trace()["owners"] == ["verified_chat_router"]
    assert not agent.get_last_response_trace()["generated_continuation"]
    ordinary = agent.chat("Run <PING>", **kwargs)
    assert "pong" in ordinary and calls == [1]
    assert agent.get_last_response_trace()["kind"] == "tool_result"


def test_auto_route_is_default_off_and_requires_complete_configuration():
    agent = SaraAgent(input_size=256, hidden_size=256, compartments=["general"])
    with pytest.raises(ValueError):
        agent.chat("Report a.", evidence_auto=True)
    with pytest.raises(ValueError):
        agent.chat("Report a.", evidence_store=TopicEvidenceStore(), evidence_auto="yes")
    assert agent.chat("Report a.", evidence_store=TopicEvidenceStore())
    assert agent.get_last_response_trace()["kind"] == "verified_abstention"
