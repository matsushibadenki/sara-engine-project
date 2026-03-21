import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.agent.sara_agent import SaraAgent
from sara_engine.inference import SaraInference
from sara_engine.utils.project_paths import model_path, workspace_path


def test_agent_soak_dialogue_keeps_bounded_state():
    agent = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )

    def calculator(_: str) -> str:
        return "5"

    agent.register_tool("<CALC>", calculator)

    for turn in range(24):
        teaching_text = f"Python の補足知識 {turn} は 可読性 を高めます。"
        agent.chat(teaching_text, teaching_mode=True)
        response = agent.chat(f"この要点を教えて <CALC> {turn}", teaching_mode=False)
        assert response

    assert len(agent.dialogue_history) <= agent.max_history_turns * 2
    assert len(agent.get_recent_issues(limit=50)) <= 20

    session_path = workspace_path("tests", "soak_agent_session.pkl")
    os.makedirs(os.path.dirname(session_path), exist_ok=True)
    agent.save_session(session_path)

    restored = SaraAgent(
        input_size=256,
        hidden_size=256,
        compartments=["general", "python_expert"],
    )
    restored.load_session(session_path)

    assert len(restored.dialogue_history) <= restored.max_history_turns * 2
    assert restored.topic_tracker.active_terms(limit=3)


def test_inference_soak_learning_and_memory_roundtrip():
    memory_path = model_path("tests", "release_soak_inference.msgpack")
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)

    engine = SaraInference.__new__(SaraInference)
    engine.model_path = memory_path
    engine.direct_map = {}
    engine.refractory_buffer = []
    engine.lif_network = None

    for offset in range(32):
        engine.learn_sequence([offset, offset + 1, offset + 2, offset + 3])

    assert engine.direct_map
    assert len(engine.direct_map) >= 16
    assert all(isinstance(key, tuple) for key in engine.direct_map.keys())

    engine.save_pretrained(memory_path)

    reloaded = SaraInference.__new__(SaraInference)
    reloaded.model_path = memory_path
    reloaded.direct_map = {}
    reloaded.refractory_buffer = []
    reloaded.lif_network = None
    reloaded._load_memory()

    assert reloaded.direct_map == engine.direct_map
