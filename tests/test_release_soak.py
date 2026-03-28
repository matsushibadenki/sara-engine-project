import importlib.util
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.agent.sara_agent import SaraAgent
from sara_engine.inference import SaraInference
from sara_engine.utils.project_paths import model_path, workspace_path


def _load_release_soak_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "release_soak.py")
    )
    spec = importlib.util.spec_from_file_location("release_soak_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_release_soak_sections_report_minimum_workload_flags():
    module = _load_release_soak_module()
    agent_report = module.run_agent_soak(duration_seconds=0.5, max_turns=12, min_turns=4)
    inference_report = module.run_inference_soak(duration_seconds=0.5, max_iterations=16, min_iterations=6)

    assert agent_report["turns"] >= 4
    assert agent_report["meets_min_turns"] is True
    assert agent_report["min_turns_required"] == 4

    assert inference_report["iterations"] >= 6
    assert inference_report["meets_min_iterations"] is True
    assert inference_report["min_iterations_required"] == 6


def test_release_soak_profile_resolution_supports_extended_shipping_profile():
    module = _load_release_soak_module()
    settings = module.resolve_soak_profile(
        profile_name="extended",
        duration_seconds=None,
        max_agent_turns=None,
        min_agent_turns=None,
        max_inference_iterations=None,
        min_inference_iterations=None,
    )

    assert settings["profile_name"] == "extended"
    assert settings["duration_seconds"] == 30.0
    assert settings["min_agent_turns"] == 60
    assert settings["min_inference_iterations"] == 96
    assert settings["shipping_ready"] is True


def test_release_soak_profile_resolution_downgrades_shipping_ready_when_thresholds_are_lowered():
    module = _load_release_soak_module()
    settings = module.resolve_soak_profile(
        profile_name="extended",
        duration_seconds=1.0,
        max_agent_turns=8,
        min_agent_turns=4,
        max_inference_iterations=12,
        min_inference_iterations=6,
    )

    assert settings["profile_name"] == "extended"
    assert settings["shipping_ready"] is False


def test_release_soak_accuracy_summary_embeds_phase3_suite(monkeypatch):
    module = _load_release_soak_module()

    monkeypatch.setattr(
        module,
        "run_phase3_accuracy_suite",
        lambda history_path, persist_history, history_limit: {
            "suite_name": "Phase3AccuracySuite",
            "overall_score": 0.92,
            "passed": True,
            "trend": {"has_previous": True, "regression_count": 0},
            "component_reports": {"agent_dialogue": {"passed": True}},
            "focus_summary": {
                "few_shot": {"score": 1.0, "passed": True},
                "continual": {"score": 1.0, "passed": True},
            },
            "history_length": 3,
        },
    )

    report = module.run_accuracy_soak(
        history_path=workspace_path("tests", "release_soak_accuracy_history.json"),
        history_limit=5,
    )

    assert report["suite_name"] == "Phase3AccuracySuite"
    assert report["passed"] is True
    assert report["history_length"] == 3
    assert report["trend"]["regression_count"] == 0
    assert report["focus_summary"]["few_shot"]["score"] == 1.0


def test_release_soak_collects_release_metadata():
    module = _load_release_soak_module()

    metadata = module.collect_release_metadata()

    assert metadata["pyproject_version"] == metadata["cargo_version"]
    assert metadata["versions_match"] is True
    assert "sara-chat" in metadata["console_scripts"]
    assert "sara-train" in metadata["console_scripts"]
    assert metadata["release_notes_heading"] == "Current Pre-Release"
    assert "Highlights" in metadata["release_note_sections"]


def test_release_soak_collects_release_gate_feedback():
    module = _load_release_soak_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "min_duration_seconds": 5.0,
            "min_agent_turns": 24,
            "min_inference_iterations": 32,
            "min_pattern_count": 1,
            "profile_name": "release",
            "require_phase3_accuracy": False,
            "shipping_ready": False,
        },
        "agent": {
            "turns": 24,
            "history_bounded": True,
            "issue_count": 0,
            "meets_min_turns": True,
        },
        "inference": {
            "iterations": 32,
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 12,
            "meets_min_iterations": True,
        },
        "release_metadata": {
            "versions_match": True,
            "has_expected_console_scripts": True,
            "release_notes_heading": "Current Pre-Release",
        },
    }

    feedback = module.collect_release_gate_feedback(report)

    assert feedback["passed"] is True
    assert feedback["error_count"] == 0
    assert feedback["errors"] == []


def test_release_soak_formats_human_readable_summary():
    module = _load_release_soak_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "profile_name": "release",
            "shipping_ready": False,
            "require_phase3_accuracy": True,
        },
        "agent": {
            "turns": 24,
            "min_turns_required": 24,
            "history_bounded": True,
            "issue_count": 0,
            "meets_min_turns": True,
        },
        "inference": {
            "iterations": 32,
            "min_iterations_required": 32,
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 12,
            "meets_min_iterations": True,
        },
        "release_metadata": {
            "pyproject_version": "0.4.6",
            "versions_match": True,
            "has_expected_console_scripts": True,
            "console_scripts": ["sara-chat", "sara-train"],
            "release_notes_heading": "Current Pre-Release",
        },
        "release_gate": {
            "passed": True,
            "error_count": 0,
            "errors": [],
        },
        "accuracy": {
            "suite_name": "Phase3AccuracySuite",
            "passed": True,
            "overall_score": 0.91,
            "trend": {"regression_count": 0},
            "focus_summary": {
                "few_shot": {"score": 1.0, "passed": True},
                "continual": {"score": 1.0, "passed": True},
            },
        },
    }

    summary = module.format_release_summary(report)

    assert "SARA Engine Release Soak Summary" in summary
    assert "overall_status: PASS" in summary
    assert "profile: release" in summary
    assert "- status: PASS" in summary
    assert "- turns: 24 / min 24" in summary
    assert "- iterations: 32 / min 32" in summary
    assert "- version: 0.4.6" in summary
    assert "- suite_name: Phase3AccuracySuite" in summary
    assert "Phase 3 Focus" in summary
    assert "- few_shot_status: PASS" in summary
    assert "- continual_status: PASS" in summary
    assert "Gate" in summary
    assert "- error_count: 0" in summary


def test_release_soak_summary_warns_when_required_accuracy_is_missing():
    module = _load_release_soak_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "profile_name": "release",
            "shipping_ready": False,
            "require_phase3_accuracy": True,
        },
        "agent": {
            "turns": 24,
            "min_turns_required": 24,
            "history_bounded": True,
            "issue_count": 0,
            "meets_min_turns": True,
        },
        "inference": {
            "iterations": 32,
            "min_iterations_required": 32,
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 12,
            "meets_min_iterations": True,
        },
        "release_metadata": {
            "pyproject_version": "0.4.6",
            "versions_match": True,
            "has_expected_console_scripts": True,
            "console_scripts": ["sara-chat", "sara-train"],
            "release_notes_heading": "Current Pre-Release",
        },
        "release_gate": {
            "passed": False,
            "error_count": 1,
            "errors": ["Release soak report requires embedded Phase 3 accuracy results."],
        },
    }

    summary = module.format_release_summary(report)

    assert "overall_status: WARN" in summary
    assert "Accuracy" in summary
    assert "- status: WARN" in summary
    assert "- suite_name: missing" in summary
    assert "Phase 3 Focus" in summary
    assert "- few_shot_status: WARN" in summary
    assert "Gate" in summary
    assert "- error_count: 1" in summary
    assert "embedded Phase 3 accuracy" in summary
