import importlib.util
import os


def _load_release_gate_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "release_gate.py")
    )
    spec = importlib.util.spec_from_file_location("release_gate_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_gate_accepts_healthy_report():
    module = _load_release_gate_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "min_duration_seconds": 5.0,
            "min_agent_turns": 8,
            "min_inference_iterations": 12,
            "min_pattern_count": 1,
        },
        "agent": {"history_bounded": True, "issue_count": 0, "turns": 8},
        "inference": {
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 10,
            "iterations": 12,
        },
    }

    assert module.validate_release_report(report) == []

    accuracy_report = {
        "suite_name": "Phase3AccuracySuite",
        "overall_score": 0.95,
        "passed": True,
        "component_reports": {
            "agent_dialogue": {"passed": True, "overall_score": 0.9},
            "sara_inference": {"passed": True, "overall_score": 1.0},
            "spiking_llm": {"passed": True, "overall_score": 0.95},
        },
        "focus_summary": {
            "few_shot": {"passed": True, "score": 1.0},
            "continual": {"passed": True, "score": 1.0},
        },
        "trend": {
            "has_previous": True,
            "regression_count": 0,
        },
    }

    assert module.validate_phase3_accuracy_report(accuracy_report) == []


def test_release_gate_rejects_unhealthy_report():
    module = _load_release_gate_module()
    report = {
        "duration_seconds": 1.0,
        "criteria": {
            "min_duration_seconds": 5.0,
            "min_agent_turns": 8,
            "min_inference_iterations": 12,
            "min_pattern_count": 2,
        },
        "agent": {"history_bounded": False, "issue_count": 2, "turns": 4},
        "inference": {
            "roundtrip_ok": False,
            "tuple_keys_only": False,
            "pattern_count": 0,
            "iterations": 3,
        },
    }

    errors = module.validate_release_report(report)

    assert errors
    assert any("history" in item.lower() for item in errors)
    assert any("round-trip" in item.lower() for item in errors)
    assert any("duration" in item.lower() for item in errors)
    assert any("turn count" in item.lower() for item in errors)
    assert any("iteration count" in item.lower() for item in errors)


def test_release_gate_rejects_unhealthy_phase3_accuracy_report():
    module = _load_release_gate_module()
    accuracy_report = {
        "suite_name": "WrongSuite",
        "overall_score": 0.0,
        "passed": False,
        "component_reports": {
            "agent_dialogue": {"passed": False},
            "sara_inference": {"passed": True},
        },
        "focus_summary": {
            "few_shot": {"passed": False, "score": 0.0},
        },
        "trend": {
            "has_previous": True,
            "regression_count": 2,
        },
    }

    errors = module.validate_phase3_accuracy_report(accuracy_report)

    assert errors
    assert any("unexpected suite name" in item.lower() for item in errors)
    assert any("did not pass" in item.lower() for item in errors)
    assert any("overall score" in item.lower() for item in errors)
    assert any("missing required components" in item.lower() for item in errors)
    assert any("focus summaries" in item.lower() for item in errors)
    assert any("regression" in item.lower() for item in errors)


def test_release_gate_accepts_embedded_accuracy_results_in_release_report():
    module = _load_release_gate_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "min_duration_seconds": 5.0,
            "min_agent_turns": 8,
            "min_inference_iterations": 12,
            "min_pattern_count": 1,
            "require_phase3_accuracy": True,
        },
        "agent": {"history_bounded": True, "issue_count": 0, "turns": 8},
        "inference": {
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 10,
            "iterations": 12,
        },
        "accuracy": {
            "suite_name": "Phase3AccuracySuite",
            "overall_score": 0.91,
            "passed": True,
            "component_reports": {
                "agent_dialogue": {"passed": True, "overall_score": 0.9},
                "sara_inference": {"passed": True, "overall_score": 1.0},
                "spiking_llm": {"passed": True, "overall_score": 0.85},
            },
            "focus_summary": {
                "few_shot": {"passed": True, "score": 1.0},
                "continual": {"passed": True, "score": 1.0},
            },
            "trend": {
                "has_previous": True,
                "regression_count": 0,
            },
        },
        "release_metadata": {
            "versions_match": True,
            "has_expected_console_scripts": True,
            "release_notes_heading": "Current Pre-Release",
        },
    }

    assert module.validate_release_report(report) == []


def test_release_gate_rejects_missing_embedded_accuracy_when_required():
    module = _load_release_gate_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "min_duration_seconds": 5.0,
            "min_agent_turns": 8,
            "min_inference_iterations": 12,
            "min_pattern_count": 1,
            "require_phase3_accuracy": True,
        },
        "agent": {"history_bounded": True, "issue_count": 0, "turns": 8},
        "inference": {
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 10,
            "iterations": 12,
        },
    }

    errors = module.validate_release_report(report)

    assert errors
    assert any("embedded phase 3 accuracy" in item.lower() for item in errors)


def test_release_gate_rejects_invalid_embedded_release_metadata():
    module = _load_release_gate_module()
    report = {
        "duration_seconds": 5.0,
        "criteria": {
            "min_duration_seconds": 5.0,
            "min_agent_turns": 8,
            "min_inference_iterations": 12,
            "min_pattern_count": 1,
            "require_phase3_accuracy": False,
        },
        "agent": {"history_bounded": True, "issue_count": 0, "turns": 8},
        "inference": {
            "roundtrip_ok": True,
            "tuple_keys_only": True,
            "pattern_count": 10,
            "iterations": 12,
        },
        "release_metadata": {
            "versions_match": False,
            "has_expected_console_scripts": False,
            "release_notes_heading": "",
        },
    }

    errors = module.validate_release_report(report)

    assert errors
    assert any("mismatched package versions" in item.lower() for item in errors)
    assert any("missing console scripts" in item.lower() for item in errors)
    assert any("release notes heading" in item.lower() for item in errors)
