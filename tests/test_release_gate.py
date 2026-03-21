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
