import importlib.util
import json
import os


def _load_module():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "phase12_completion_gate.py"))
    spec = importlib.util.spec_from_file_location("phase12_completion_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase12_gate_accepts_dashboard_and_guide(tmp_path):
    module = _load_module()
    dashboard = tmp_path / "dashboard.json"
    dashboard.write_text(json.dumps({
        "schema": "sara-operator-dashboard-v1",
        "artifact_states": {name: {"status": "passed"} for name in module.REQUIRED_ARTIFACTS},
        "next_actions": [{"command": "next"}],
        "operator_commands": {"refresh_dashboard": "refresh"},
        "what_is_proven": ["x"],
        "what_is_not_proven": ["y"],
    }), encoding="utf-8")
    guide = tmp_path / "guide.md"
    guide.write_text("Daily Review Reproduce Evidence Troubleshooting Managed output violation Physical energy pending", encoding="utf-8")
    report = module.build_report(dashboard_path=str(dashboard), guide_path=str(guide))
    assert report["phase12_complete"] is True


def test_phase12_gate_rejects_missing_dashboard(tmp_path):
    module = _load_module()
    guide = tmp_path / "guide.md"
    guide.write_text("Daily Review Reproduce Evidence Troubleshooting Managed output violation Physical energy pending", encoding="utf-8")
    report = module.build_report(dashboard_path=str(tmp_path / "missing.json"), guide_path=str(guide))
    assert report["phase12_complete"] is False
