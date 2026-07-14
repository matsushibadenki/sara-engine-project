import importlib.util
import json
import os


def _load_module():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "phase13_completion_gate.py"))
    spec = importlib.util.spec_from_file_location("phase13_completion_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase13_gate_accepts_observed_capability_reports(monkeypatch, tmp_path):
    module = _load_module()
    for filename in module.CAPABILITIES.values():
        (tmp_path / filename).write_text(json.dumps({"passed": True, "observed_only": True, "schema": "test-v1", "case_count": 3}), encoding="utf-8")
    monkeypatch.setattr(module, "workspace_path", lambda *parts: str(tmp_path / parts[-1]))
    report = module.build_report()
    assert report["phase13_complete"] is True
    assert len(report["capabilities"]) == 8


def test_phase13_gate_rejects_missing_capability(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setattr(module, "workspace_path", lambda *parts: str(tmp_path / parts[-1]))
    report = module.build_report()
    assert report["phase13_complete"] is False
