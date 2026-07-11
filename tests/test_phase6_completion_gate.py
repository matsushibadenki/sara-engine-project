import importlib.util
import os
import sys


def _load_module():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "phase6_completion_gate.py"))
    spec = importlib.util.spec_from_file_location("phase6_completion_gate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase6_gate_separates_implementation_readiness_from_missing_measurements():
    module = _load_module()
    report = module.build_report({"protocol_ready": True, "measurement_session_plan": {"planned_runs": [1]}, "checks": {"protocol": True}, "status": "protocol_ready_pending_measurements", "measurement_session_progress": {"status": "pending"}})
    assert report["implementation_ready"] is True
    assert report["phase6_complete"] is False
    assert report["status"] == "implementation_complete_physical_measurement_pending"


def test_phase6_gate_accepts_valid_completed_physical_evidence():
    module = _load_module()
    report = module.build_report({"protocol_ready": True, "measurement_session_plan": {"planned_runs": [1]}, "checks": {"protocol": True}, "has_real_measurements": True, "measurement_count": 4, "status": "real_joule_evidence_passed", "passed": True, "measurement_session_progress": {"status": "complete", "planned_pair_count": 2, "complete_valid_pair_count": 2}})
    assert report["phase6_complete"] is True
    assert report["status"] == "phase6_complete"
