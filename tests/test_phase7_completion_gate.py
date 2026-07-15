import importlib.util
import os
import sys


def _load_module():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "phase7_completion_gate.py"))
    spec = importlib.util.spec_from_file_location("phase7_completion_gate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _readiness():
    return {"schema": "sara-autobot-gap-loop-readiness-v1", "passed": True, "checks": {"gap_enqueue_ready": {"passed": True}}}


def test_phase7_gate_separates_implementation_from_missing_isolation_evidence():
    module = _load_module()
    report = module.build_report(_readiness(), {"schema": "sara-phase7-isolation-audit-v1", "passed": False, "checks": {"train_rows_present": False}, "metrics": {}})
    assert report["implementation_ready"] is True
    assert report["phase7_complete"] is False
    assert report["status"] == "implementation_complete_isolation_evidence_pending"


def test_phase7_gate_accepts_isolated_gap_loop_evidence():
    module = _load_module()
    report = module.build_report(_readiness(), {"schema": "sara-phase7-isolation-audit-v1", "passed": True, "checks": {"train_rows_present": True, "independent_evidence_scope_valid": True}, "metrics": {"train_row_count": 3, "evaluation_row_count": 2}})
    assert report["phase7_complete"] is True
    assert report["status"] == "phase7_complete"


def test_phase7_gate_rejects_fixture_pass_without_independent_scope():
    module = _load_module()
    report = module.build_report(_readiness(), {"schema": "sara-phase7-isolation-audit-v1", "passed": True, "checks": {"train_rows_present": True}, "metrics": {"train_row_count": 3, "evaluation_row_count": 2}})
    assert report["phase7_complete"] is False
    assert report["independent_evidence_scope_valid"] is False
