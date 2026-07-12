import importlib.util
import os
import sys


def _load_module():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "phase8_completion_gate.py"))
    spec = importlib.util.spec_from_file_location("phase8_completion_gate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _comparison(stronger_reference: bool):
    checks = {
        "external_validity_passed": True,
        "ladder_passed": True,
        "bm25_reference_present": True,
        "stronger_real_reference_present": stronger_reference,
        "per_task_summary_present": True,
        "quality_and_cost_reported_together": True,
        "offline_references_labeled": True,
    }
    return {"schema": "sara-ann-comparison-report-v1", "checks": checks, "baseline_cards": [], "reference_readiness": {}, "status": "proxy_only_or_partial_reference_surface"}


def test_phase8_gate_separates_implementation_from_missing_stronger_reference():
    report = _load_module().build_report(_comparison(False))
    assert report["implementation_ready"] is True
    assert report["phase8_complete"] is False
    assert report["status"] == "implementation_complete_stronger_baseline_pending"


def test_phase8_gate_accepts_frozen_stronger_reference_evidence():
    report = _load_module().build_report(_comparison(True))
    assert report["phase8_complete"] is True
    assert report["physical_evidence_separate"] is True
