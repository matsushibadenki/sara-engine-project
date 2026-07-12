import importlib.util
import os
import sys


def _load_module():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "phase8_evidence_cycle.py"))
    spec = importlib.util.spec_from_file_location("phase8_evidence_cycle", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase8_cycle_uses_completion_gate_as_promotion_authority():
    module = _load_module()
    report = module.build_cycle_report(
        external_validity={"passed": True},
        ladder={"passed": True},
        comparison={"passed": False},
        gate={"phase8_complete": False, "implementation_ready": True, "status": "implementation_complete_stronger_baseline_pending", "next_action": "configure reference"},
        return_codes={"external_validity": 0, "ladder": 0, "comparison": 1, "completion_gate": 1},
    )
    assert report["passed"] is False
    assert report["implementation_ready"] is True
    assert report["stages"]["comparison_passed"] is False
    assert report["status"] == "implementation_complete_stronger_baseline_pending"
