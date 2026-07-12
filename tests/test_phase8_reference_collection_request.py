import importlib.util
import os
import sys


def _load_module():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "phase8_reference_collection_request.py"))
    spec = importlib.util.spec_from_file_location("phase8_reference_collection_request", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase8_reference_request_preserves_missing_checks_without_downloading():
    request = _load_module().build_request({
        "schema": "sara-phase8-completion-gate-v1",
        "status": "implementation_complete_stronger_baseline_pending",
        "phase8_complete": False,
        "required_checks": {"bm25_reference_present": True, "stronger_real_reference_present": False},
    })
    assert request["blocked"] is True
    assert request["target_count"] == 1
    assert request["targets"][0]["missing_checks"] == ["stronger_real_reference_present"]
    assert request["targets"][0]["requirements"]["local_files_only"] is True


def test_phase8_reference_request_is_empty_after_completion():
    request = _load_module().build_request({"schema": "sara-phase8-completion-gate-v1", "phase8_complete": True})
    assert request["blocked"] is False
    assert request["target_count"] == 0
