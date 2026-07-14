import importlib.util
import json
import os


def _load_module():
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "phase11_completion_gate.py")
    )
    spec = importlib.util.spec_from_file_location("phase11_completion_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase11_gate_accepts_complete_matrix(tmp_path):
    module = _load_module()
    matrix = tmp_path / "matrix.json"
    rows = {
        profile: {
            "adapter": f"{profile}_profile",
            "state_trace_adapter_policy": "native_online_update",
            "unsupported_operations": [],
            "notes": "profile",
        }
        for profile in ["lava", "spinnaker", "akida"]
    }
    matrix.write_text(
        json.dumps(
            {
                "schema": "sara-neuromorphic-capability-matrix-report-v1",
                "passed": True,
                "profiles": ["lava", "spinnaker", "akida"],
                "cpu_reference": {"validated": True, "release_critical": True},
                "hardware_runtime_required": False,
                "capability_matrix": {
                    "all_profiles_compatible": True,
                    "unsupported_summary": {},
                    "common_event_ir": {"schema": "sara-spike-event-ir-v1", "budget_ok": True, "event_count": 3},
                    "profiles": rows,
                },
            }
        ),
        encoding="utf-8",
    )
    report = module.build_report(matrix_path=str(matrix))
    assert report["phase11_complete"] is True


def test_phase11_gate_rejects_missing_profile(tmp_path):
    module = _load_module()
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({"schema": "sara-neuromorphic-capability-matrix-report-v1", "passed": True}), encoding="utf-8")
    report = module.build_report(matrix_path=str(matrix))
    assert report["phase11_complete"] is False
