import importlib.util
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_module():
    path = os.path.join(ROOT, "scripts", "eval", "phase19_completion_gate.py")
    spec = importlib.util.spec_from_file_location("phase19_completion_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phase19_gate_passes_existing_observed_evidence():
    module = _load_module()
    report = module.build_report()
    assert report["phase19_complete"] is True
    assert all(report["checks"].values())
    assert report["promotion_rule"]["release_critical"] is False


def test_phase19_gate_rejects_missing_control_improvement(tmp_path):
    module = _load_module()
    path = tmp_path / "benchmark.json"
    path.write_text('{"passed":true,"observed_only":true,"case_count":4,"metrics":{"liquid_improves_fixed":0,"liquid_improves_multiscale":1,"replay_determinism":1,"abstention_integrity":1,"max_event_cost":4,"max_update_count":1,"max_state_budget_units":1,"max_time_constant":8},"policy_notes":["sparse CPU closed-form backpropagation; fixed-time-constant SNN remains the default and does not alter production"]}', encoding="utf-8")
    report = module.build_report(benchmark_path=str(path))
    assert report["phase19_complete"] is False
    assert report["checks"]["improves_fixed_control"] is False
