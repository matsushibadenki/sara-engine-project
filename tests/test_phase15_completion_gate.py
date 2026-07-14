import importlib.util
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_module():
    path = os.path.join(ROOT, "scripts", "eval", "phase15_completion_gate.py")
    spec = importlib.util.spec_from_file_location("phase15_completion_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phase15_gate_passes_existing_observed_evidence():
    module = _load_module()
    report = module.build_report()
    assert report["phase15_complete"] is True
    assert all(report["checks"].values())
    assert report["promotion_rule"]["release_critical"] is False


def test_phase15_gate_rejects_unbounded_event_cost(tmp_path):
    module = _load_module()
    path = tmp_path / "benchmark.json"
    path.write_text(
        '{"passed": true, "observed_only": true, "case_count": 4, "robustness_delta": 0.1, '
        '"fallback_rate": 0, "max_event_cost": 999, "max_state_budget_units": 8, '
        '"rows": [{"case_id":"x", "convergence_steps":1}], '
        '"policy_notes":["sparse CPU-first bounded-state backpropagation-free; does not alter default production inference"]}',
        encoding="utf-8",
    )
    report = module.build_report(benchmark_path=str(path))
    assert report["phase15_complete"] is False
    assert report["checks"]["event_cost_bounded"] is False
