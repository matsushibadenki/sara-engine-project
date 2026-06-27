import importlib.util
import json
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_module():
    path = os.path.join(ROOT, "scripts", "eval", "internal_maintenance_efficiency_benchmark.py")
    spec = importlib.util.spec_from_file_location("internal_maintenance_efficiency_benchmark", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_internal_maintenance_efficiency_benchmark_passes():
    module = _load_module()

    report = module.build_report()

    assert report["schema"] == "sara-internal-maintenance-efficiency-benchmark-v1"
    assert report["observed_only"] is True
    assert report["passed"] is True
    assert report["counts"]["maintenance_selected_count"] >= 1
    assert report["counts"]["maintenance_refresh_count"] >= 1
    assert report["counts"]["maintenance_idle_self_state_ok_count"] >= 1
    assert report["counts"]["maintenance_predicted_event_count"] >= 1
    assert report["metrics"]["maintenance_self_state_continuity_observed"] == 1.0
    assert report["metrics"]["maintenance_event_cost_efficiency_observed"] == 1.0
    assert report["normalized_metrics"]["maintenance_event_cost_per_selected"] >= 0.0


def test_internal_maintenance_efficiency_benchmark_writes_outputs():
    module = _load_module()
    report_path = module.workspace_path("evaluation", "test_internal_maintenance_efficiency_benchmark.json")
    summary_path = module.workspace_path("evaluation", "test_internal_maintenance_efficiency_benchmark_summary.txt")

    report = module.run_benchmark(report_path=report_path, summary_path=summary_path)

    assert report["passed"] is True
    with open(report_path, "r", encoding="utf-8") as handle:
        saved = json.load(handle)
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = handle.read()
    assert saved["schema"] == "sara-internal-maintenance-efficiency-benchmark-v1"
    assert "SARA internal maintenance efficiency benchmark" in summary
