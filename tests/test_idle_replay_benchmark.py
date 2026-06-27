import importlib.util
import json
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_module():
    path = os.path.join(ROOT, "scripts", "eval", "idle_replay_benchmark.py")
    spec = importlib.util.spec_from_file_location("idle_replay_benchmark", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_idle_replay_benchmark_report_passes():
    module = _load_module()

    report = module.build_report()

    assert report["schema"] == "sara-idle-replay-benchmark-v1"
    assert report["observed_only"] is True
    assert report["passed"] is True
    assert report["metrics"]["idle_replay_candidate_selection_observed"] == 1.0
    assert report["metrics"]["idle_replay_budget_observed"] == 1.0
    assert report["metrics"]["idle_replay_self_state_alignment_observed"] == 1.0
    assert report["metrics"]["idle_replay_memory_reactivation_observed"] == 1.0
    assert report["metrics"]["idle_replay_state_continuity_observed"] == 1.0
    assert report["metrics"]["idle_replay_astro_modulation_observed"] == 1.0
    assert report["traces"]["aligned"]["selected"][0]["entry_id"] == "aligned"


def test_idle_replay_benchmark_writes_managed_outputs(tmp_path):
    module = _load_module()
    report_path = module.workspace_path("evaluation", "test_idle_replay_benchmark.json")
    summary_path = module.workspace_path("evaluation", "test_idle_replay_benchmark_summary.txt")

    report = module.run_benchmark(report_path=report_path, summary_path=summary_path)

    assert report["passed"] is True
    with open(report_path, "r", encoding="utf-8") as handle:
        saved = json.load(handle)
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = handle.read()
    assert saved["schema"] == "sara-idle-replay-benchmark-v1"
    assert "SARA idle replay benchmark" in summary
