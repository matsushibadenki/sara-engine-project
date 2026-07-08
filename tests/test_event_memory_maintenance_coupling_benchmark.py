import importlib.util
import json
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_module():
    path = os.path.join(
        ROOT,
        "scripts",
        "eval",
        "event_memory_maintenance_coupling_benchmark.py",
    )
    spec = importlib.util.spec_from_file_location(
        "event_memory_maintenance_coupling_benchmark",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_event_memory_maintenance_coupling_exposes_profile_tradeoffs():
    module = _load_module()

    report = module.build_report()

    assert report["schema"] == "sara-event-memory-maintenance-coupling-benchmark-v1"
    assert report["observed_only"] is True
    assert report["passed"] is True
    assert report["profile_count"] == 3
    assert len(report["profiles"]) == 3
    assert report["best_profile"]["profile_id"] in {"tight", "balanced", "wide"}
    assert (
        report["metrics"]["best_profile_compression_efficiency_per_maintenance"] > 0.0
    )
    assert report["metrics"]["best_profile_episode_compression_ratio"] >= 1.0
    assert report["metrics"]["best_profile_self_state_continuity"] >= 0.0
    assert report["metrics"]["best_profile_multimodal_bundle_compression_contribution"] > 0.0
    assert all(
        float(profile["metrics"]["avg_multimodal_bundle_promotion_rate"]) > 0.0
        for profile in report["profiles"]
    )
    summary = module.build_summary(report)
    assert "compression_to_maintenance_correlation:" in summary
    assert "best_profile_compression_efficiency_per_maintenance:" in summary
    assert "best_profile_multimodal_bundle_compression_contribution:" in summary


def test_event_memory_maintenance_coupling_writes_outputs():
    module = _load_module()
    report_path = module.workspace_path(
        "evaluation",
        "test_event_memory_maintenance_coupling_benchmark.json",
    )
    summary_path = module.workspace_path(
        "evaluation",
        "test_event_memory_maintenance_coupling_benchmark_summary.txt",
    )

    report = module.run_benchmark(report_path=report_path, summary_path=summary_path)

    assert report["passed"] is True
    with open(report_path, "r", encoding="utf-8") as handle:
        saved = json.load(handle)
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = handle.read()
    assert saved["schema"] == "sara-event-memory-maintenance-coupling-benchmark-v1"
    assert "SARA Event Memory maintenance coupling benchmark" in summary
