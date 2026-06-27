import importlib.util
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_module():
    path = os.path.join(ROOT, "scripts", "eval", "event_memory_ingest_pipeline.py")
    spec = importlib.util.spec_from_file_location("event_memory_ingest_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_event_memory_ingest_pipeline_exposes_compression_metrics():
    module = _load_module()

    report = module.build_report()

    assert report["schema"] == "sara-event-memory-ingest-pipeline-report-v1"
    assert report["passed"] is True
    assert report["metrics"]["eventization_emission_ratio"] > 0.0
    assert report["metrics"]["candidate_event_acceptance_rate"] >= 1.0
    assert report["metrics"]["episode_compression_ratio"] >= 1.0
    assert report["metrics"]["relation_verification_yield"] >= 1.0
    assert report["metrics"]["lineage_coverage_ratio"] >= 1.0
    assert report["metrics"]["self_state_continuity"] >= 0.0
    summary = module.build_summary(report)
    assert "episode_compression_ratio:" in summary
    assert "relation_verification_yield:" in summary

