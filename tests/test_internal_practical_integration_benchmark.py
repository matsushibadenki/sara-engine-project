import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import workspace_path


def _load_module():
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "scripts",
            "eval",
            "internal_practical_integration_benchmark.py",
        )
    )
    spec = importlib.util.spec_from_file_location("internal_practical_integration_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_internal_practical_integration_benchmark_passes_without_external_device():
    module = _load_module()
    report_path = workspace_path("evaluation", "test_internal_practical_integration.json")
    summary_path = workspace_path("evaluation", "test_internal_practical_integration.txt")

    report = module.run_benchmark(report_path=report_path, summary_path=summary_path)

    assert report["passed"] is True
    assert report["external_device_required"] is False
    assert report["execution_policy"]["cpu_only"] is True
    assert report["checks"]["practical_task_quality"] is True
    assert report["checks"]["continual_learning_and_drift_recovery"] is True
    assert report["checks"]["revision_uptake_against_frozen_control"] is True
    assert report["checks"]["source_grounding_and_citation"] is True
    assert report["checks"]["architecture_change_knowledge_reuse"] is True
    assert report["checks"]["reproducible_state_migration"] is True
    assert report["metrics"]["practical_task_count"] >= 6

    with open(report_path, "r", encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert persisted["schema"] == "sara-internal-practical-integration-benchmark-v1"
    with open(summary_path, "r", encoding="utf-8") as handle:
        assert "Internal practical integration benchmark: PASS" in handle.read()
