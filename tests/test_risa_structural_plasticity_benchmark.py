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
            "risa_structural_plasticity_benchmark.py",
        )
    )
    spec = importlib.util.spec_from_file_location("risa_structural_plasticity_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_risa_structural_plasticity_benchmark_keeps_verified_route_under_equal_budget():
    module = _load_module()
    report_path = workspace_path("evaluation", "test_risa_structural_plasticity_benchmark.json")
    exit_code = module.main(
        [
            "--report-path",
            report_path,
            "--summary-path",
            workspace_path("evaluation", "test_risa_structural_plasticity_benchmark.txt"),
        ]
    )

    assert exit_code == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["passed"] is True
    assert report["frozen_fixture"] is True
    assert report["metrics"]["predictive_route_retention_improved"] == 1.0
    assert report["metrics"]["contradiction_recovery_maintained"] == 1.0
    assert report["metrics"]["maintenance_cost_equal"] == 1.0
