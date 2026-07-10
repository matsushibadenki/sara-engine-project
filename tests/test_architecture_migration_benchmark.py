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
            "architecture_migration_benchmark.py",
        )
    )
    spec = importlib.util.spec_from_file_location("architecture_migration_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_architecture_migration_benchmark_replays_source_isolated_memory():
    module = _load_module()
    report_path = workspace_path("evaluation", "test_architecture_migration_benchmark.json")
    exit_code = module.main(
        [
            "--report-path",
            report_path,
            "--summary-path",
            workspace_path("evaluation", "test_architecture_migration_benchmark.txt"),
        ]
    )

    assert exit_code == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["passed"] is True
    assert report["source_isolated_fixture"] is True
    assert report["independent_external_source_evidence"] is False
    assert report["metrics"]["legacy_reference_recall"] == 1.0
    assert report["metrics"]["target_replay_recall"] == 1.0
    assert report["metrics"]["concept_review_recovered"] == 1.0
    assert report["metrics"]["risa_reconstruction_observed"] == 1.0
