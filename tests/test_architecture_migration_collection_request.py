import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import workspace_path


def _load_module():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "architecture_migration_collection_request.py"))
    spec = importlib.util.spec_from_file_location("architecture_migration_collection_request", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collection_request_turns_external_gate_failure_into_managed_target():
    module = _load_module()
    gate_path = workspace_path("evaluation", "test_architecture_migration_gate.json")
    targets_path = workspace_path("autobot", "test_architecture_migration_targets.json")
    with open(gate_path, "w", encoding="utf-8") as handle:
        json.dump({"schema": "sara-architecture-migration-external-gate-v1", "blocked_reasons": ["insufficient_independent_source_domains"]}, handle)

    assert module.main(["--gate-path", gate_path, "--targets-path", targets_path, "--report-path", workspace_path("evaluation", "test_architecture_migration_request.json")]) == 0
    with open(targets_path, "r", encoding="utf-8") as handle:
        targets = json.load(handle)
    assert targets["target_count"] == 1
    requirements = targets["targets"][0]["architecture_migration_requirements"]
    assert requirements["minimum_distinct_https_source_sites"] == 2
    assert requirements["minimum_records_per_independent_source_site"] == 3
