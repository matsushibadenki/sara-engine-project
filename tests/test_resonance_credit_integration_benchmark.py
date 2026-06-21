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
            "resonance_credit_integration_benchmark.py",
        )
    )
    spec = importlib.util.spec_from_file_location(
        "resonance_credit_integration_benchmark", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_integration_benchmark_bridges_managed_report_shapes():
    module = _load_module()
    paths = {
        name: workspace_path("evaluation", f"test_bridge_{name}.json")
        for name in module.DEFAULT_SOURCE_PATHS
    }
    reports = {
        "reasoning_prior": {
            "schema": "reason-v1",
            "passed": True,
            "observed_only": True,
            "metrics": {
                "logic_to_state_consistency": 1.0,
                "external_event_missing_abstention": 1.0,
            },
        },
        "plan_verifier": {
            "schema": "plan-v1",
            "passed": True,
            "observed_only": True,
            "case_count": 2,
            "expected_match_count": 2,
        },
        "multimodal_binding": {
            "schema": "multi-v1",
            "passed": True,
            "observed_only": True,
            "metrics": {
                "cross_modal_link_precision": 1.0,
                "route_traceability": 1.0,
                "missing_modality_abstention_integrity": 1.0,
            },
        },
        "dendritic_feedback": {
            "schema": "dendritic-v1",
            "passed": True,
            "observed_only": True,
            "gated_precision": 0.9,
        },
        "own_latent": {
            "schema": "latent-v1",
            "passed": True,
            "observed_only": True,
            "metrics": {"own_latent_sample_efficiency_ok": 1.0},
        },
    }
    for name, path in paths.items():
        _write(path, reports[name])
    report_path = workspace_path("evaluation", "test_resonance_integration.json")
    argv = []
    for name, path in paths.items():
        argv.extend([f"--{name.replace('_', '-')}-path", path])
    argv.extend(
        [
            "--trace-path",
            workspace_path("evaluation", "test_resonance_integration_traces.jsonl"),
            "--report-path",
            report_path,
            "--summary-path",
            workspace_path("evaluation", "test_resonance_integration.txt"),
        ]
    )

    assert module.main(argv) == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["passed"] is True
    assert report["metrics"]["decision_integrity"] == 1.0
    assert report["metrics"]["integration_freeze_case_count"] == 4
