import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import processed_data_path, workspace_path


def _load_module():
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "scripts",
            "eval",
            "physical_energy_pair_runner.py",
        )
    )
    spec = importlib.util.spec_from_file_location("physical_energy_pair_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_physical_energy_pair_runner_dry_run_freezes_manifest():
    module = _load_module()
    manifest_path = workspace_path("evaluation", "test_physical_pair_manifest.json")
    trace_path = workspace_path("evaluation", "test_physical_pair_trace.jsonl")

    exit_code = module.main(
        [
            "--pair-id",
            "test-pair-1",
            "--replicate-index",
            "1",
            "--corpus-path",
            processed_data_path("corpus.txt"),
            "--manifest-path",
            manifest_path,
            "--trace-path",
            trace_path,
            "--dry-run",
        ]
    )

    assert exit_code == 0
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["protocol_version"] == "sara-energy-fair-comparison-v2"
    assert manifest["run_order"] == ["sara", "ann"]
    assert len(manifest["task_fixture_hash"]) == 64
    with open(trace_path, "r", encoding="utf-8") as handle:
        traces = [json.loads(line) for line in handle if line.strip()]
    assert [trace["system"] for trace in traces] == ["sara", "ann"]
    assert all(trace["status"] == "planned" for trace in traces)


def test_physical_energy_pair_runner_alternates_even_replicate_order():
    module = _load_module()
    manifest = module.build_manifest(
        corpus_path=processed_data_path("corpus.txt"),
        replicate_index=2,
        repetitions=3,
        warmup_count=1,
        thread_count=1,
        process_affinity="unbound",
        power_mode="ac",
        measurement_tool="manual",
        pair_id="test-pair-2",
    )

    assert manifest["run_order"] == ["ann", "sara"]
