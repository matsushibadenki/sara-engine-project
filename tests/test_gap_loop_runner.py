import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import processed_data_path, workspace_path


def _load_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "bot", "run_gap_loop.py")
    )
    spec = importlib.util.spec_from_file_location("run_gap_loop", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_records(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [
        {
            "source": "hot_inbox",
            "record_text": (
                "Sparse event routing is a CPU first SNN technique. "
                "It keeps runtime state bounded and supports source backed retrieval."
            ),
            "meta": {
                "quality": 0.9,
                "source_type": "offline_batch",
                "path": "data/raw/offline_batch_inbox/sparse.txt",
                "language": "en",
                "license_hint": "operator_supplied",
                "compliance_level": "allow",
            },
            "ts": "2026-06-02T00:00:00",
        },
        {
            "source": "web",
            "record_text": (
                "Local plasticity means synapses update from nearby events. "
                "Then replay can preserve useful traces without dense backpropagation."
            ),
            "meta": {
                "quality": 0.82,
                "source_type": "official_docs",
                "url": "https://example.org/local-plasticity",
                "language": "en",
                "license_hint": "reference",
                "compliance_level": "allow",
            },
            "ts": "2026-06-02T00:01:00",
        },
    ]
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_fixture_request_plan(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "schema": "sara-autobot-material-request-plan-v1",
        "request_count": 2,
        "request_source": "fixture_feedback",
        "requests": [
            {
                "request_id": "fixture_counterexample_gap",
                "material_types": ["contrastive_pair", "counterexample", "qa_pair"],
                "missing_material_types": ["counterexample"],
                "evaluation_gaps": ["negative_control", "contrastive_control"],
                "priority": 0.8,
                "reason": "fixture counterexample pressure gap",
            },
            {
                "request_id": "fixture_source_diversity_gap",
                "material_types": ["source_claim", "qa_pair", "transcript_segment"],
                "missing_material_types": ["transcript_segment"],
                "evaluation_gaps": ["retrieval_grounding"],
                "priority": 1.0,
                "reason": "fixture source diversity gap",
            },
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def test_gap_loop_runner_builds_and_enqueues_gap_cycle():
    module = _load_module()
    records_path = processed_data_path("autobot", "test_gap_loop_records.jsonl")
    fixture_request_plan_path = workspace_path("autobot", "test_gap_loop_fixture_request_plan.json")
    accepted_path = processed_data_path("autobot", "test_gap_loop_learning_materials.jsonl")
    curriculum_path = processed_data_path("autobot", "test_gap_loop_curriculum_manifest.jsonl")
    collection_targets_path = workspace_path("autobot", "test_gap_loop_collection_targets.json")
    gap_output_path = processed_data_path("autobot", "test_gap_loop_gap_materials.jsonl")
    gap_curriculum_path = processed_data_path("autobot", "test_gap_loop_gap_curriculum.jsonl")
    queue_path = workspace_path("autobot", "test_gap_loop_train_queue.json")
    report_path = workspace_path("autobot", "test_gap_loop_report.json")
    summary_path = workspace_path("autobot", "test_gap_loop_summary.txt")
    _write_records(records_path)
    _write_fixture_request_plan(fixture_request_plan_path)

    exit_code = module.main(
        [
            "--records-path",
            records_path,
            "--accepted-path",
            accepted_path,
            "--curriculum-path",
            curriculum_path,
            "--fixture-request-plan-path",
            fixture_request_plan_path,
            "--collection-targets-path",
            collection_targets_path,
            "--gap-output-path",
            gap_output_path,
            "--gap-curriculum-path",
            gap_curriculum_path,
            "--queue-path",
            queue_path,
            "--report-path",
            report_path,
            "--summary-path",
            summary_path,
            "--evaluation-gap",
            "negative_control",
        ]
    )

    assert exit_code == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["passed"] is True
    assert report["dataset_accepted_count"] >= 8
    assert report["collection_target_count"] >= 2
    assert report["gap_material_built_count"] >= 2
    assert report["gap_curriculum_enqueued_count"] >= 2
    with open(queue_path, "r", encoding="utf-8") as handle:
        queue_rows = json.load(handle)
    assert len(queue_rows) >= 2
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = handle.read()
    assert "Gap loop: PASS" in summary


def test_gap_loop_runner_persists_blocked_requests_into_targets_and_builder():
    module = _load_module()
    records_path = processed_data_path("autobot", "test_gap_loop_blocked_records.jsonl")
    fixture_request_plan_path = workspace_path("autobot", "test_gap_loop_blocked_fixture_request_plan.json")
    accepted_path = processed_data_path("autobot", "test_gap_loop_blocked_learning_materials.jsonl")
    curriculum_path = processed_data_path("autobot", "test_gap_loop_blocked_curriculum_manifest.jsonl")
    collection_targets_path = workspace_path("autobot", "test_gap_loop_blocked_collection_targets.json")
    gap_output_path = processed_data_path("autobot", "test_gap_loop_blocked_gap_materials.jsonl")
    gap_curriculum_path = processed_data_path("autobot", "test_gap_loop_blocked_gap_curriculum.jsonl")
    queue_path = workspace_path("autobot", "test_gap_loop_blocked_train_queue.json")
    report_path = workspace_path("autobot", "test_gap_loop_blocked_report.json")
    summary_path = workspace_path("autobot", "test_gap_loop_blocked_summary.txt")
    _write_records(records_path)
    _write_fixture_request_plan(fixture_request_plan_path)

    exit_code = module.main(
        [
            "--records-path",
            records_path,
            "--accepted-path",
            accepted_path,
            "--curriculum-path",
            curriculum_path,
            "--fixture-request-plan-path",
            fixture_request_plan_path,
            "--collection-targets-path",
            collection_targets_path,
            "--gap-output-path",
            gap_output_path,
            "--gap-curriculum-path",
            gap_curriculum_path,
            "--queue-path",
            queue_path,
            "--report-path",
            report_path,
            "--summary-path",
            summary_path,
            "--blocked-request-id",
            "fixture_counterexample_gap",
        ]
    )

    assert exit_code == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["blocked_request_ids"] == ["fixture_counterexample_gap"]
    assert report["gap_material_built_count"] >= 1
    with open(collection_targets_path, "r", encoding="utf-8") as handle:
        targets = json.load(handle)
    assert targets["blocked_request_ids"] == ["fixture_counterexample_gap"]
    with open(gap_output_path, "r", encoding="utf-8") as handle:
        gap_rows = [json.loads(line) for line in handle if line.strip()]
    assert "fixture_counterexample_gap" not in {
        row.get("request_id", "") for row in gap_rows
    }
