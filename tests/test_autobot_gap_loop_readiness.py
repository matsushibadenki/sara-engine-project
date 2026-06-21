import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import processed_data_path, workspace_path


def _load_gap_loop_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "bot", "run_gap_loop.py")
    )
    spec = importlib.util.spec_from_file_location("run_gap_loop_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_readiness_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "autobot_gap_loop_readiness.py")
    )
    spec = importlib.util.spec_from_file_location("autobot_gap_loop_readiness", module_path)
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


def test_autobot_gap_loop_readiness_reports_pass_on_managed_gap_cycle():
    gap_loop = _load_gap_loop_module()
    readiness = _load_readiness_module()
    records_path = processed_data_path("autobot", "test_gap_readiness_records.jsonl")
    fixture_request_plan_path = workspace_path("autobot", "test_gap_readiness_fixture_request_plan.json")
    accepted_path = processed_data_path("autobot", "test_gap_readiness_learning_materials.jsonl")
    curriculum_path = processed_data_path("autobot", "test_gap_readiness_curriculum_manifest.jsonl")
    collection_targets_path = workspace_path("autobot", "test_gap_readiness_collection_targets.json")
    gap_output_path = processed_data_path("autobot", "test_gap_readiness_gap_materials.jsonl")
    gap_curriculum_path = processed_data_path("autobot", "test_gap_readiness_gap_curriculum.jsonl")
    queue_path = workspace_path("autobot", "test_gap_readiness_train_queue.json")
    loop_report_path = workspace_path("autobot", "test_gap_readiness_loop_report.json")
    loop_summary_path = workspace_path("autobot", "test_gap_readiness_loop_summary.txt")
    readiness_report_path = workspace_path("evaluation", "test_autobot_gap_loop_readiness.json")
    readiness_summary_path = workspace_path("evaluation", "test_autobot_gap_loop_readiness.txt")
    _write_records(records_path)
    _write_fixture_request_plan(fixture_request_plan_path)

    loop_report = gap_loop.run_gap_loop(
        records_path=records_path,
        accepted_path=accepted_path,
        curriculum_path=curriculum_path,
        fixture_request_plan_path=fixture_request_plan_path,
        collection_targets_path=collection_targets_path,
        gap_output_path=gap_output_path,
        gap_curriculum_path=gap_curriculum_path,
        queue_path=queue_path,
        report_path=loop_report_path,
        summary_path=loop_summary_path,
        evaluation_gaps=("negative_control",),
    )
    assert loop_report["passed"] is True

    report = readiness.run_readiness(
        loop_report_path=loop_report_path,
        collection_targets_path=collection_targets_path,
        report_path=readiness_report_path,
        summary_path=readiness_summary_path,
        min_accepted_count=4,
        min_gap_build_coverage=1.0,
    )

    assert report["passed"] is True
    assert report["metrics"]["requested_slot_count"] == 2
    assert report["metrics"]["gap_material_built_count"] >= 2
    assert report["metrics"]["gap_curriculum_enqueued_count"] >= 2
    assert report["checks"]["gap_material_coverage_ready"]["passed"] is True
    assert report["checks"]["gap_enqueue_ready"]["passed"] is True
    with open(readiness_summary_path, "r", encoding="utf-8") as handle:
        summary = handle.read()
    assert "Autobot gap loop readiness: PASS" in summary
