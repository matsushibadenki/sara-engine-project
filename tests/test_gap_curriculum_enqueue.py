import importlib.util
import json
import os
import sys

from sara_engine.utils.project_paths import processed_data_path, workspace_path


def _load_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "bot", "enqueue_curriculum.py")
    )
    spec = importlib.util.spec_from_file_location("enqueue_curriculum", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_curriculum(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [
        {
            "manifest_id": "autobot-material-000001",
            "material_hash": "hash-counter",
            "material_type": "counterexample",
            "curriculum_stage": "repair",
            "priority": 1.05,
            "source": "web",
            "source_url": "https://example.org/a",
            "quality_score": 0.9,
        },
        {
            "manifest_id": "autobot-material-000002",
            "material_hash": "hash-transcript",
            "material_type": "transcript_segment",
            "curriculum_stage": "replay",
            "priority": 0.8,
            "source": "local",
            "source_path": "data/raw/example.txt",
            "quality_score": 0.9,
        },
    ]
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_gap_curriculum_enqueue_pushes_manifest_items_into_queue():
    module = _load_module()
    curriculum_path = processed_data_path("autobot", "test_gap_curriculum_enqueue.jsonl")
    queue_path = workspace_path("autobot", "test_gap_curriculum_train_queue.json")
    report_path = workspace_path("autobot", "test_gap_curriculum_enqueue_report.json")
    summary_path = workspace_path("autobot", "test_gap_curriculum_enqueue_summary.txt")
    _write_curriculum(curriculum_path)

    exit_code = module.main(
        [
            "--curriculum-path",
            curriculum_path,
            "--queue-path",
            queue_path,
            "--report-path",
            report_path,
            "--summary-path",
            summary_path,
        ]
    )

    assert exit_code == 0
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["passed"] is True
    assert report["enqueued_count"] == 2
    with open(queue_path, "r", encoding="utf-8") as handle:
        queue_rows = json.load(handle)
    assert len(queue_rows) == 2
    assert all(item["path"] == os.path.abspath(curriculum_path) for item in queue_rows)
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = handle.read()
    assert "Gap curriculum enqueue: PASS" in summary
