from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from bot.training_queue import TrainingQueue
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path


DEFAULT_CURRICULUM_PATH = processed_data_path("autobot", "gap_curriculum_manifest.jsonl")
DEFAULT_QUEUE_PATH = workspace_path("autobot", "train_queue.json")
DEFAULT_REPORT_PATH = workspace_path("autobot", "gap_curriculum_enqueue_report.json")
DEFAULT_SUMMARY_PATH = workspace_path("autobot", "gap_curriculum_enqueue_summary.txt")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_json(path: str, payload: Dict[str, Any]) -> str:
    resolved = ensure_parent_directory(path)
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return resolved


def summarize_report(report: Dict[str, Any]) -> str:
    lines = [
        f"Gap curriculum enqueue: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Manifest items: {report.get('manifest_item_count')}",
        f"Enqueued: {report.get('enqueued_count')}",
        f"Queue pending: {report.get('queue_pending')}",
    ]
    return "\n".join(lines) + "\n"


def run_enqueue(
    *,
    curriculum_path: str = DEFAULT_CURRICULUM_PATH,
    queue_path: str = DEFAULT_QUEUE_PATH,
    report_path: str = DEFAULT_REPORT_PATH,
    summary_path: str = DEFAULT_SUMMARY_PATH,
    source_label: str = "autobot_gap_materials_builder",
) -> Dict[str, Any]:
    manifest_rows = read_jsonl(curriculum_path)
    prepared_rows: List[Dict[str, Any]] = []
    for row in manifest_rows:
        item = dict(row)
        item["path"] = os.path.abspath(curriculum_path)
        item["source"] = str(item.get("source", source_label) or source_label)
        prepared_rows.append(item)

    queue = TrainingQueue(queue_path)
    enqueued_count = queue.enqueue_learning_materials(prepared_rows)
    stats = queue.stats()
    report = {
        "schema": "sara-autobot-gap-curriculum-enqueue-report-v1",
        "passed": bool(prepared_rows) and enqueued_count > 0,
        "curriculum_path": os.path.abspath(curriculum_path),
        "queue_path": os.path.abspath(queue_path),
        "manifest_item_count": len(prepared_rows),
        "enqueued_count": int(enqueued_count),
        "queue_pending": int(stats.pending),
        "queue_recovered_count": int(stats.recovered_count),
    }
    report["report_path"] = write_json(report_path, report)
    resolved_summary_path = ensure_parent_directory(summary_path)
    with open(resolved_summary_path, "w", encoding="utf-8") as handle:
        handle.write(summarize_report(report))
    report["summary_path"] = resolved_summary_path
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enqueue gap curriculum materials into the autobot training queue.")
    parser.add_argument("--curriculum-path", default=DEFAULT_CURRICULUM_PATH)
    parser.add_argument("--queue-path", default=DEFAULT_QUEUE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--source-label", default="autobot_gap_materials_builder")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_enqueue(
        curriculum_path=args.curriculum_path,
        queue_path=args.queue_path,
        report_path=args.report_path,
        summary_path=args.summary_path,
        source_label=args.source_label,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "manifest_item_count": report["manifest_item_count"],
                "enqueued_count": report["enqueued_count"],
                "queue_pending": report["queue_pending"],
                "report_path": report["report_path"],
                "summary_path": report["summary_path"],
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
