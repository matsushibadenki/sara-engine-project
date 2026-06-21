from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from bot.dataset_builder import run_dataset_builder
from bot.enqueue_curriculum import run_enqueue
from bot.gap_materials_builder import run_builder as run_gap_materials_builder
from sara_engine.utils.project_paths import ensure_parent_directory, interim_data_path, processed_data_path, workspace_path


DEFAULT_RECORDS_PATH = processed_data_path("autobot", "multimodal_records.jsonl")
DEFAULT_CANDIDATE_PATH = interim_data_path("autobot", "candidate_learning_materials.jsonl")
DEFAULT_REJECTED_PATH = interim_data_path("autobot", "rejected_learning_materials.jsonl")
DEFAULT_ACCEPTED_PATH = processed_data_path("autobot", "learning_materials.jsonl")
DEFAULT_CURRICULUM_PATH = processed_data_path("autobot", "curriculum_manifest.jsonl")
DEFAULT_FIXTURE_REQUEST_PLAN_PATH = workspace_path("autobot", "fixture_material_request_plan.json")
DEFAULT_COLLECTION_TARGETS_PATH = workspace_path("autobot", "dataset_builder_collection_targets.json")
DEFAULT_GAP_OUTPUT_PATH = processed_data_path("autobot", "gap_materials.jsonl")
DEFAULT_GAP_CURRICULUM_PATH = processed_data_path("autobot", "gap_curriculum_manifest.jsonl")
DEFAULT_QUEUE_PATH = workspace_path("autobot", "train_queue.json")
DEFAULT_REPORT_PATH = workspace_path("autobot", "gap_loop_report.json")
DEFAULT_SUMMARY_PATH = workspace_path("autobot", "gap_loop_summary.txt")


def write_json(path: str, payload: Dict[str, Any]) -> str:
    resolved = ensure_parent_directory(path)
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return resolved


def summarize_report(report: Dict[str, Any]) -> str:
    lines = [
        f"Gap loop: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Dataset accepted: {report.get('dataset_accepted_count')}",
        f"Collection targets: {report.get('collection_target_count')}",
        f"Gap materials built: {report.get('gap_material_built_count')}",
        f"Gap curriculum enqueued: {report.get('gap_curriculum_enqueued_count')}",
        f"Queue pending: {report.get('queue_pending')}",
    ]
    return "\n".join(lines) + "\n"


def run_gap_loop(
    *,
    records_path: str = DEFAULT_RECORDS_PATH,
    candidate_path: str = DEFAULT_CANDIDATE_PATH,
    rejected_path: str = DEFAULT_REJECTED_PATH,
    accepted_path: str = DEFAULT_ACCEPTED_PATH,
    curriculum_path: str = DEFAULT_CURRICULUM_PATH,
    fixture_request_plan_path: str = DEFAULT_FIXTURE_REQUEST_PLAN_PATH,
    collection_targets_path: str = DEFAULT_COLLECTION_TARGETS_PATH,
    gap_output_path: str = DEFAULT_GAP_OUTPUT_PATH,
    gap_curriculum_path: str = DEFAULT_GAP_CURRICULUM_PATH,
    queue_path: str = DEFAULT_QUEUE_PATH,
    report_path: str = DEFAULT_REPORT_PATH,
    summary_path: str = DEFAULT_SUMMARY_PATH,
    evaluation_gaps: Sequence[str] = (),
) -> Dict[str, Any]:
    dataset_report = run_dataset_builder(
        records_path=records_path,
        candidate_path=candidate_path,
        rejected_path=rejected_path,
        accepted_path=accepted_path,
        curriculum_path=curriculum_path,
        report_path=workspace_path("autobot", "gap_loop_dataset_builder_report.json"),
        summary_path=workspace_path("autobot", "gap_loop_dataset_builder_summary.txt"),
        fixture_request_plan_path=fixture_request_plan_path,
        collection_targets_path=collection_targets_path,
        evaluation_gaps=evaluation_gaps,
    )
    gap_report = run_gap_materials_builder(
        accepted_path=accepted_path,
        targets_path=collection_targets_path,
        output_path=gap_output_path,
        curriculum_path=gap_curriculum_path,
        report_path=workspace_path("autobot", "gap_loop_gap_materials_report.json"),
        summary_path=workspace_path("autobot", "gap_loop_gap_materials_summary.txt"),
    )
    enqueue_report = run_enqueue(
        curriculum_path=gap_curriculum_path,
        queue_path=queue_path,
        report_path=workspace_path("autobot", "gap_loop_enqueue_report.json"),
        summary_path=workspace_path("autobot", "gap_loop_enqueue_summary.txt"),
    )
    report = {
        "schema": "sara-autobot-gap-loop-report-v1",
        "passed": bool(dataset_report.get("passed")) and bool(gap_report.get("passed")) and bool(enqueue_report.get("passed")),
        "dataset_accepted_count": int(dataset_report.get("accepted_count", 0) or 0),
        "collection_target_count": int(dataset_report.get("collection_target_count", 0) or 0),
        "gap_material_built_count": int(gap_report.get("built_count", 0) or 0),
        "gap_curriculum_enqueued_count": int(enqueue_report.get("enqueued_count", 0) or 0),
        "queue_pending": int(enqueue_report.get("queue_pending", 0) or 0),
        "evaluation_gaps": list(evaluation_gaps),
        "outputs": {
            "accepted_materials": accepted_path,
            "collection_targets": collection_targets_path,
            "gap_materials": gap_output_path,
            "gap_curriculum": gap_curriculum_path,
            "queue_path": queue_path,
            "dataset_report": dataset_report.get("outputs", {}).get("report", ""),
            "gap_report": gap_report.get("report_path", ""),
            "enqueue_report": enqueue_report.get("report_path", ""),
        },
    }
    report["report_path"] = write_json(report_path, report)
    resolved_summary_path = ensure_parent_directory(summary_path)
    with open(resolved_summary_path, "w", encoding="utf-8") as handle:
        handle.write(summarize_report(report))
    report["summary_path"] = resolved_summary_path
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the autobot gap loop from dataset build through gap enqueue.")
    parser.add_argument("--records-path", default=DEFAULT_RECORDS_PATH)
    parser.add_argument("--candidate-path", default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--rejected-path", default=DEFAULT_REJECTED_PATH)
    parser.add_argument("--accepted-path", default=DEFAULT_ACCEPTED_PATH)
    parser.add_argument("--curriculum-path", default=DEFAULT_CURRICULUM_PATH)
    parser.add_argument("--fixture-request-plan-path", default=DEFAULT_FIXTURE_REQUEST_PLAN_PATH)
    parser.add_argument("--collection-targets-path", default=DEFAULT_COLLECTION_TARGETS_PATH)
    parser.add_argument("--gap-output-path", default=DEFAULT_GAP_OUTPUT_PATH)
    parser.add_argument("--gap-curriculum-path", default=DEFAULT_GAP_CURRICULUM_PATH)
    parser.add_argument("--queue-path", default=DEFAULT_QUEUE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--evaluation-gap", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_gap_loop(
        records_path=args.records_path,
        candidate_path=args.candidate_path,
        rejected_path=args.rejected_path,
        accepted_path=args.accepted_path,
        curriculum_path=args.curriculum_path,
        fixture_request_plan_path=args.fixture_request_plan_path,
        collection_targets_path=args.collection_targets_path,
        gap_output_path=args.gap_output_path,
        gap_curriculum_path=args.gap_curriculum_path,
        queue_path=args.queue_path,
        report_path=args.report_path,
        summary_path=args.summary_path,
        evaluation_gaps=args.evaluation_gap or (),
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "dataset_accepted_count": report["dataset_accepted_count"],
                "gap_material_built_count": report["gap_material_built_count"],
                "gap_curriculum_enqueued_count": report["gap_curriculum_enqueued_count"],
                "report_path": report["report_path"],
                "summary_path": report["summary_path"],
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
