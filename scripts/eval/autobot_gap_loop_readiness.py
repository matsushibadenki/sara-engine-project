#!/usr/bin/env python3
"""Evaluate whether the managed autobot gap loop is producing usable repair evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Mapping, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_LOOP_REPORT_PATH = workspace_path("autobot", "gap_loop_report.json")
DEFAULT_COLLECTION_TARGETS_PATH = workspace_path("autobot", "dataset_builder_collection_targets.json")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "autobot_gap_loop_readiness.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "autobot_gap_loop_readiness_summary.txt")


def read_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_json(path: str, payload: Mapping[str, Any]) -> str:
    resolved = ensure_parent_directory(path)
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return resolved


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _resolve_output_path(loop_report: Optional[Dict[str, Any]], key: str) -> str:
    if not isinstance(loop_report, dict):
        return ""
    outputs = loop_report.get("outputs", {})
    if not isinstance(outputs, dict):
        return ""
    return str(outputs.get(key, "") or "")


def _count_requested_slots(targets_payload: Optional[Dict[str, Any]]) -> int:
    if not isinstance(targets_payload, dict):
        return 0
    targets = targets_payload.get("targets", [])
    if not isinstance(targets, list):
        return 0
    total = 0
    for item in targets:
        if not isinstance(item, dict):
            continue
        total += len([value for value in item.get("missing_material_types", []) if str(value)])
    return total


def _check(condition: bool, value: Any, detail: str = "") -> Dict[str, Any]:
    return {
        "passed": bool(condition),
        "value": value,
        "detail": detail,
    }


def build_report(
    *,
    loop_report: Optional[Dict[str, Any]],
    dataset_report: Optional[Dict[str, Any]],
    gap_report: Optional[Dict[str, Any]],
    enqueue_report: Optional[Dict[str, Any]],
    collection_targets: Optional[Dict[str, Any]],
    min_accepted_count: int,
    min_gap_build_coverage: float,
) -> Dict[str, Any]:
    target_count = 0 if not isinstance(collection_targets, dict) else _safe_int(collection_targets.get("target_count"))
    requested_slot_count = _count_requested_slots(collection_targets)
    accepted_count = 0 if not isinstance(dataset_report, dict) else _safe_int(dataset_report.get("accepted_count"))
    built_count = 0 if not isinstance(gap_report, dict) else _safe_int(gap_report.get("built_count"))
    skipped_count = 0 if not isinstance(gap_report, dict) else _safe_int(gap_report.get("skipped_count"))
    enqueued_count = 0 if not isinstance(enqueue_report, dict) else _safe_int(enqueue_report.get("enqueued_count"))
    queue_pending = 0 if not isinstance(enqueue_report, dict) else _safe_int(enqueue_report.get("queue_pending"))
    build_coverage = 1.0 if requested_slot_count <= 0 else built_count / float(requested_slot_count)
    enqueue_coverage = 1.0 if built_count <= 0 else enqueued_count / float(built_count)
    skipped_ratio = 0.0 if requested_slot_count <= 0 else skipped_count / float(requested_slot_count)
    gap_curriculum_distribution = {}
    if isinstance(gap_report, dict) and isinstance(gap_report.get("curriculum_distribution"), dict):
        gap_curriculum_distribution = dict(gap_report.get("curriculum_distribution", {}))
    repair_count = _safe_int(gap_curriculum_distribution.get("repair"))
    replay_count = _safe_int(gap_curriculum_distribution.get("replay"))
    total_curriculum = sum(_safe_int(value) for value in gap_curriculum_distribution.values())
    repair_share = 0.0 if total_curriculum <= 0 else repair_count / float(total_curriculum)
    replay_share = 0.0 if total_curriculum <= 0 else replay_count / float(total_curriculum)
    checks = {
        "loop_report_present": _check(isinstance(loop_report, dict), bool(loop_report)),
        "dataset_report_present": _check(isinstance(dataset_report, dict), bool(dataset_report)),
        "gap_report_present": _check(isinstance(gap_report, dict), bool(gap_report)),
        "enqueue_report_present": _check(isinstance(enqueue_report, dict), bool(enqueue_report)),
        "collection_targets_present": _check(isinstance(collection_targets, dict), bool(collection_targets)),
        "loop_passed": _check(bool(loop_report and loop_report.get("passed")), bool(loop_report and loop_report.get("passed"))),
        "accepted_materials_ready": _check(
            accepted_count >= int(min_accepted_count),
            accepted_count,
            f"min_accepted_count={int(min_accepted_count)}",
        ),
        "target_generation_ready": _check(target_count >= 0, target_count),
        "gap_material_coverage_ready": _check(
            requested_slot_count <= 0 or build_coverage >= float(min_gap_build_coverage),
            round(build_coverage, 6),
            f"min_gap_build_coverage={float(min_gap_build_coverage):.3f}",
        ),
        "gap_enqueue_ready": _check(
            built_count <= 0 or enqueued_count > 0,
            enqueued_count,
            "gap materials should reach the managed training queue",
        ),
        "repair_curriculum_present": _check(
            built_count <= 0 or (repair_count + replay_count) > 0,
            {"repair": repair_count, "replay": replay_count},
        ),
    }
    passed = all(bool(item.get("passed")) for item in checks.values())
    return {
        "schema": "sara-autobot-gap-loop-readiness-v1",
        "passed": passed,
        "metrics": {
            "accepted_count": accepted_count,
            "collection_target_count": target_count,
            "requested_slot_count": requested_slot_count,
            "gap_material_built_count": built_count,
            "gap_material_skipped_count": skipped_count,
            "gap_curriculum_enqueued_count": enqueued_count,
            "queue_pending": queue_pending,
            "gap_build_coverage": build_coverage,
            "gap_enqueue_coverage": enqueue_coverage,
            "gap_skip_ratio": skipped_ratio,
            "repair_curriculum_share": repair_share,
            "replay_curriculum_share": replay_share,
        },
        "checks": checks,
        "input_paths": {},
        "policy_notes": [
            "Readiness does not claim benchmark quality gains by itself; it only verifies that source-backed gap requests become managed repair or replay curriculum.",
            "Requested-slot coverage counts missing material slots, not abstract request objects, so counterexample and transcript needs remain separately visible.",
            "This report is Phase 7 evidence about autonomous data preparation, not Phase 6 physical energy evidence or Phase 8 ANN baseline evidence.",
        ],
    }


def summarize_report(report: Mapping[str, Any]) -> str:
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    checks = report.get("checks", {}) if isinstance(report.get("checks"), dict) else {}
    lines = [
        f"Autobot gap loop readiness: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Accepted materials: {metrics.get('accepted_count')}",
        f"Collection targets: {metrics.get('collection_target_count')}",
        f"Requested slots: {metrics.get('requested_slot_count')}",
        f"Gap materials built: {metrics.get('gap_material_built_count')}",
        f"Gap materials skipped: {metrics.get('gap_material_skipped_count')}",
        f"Gap curriculum enqueued: {metrics.get('gap_curriculum_enqueued_count')}",
        f"Gap build coverage: {float(metrics.get('gap_build_coverage', 0.0) or 0.0):.3f}",
        f"Gap enqueue coverage: {float(metrics.get('gap_enqueue_coverage', 0.0) or 0.0):.3f}",
        "Checks:",
    ]
    for name, payload in sorted(checks.items()):
        if not isinstance(payload, dict):
            continue
        lines.append(f"- {name}: {bool(payload.get('passed'))} ({payload.get('value')})")
    return "\n".join(lines) + "\n"


def run_readiness(
    *,
    loop_report_path: str = DEFAULT_LOOP_REPORT_PATH,
    collection_targets_path: str = DEFAULT_COLLECTION_TARGETS_PATH,
    dataset_report_path: str = "",
    gap_report_path: str = "",
    enqueue_report_path: str = "",
    report_path: str = DEFAULT_REPORT_PATH,
    summary_path: str = DEFAULT_SUMMARY_PATH,
    min_accepted_count: int = 4,
    min_gap_build_coverage: float = 0.5,
) -> Dict[str, Any]:
    loop_report = read_json(loop_report_path)
    if not dataset_report_path:
        dataset_report_path = _resolve_output_path(loop_report, "dataset_report")
    if not gap_report_path:
        gap_report_path = _resolve_output_path(loop_report, "gap_report")
    if not enqueue_report_path:
        enqueue_report_path = _resolve_output_path(loop_report, "enqueue_report")
    dataset_report = read_json(dataset_report_path)
    gap_report = read_json(gap_report_path)
    enqueue_report = read_json(enqueue_report_path)
    collection_targets = read_json(collection_targets_path)
    report = build_report(
        loop_report=loop_report,
        dataset_report=dataset_report,
        gap_report=gap_report,
        enqueue_report=enqueue_report,
        collection_targets=collection_targets,
        min_accepted_count=min_accepted_count,
        min_gap_build_coverage=min_gap_build_coverage,
    )
    report["input_paths"] = {
        "loop_report": os.path.abspath(loop_report_path),
        "collection_targets": os.path.abspath(collection_targets_path),
        "dataset_report": os.path.abspath(dataset_report_path) if dataset_report_path else "",
        "gap_report": os.path.abspath(gap_report_path) if gap_report_path else "",
        "enqueue_report": os.path.abspath(enqueue_report_path) if enqueue_report_path else "",
    }
    report["report_path"] = write_json(report_path, report)
    resolved_summary_path = ensure_parent_directory(summary_path)
    with open(resolved_summary_path, "w", encoding="utf-8") as handle:
        handle.write(summarize_report(report))
    report["summary_path"] = resolved_summary_path
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate managed autobot gap-loop readiness.")
    parser.add_argument("--loop-report-path", default=DEFAULT_LOOP_REPORT_PATH)
    parser.add_argument("--collection-targets-path", default=DEFAULT_COLLECTION_TARGETS_PATH)
    parser.add_argument("--dataset-report-path", default="")
    parser.add_argument("--gap-report-path", default="")
    parser.add_argument("--enqueue-report-path", default="")
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--min-accepted-count", type=int, default=4)
    parser.add_argument("--min-gap-build-coverage", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_readiness(
        loop_report_path=args.loop_report_path,
        collection_targets_path=args.collection_targets_path,
        dataset_report_path=args.dataset_report_path,
        gap_report_path=args.gap_report_path,
        enqueue_report_path=args.enqueue_report_path,
        report_path=args.report_path,
        summary_path=args.summary_path,
        min_accepted_count=args.min_accepted_count,
        min_gap_build_coverage=args.min_gap_build_coverage,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "accepted_count": report["metrics"]["accepted_count"],
                "gap_material_built_count": report["metrics"]["gap_material_built_count"],
                "gap_curriculum_enqueued_count": report["metrics"]["gap_curriculum_enqueued_count"],
                "gap_build_coverage": report["metrics"]["gap_build_coverage"],
                "report_path": report["report_path"],
                "summary_path": report["summary_path"],
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
