#!/usr/bin/env python3
"""Summarize progress for one physical-energy measurement session."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_DIR = os.path.dirname(__file__)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for path in (SCRIPT_DIR, SRC_PATH):
    if path not in sys.path:
        sys.path.insert(0, path)

from sara_engine.utils.project_paths import ensure_parent_directory, raw_data_path, workspace_path  # noqa: E402
from energy_measurement_readiness import (  # noqa: E402
    _internal_maintenance_reference_summary,
    _load_optional_json,
    _pair_fairness_errors,
    _pair_key,
    _safe_float,
    _safe_int,
    _validate_measurement,
    load_measurements,
)


DEFAULT_BATCH_PATH = workspace_path("evaluation", "physical_energy_session_batch.json")
DEFAULT_MEASUREMENT_PATH = raw_data_path("energy_measurements.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "physical_energy_session_progress.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "physical_energy_session_progress.txt")
DEFAULT_INTERNAL_MAINTENANCE_REPORT_PATH = workspace_path(
    "evaluation", "internal_maintenance_efficiency_benchmark.json"
)


def _load_batch(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Physical energy session batch must be a JSON object.")
    return payload


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[Tuple[str, str, int], Dict[str, Mapping[str, Any]]], List[Dict[str, Any]]]:
    pair_index: Dict[Tuple[str, str, int], Dict[str, Mapping[str, Any]]] = {}
    row_errors: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        errors = _validate_measurement(row)
        if errors:
            row_errors.append({"index": index, "errors": errors})
            continue
        system = str(row.get("system", "")).lower()
        if system not in {"sara", "ann"}:
            continue
        pair_index.setdefault(_pair_key(row), {})[system] = row
    return pair_index, row_errors


def build_physical_energy_session_progress(
    batch_report: Mapping[str, Any],
    measurements: Sequence[Mapping[str, Any]],
    *,
    internal_maintenance_report: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    runs = batch_report.get("batch_runs", []) if isinstance(batch_report.get("batch_runs"), list) else []
    pair_index, row_errors = _index_rows(measurements)
    expected_keys: set[Tuple[str, str, int]] = set()
    pair_statuses: List[Dict[str, Any]] = []

    for item in runs:
        if not isinstance(item, Mapping):
            continue
        task = str(item.get("task", "") or "").strip()
        pair_id = str(item.get("pair_id", "") or "").strip()
        replicate_index = _safe_int(item.get("replicate_index"))
        key = (task, pair_id, replicate_index)
        if not task or not pair_id or replicate_index <= 0:
            continue
        expected_keys.add(key)
        systems = pair_index.get(key, {})
        present_systems = sorted(systems.keys())
        status = "missing_pair"
        errors: List[str] = []
        ann_to_sara_ratio = 0.0
        if set(present_systems) == {"ann", "sara"}:
            sara_row = systems["sara"]
            ann_row = systems["ann"]
            errors = _pair_fairness_errors(sara_row, ann_row)
            if errors:
                status = "invalid_pair"
            else:
                status = "complete_valid_pair"
                sara_jps = _safe_float(sara_row.get("joules")) / max(
                    _safe_int(sara_row.get("success_count")),
                    1,
                )
                ann_jps = _safe_float(ann_row.get("joules")) / max(
                    _safe_int(ann_row.get("success_count")),
                    1,
                )
                ann_to_sara_ratio = ann_jps / max(sara_jps, 1e-9)
        elif present_systems:
            status = "partial_pair"
            missing = sorted({"sara", "ann"} - set(present_systems))
            errors = [f"missing_system:{name}" for name in missing]
        pair_statuses.append(
            {
                "category": str(item.get("category", "") or ""),
                "task": task,
                "priority": str(item.get("priority", "") or ""),
                "pair_id": pair_id,
                "replicate_index": replicate_index,
                "status": status,
                "present_systems": present_systems,
                "errors": errors,
                "ann_to_sara_joule_efficiency_ratio": float(ann_to_sara_ratio),
                "meter_template_path": str(item.get("meter_template_path", "") or ""),
                "report_path": str(item.get("report_path", "") or ""),
                "summary_path": str(item.get("summary_path", "") or ""),
            }
        )

    orphan_pairs: List[Dict[str, Any]] = []
    for key, systems in sorted(pair_index.items()):
        if key in expected_keys:
            continue
        orphan_pairs.append(
            {
                "task": key[0],
                "pair_id": key[1],
                "replicate_index": key[2],
                "present_systems": sorted(systems.keys()),
            }
        )

    task_progress: Dict[str, Dict[str, int]] = {}
    for item in pair_statuses:
        task_bucket = task_progress.setdefault(
            str(item["task"]),
            {
                "planned_pair_count": 0,
                "complete_valid_pair_count": 0,
                "invalid_pair_count": 0,
                "partial_pair_count": 0,
                "missing_pair_count": 0,
            },
        )
        task_bucket["planned_pair_count"] += 1
        task_bucket[f"{item['status']}_count"] += 1

    complete_valid_pair_count = sum(
        1 for item in pair_statuses if str(item.get("status", "")) == "complete_valid_pair"
    )
    invalid_pair_count = sum(
        1 for item in pair_statuses if str(item.get("status", "")) == "invalid_pair"
    )
    partial_pair_count = sum(
        1 for item in pair_statuses if str(item.get("status", "")) == "partial_pair"
    )
    missing_pair_count = sum(
        1 for item in pair_statuses if str(item.get("status", "")) == "missing_pair"
    )
    planned_pair_count = len(pair_statuses)
    return {
        "schema": "sara-physical-energy-session-progress-v1",
        "session_id": str(batch_report.get("session_id", "") or ""),
        "status": (
            "complete"
            if planned_pair_count > 0 and complete_valid_pair_count == planned_pair_count
            else ("in_progress" if complete_valid_pair_count or partial_pair_count else "pending")
        ),
        "planned_pair_count": planned_pair_count,
        "complete_valid_pair_count": complete_valid_pair_count,
        "invalid_pair_count": invalid_pair_count,
        "partial_pair_count": partial_pair_count,
        "missing_pair_count": missing_pair_count,
        "orphan_pair_count": len(orphan_pairs),
        "invalid_measurement_row_count": len(row_errors),
        "task_progress": task_progress,
        "pair_statuses": pair_statuses,
        "orphan_pairs": orphan_pairs,
        "invalid_measurement_rows": row_errors,
        "internal_maintenance_reference": _internal_maintenance_reference_summary(
            internal_maintenance_report
        ),
    }


def format_summary(report: Mapping[str, Any]) -> str:
    task_progress = report.get("task_progress", {}) if isinstance(report.get("task_progress"), Mapping) else {}
    pair_statuses = report.get("pair_statuses", []) if isinstance(report.get("pair_statuses"), list) else []
    orphan_pairs = report.get("orphan_pairs", []) if isinstance(report.get("orphan_pairs"), list) else []
    internal_maintenance_reference = (
        report.get("internal_maintenance_reference", {})
        if isinstance(report.get("internal_maintenance_reference"), Mapping)
        else {}
    )
    lines = [
        "# SARA Physical Energy Session Progress",
        f"- session_id: {report.get('session_id', '')}",
        f"- status: {report.get('status', '')}",
        f"- planned_pair_count: {_safe_int(report.get('planned_pair_count'))}",
        f"- complete_valid_pair_count: {_safe_int(report.get('complete_valid_pair_count'))}",
        f"- invalid_pair_count: {_safe_int(report.get('invalid_pair_count'))}",
        f"- partial_pair_count: {_safe_int(report.get('partial_pair_count'))}",
        f"- missing_pair_count: {_safe_int(report.get('missing_pair_count'))}",
        f"- orphan_pair_count: {_safe_int(report.get('orphan_pair_count'))}",
        f"- invalid_measurement_row_count: {_safe_int(report.get('invalid_measurement_row_count'))}",
        "Task Progress:",
    ]
    if task_progress:
        for task, metrics in sorted(task_progress.items()):
            if not isinstance(metrics, Mapping):
                continue
            lines.append(
                "- "
                f"task={task}, "
                f"planned={_safe_int(metrics.get('planned_pair_count'))}, "
                f"complete={_safe_int(metrics.get('complete_valid_pair_count'))}, "
                f"partial={_safe_int(metrics.get('partial_pair_count'))}, "
                f"invalid={_safe_int(metrics.get('invalid_pair_count'))}, "
                f"missing={_safe_int(metrics.get('missing_pair_count'))}"
            )
    else:
        lines.append("- none")
    lines.append("Pair Statuses:")
    if pair_statuses:
        for item in pair_statuses:
            if not isinstance(item, Mapping):
                continue
            ratio = _safe_float(item.get("ann_to_sara_joule_efficiency_ratio"))
            ratio_text = f"{ratio:.3f}" if ratio > 0.0 else "n/a"
            lines.append(
                "- "
                f"task={item.get('task', '')}, "
                f"pair_id={item.get('pair_id', '')}, "
                f"replicate_index={_safe_int(item.get('replicate_index'))}, "
                f"status={item.get('status', '')}, "
                f"ratio={ratio_text}"
            )
    else:
        lines.append("- none")
    lines.append("Orphan Pairs:")
    if orphan_pairs:
        for item in orphan_pairs:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "- "
                f"task={item.get('task', '')}, "
                f"pair_id={item.get('pair_id', '')}, "
                f"replicate_index={_safe_int(item.get('replicate_index'))}, "
                f"systems={','.join(str(name) for name in item.get('present_systems', []))}"
            )
    else:
        lines.append("- none")
    if internal_maintenance_reference:
        lines.append("Internal Maintenance Reference:")
        lines.append(
            "- "
            f"available={bool(internal_maintenance_reference.get('available', False))}, "
            f"passed={bool(internal_maintenance_reference.get('passed', False))}, "
            f"event_cost_per_selected={_safe_float(internal_maintenance_reference.get('maintenance_event_cost_per_selected')):.3f}, "
            f"continuity={_safe_float(internal_maintenance_reference.get('maintenance_self_state_continuity_observed')):.3f}"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize physical-energy session progress.")
    parser.add_argument("--batch-report-path", default=DEFAULT_BATCH_PATH)
    parser.add_argument("--measurement-path", default=DEFAULT_MEASUREMENT_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--internal-maintenance-report-path", default=DEFAULT_INTERNAL_MAINTENANCE_REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    batch_report = _load_batch(args.batch_report_path)
    measurements = load_measurements(args.measurement_path)
    report = build_physical_energy_session_progress(
        batch_report,
        measurements,
        internal_maintenance_report=_load_optional_json(args.internal_maintenance_report_path),
    )
    report_path = ensure_parent_directory(args.report_path)
    summary_path = ensure_parent_directory(args.summary_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_summary(report))
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
