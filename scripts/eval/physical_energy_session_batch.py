#!/usr/bin/env python3
"""Build or execute a thin batch plan for physical-energy pair sessions."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path  # noqa: E402


DEFAULT_SESSION_PLAN_PATH = workspace_path("evaluation", "energy_measurement_session_plan.json")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "physical_energy_session_batch.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "physical_energy_session_batch.txt")


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _load_session_plan(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Session plan must be a JSON object.")
    return payload


def _replace_replicate_placeholders(value: str, replicate_index: int) -> str:
    return str(value).replace("<replicate>", str(replicate_index))


def build_physical_energy_session_batch(
    session_plan: Mapping[str, Any],
) -> Dict[str, Any]:
    planned_runs = (
        session_plan.get("planned_runs", [])
        if isinstance(session_plan.get("planned_runs"), list)
        else []
    )
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in planned_runs:
        if not isinstance(item, Mapping):
            continue
        category = str(item.get("category", "") or "").strip()
        task = str(item.get("task", "") or "").strip()
        pair_command_template = str(item.get("pair_command_template", "") or "").strip()
        pair_id_template = str(item.get("pair_id_template", "") or "").strip()
        if not category or not task or not pair_command_template or not pair_id_template:
            continue
        key = (category, task)
        entry = grouped.setdefault(
            key,
            {
                "category": category,
                "task": task,
                "priority": str(item.get("priority", "") or ""),
                "replicate_count": 0,
                "pair_command_template": pair_command_template,
                "pair_id_template": pair_id_template,
                "manifest_path_template": str(item.get("manifest_path_template", "") or ""),
                "trace_path_template": str(item.get("trace_path_template", "") or ""),
                "report_path_template": str(item.get("report_path_template", "") or ""),
                "summary_path_template": str(item.get("summary_path_template", "") or ""),
                "meter_template_path": str(item.get("meter_template_path", "") or ""),
                "systems": set(),
            },
        )
        entry["replicate_count"] = max(
            int(entry.get("replicate_count", 0) or 0),
            _safe_int(item.get("replicate_count")) or 1,
        )
        system = str(item.get("system", "") or "").strip().lower()
        if system:
            entry["systems"].add(system)

    batch_runs: List[Dict[str, Any]] = []
    for _, item in sorted(grouped.items()):
        replicate_count = max(int(item.get("replicate_count", 0) or 0), 1)
        for replicate_index in range(1, replicate_count + 1):
            batch_runs.append(
                {
                    "category": str(item["category"]),
                    "task": str(item["task"]),
                    "priority": str(item.get("priority", "") or ""),
                    "replicate_index": int(replicate_index),
                    "systems": sorted(str(system) for system in item.get("systems", set())),
                    "pair_id": _replace_replicate_placeholders(
                        str(item["pair_id_template"]),
                        replicate_index,
                    ),
                    "command": _replace_replicate_placeholders(
                        str(item["pair_command_template"]),
                        replicate_index,
                    ),
                    "manifest_path": _replace_replicate_placeholders(
                        str(item.get("manifest_path_template", "") or ""),
                        replicate_index,
                    ),
                    "trace_path": _replace_replicate_placeholders(
                        str(item.get("trace_path_template", "") or ""),
                        replicate_index,
                    ),
                    "report_path": _replace_replicate_placeholders(
                        str(item.get("report_path_template", "") or ""),
                        replicate_index,
                    ),
                    "summary_path": _replace_replicate_placeholders(
                        str(item.get("summary_path_template", "") or ""),
                        replicate_index,
                    ),
                    "meter_template_path": _replace_replicate_placeholders(
                        str(item.get("meter_template_path", "") or ""),
                        replicate_index,
                    ),
                }
            )
    return {
        "schema": "sara-physical-energy-session-batch-v1",
        "session_id": str(session_plan.get("session_id", "") or ""),
        "status": "ready" if batch_runs else "empty",
        "planned_pair_count": len(batch_runs),
        "batch_runs": batch_runs,
    }


def execute_dry_run_pairs(batch_plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    runs = batch_plan.get("batch_runs", []) if isinstance(batch_plan.get("batch_runs"), list) else []
    for item in runs:
        if not isinstance(item, Mapping):
            continue
        command_text = str(item.get("command", "") or "").strip()
        if not command_text:
            continue
        command = shlex.split(command_text)
        if "--dry-run" not in command:
            command.append("--dry-run")
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        results.append(
            {
                "pair_id": str(item.get("pair_id", "") or ""),
                "replicate_index": _safe_int(item.get("replicate_index")),
                "returncode": int(result.returncode),
                "status": "passed" if result.returncode == 0 else "failed",
                "meter_template_path": str(item.get("meter_template_path", "") or ""),
            }
        )
    return results


def format_summary(report: Mapping[str, Any]) -> str:
    runs = report.get("batch_runs", []) if isinstance(report.get("batch_runs"), list) else []
    execution = report.get("execution_results", []) if isinstance(report.get("execution_results"), list) else []
    lines = [
        "# SARA Physical Energy Session Batch",
        f"- session_id: {report.get('session_id', '')}",
        f"- status: {report.get('status', '')}",
        f"- planned_pair_count: {_safe_int(report.get('planned_pair_count'))}",
        f"- executed_pair_count: {len(execution)}",
    ]
    lines.append("Planned Pairs:")
    if runs:
        for item in runs:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "- "
                f"category={item.get('category', '')}, "
                f"task={item.get('task', '')}, "
                f"replicate_index={_safe_int(item.get('replicate_index'))}, "
                f"pair_id={item.get('pair_id', '')}"
            )
            lines.append(f"  command: {item.get('command', '')}")
            lines.append(f"  meter_template_path: {item.get('meter_template_path', '')}")
    else:
        lines.append("- none")
    lines.append("Execution Results:")
    if execution:
        for item in execution:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "- "
                f"pair_id={item.get('pair_id', '')}, "
                f"replicate_index={_safe_int(item.get('replicate_index'))}, "
                f"status={item.get('status', '')}, "
                f"returncode={_safe_int(item.get('returncode'))}"
            )
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a thin physical-energy session batch plan.")
    parser.add_argument("--session-plan-path", default=DEFAULT_SESSION_PLAN_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--execute-dry-run-pairs", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    session_plan = _load_session_plan(args.session_plan_path)
    report = build_physical_energy_session_batch(session_plan)
    execution_results: List[Dict[str, Any]] = []
    if args.execute_dry_run_pairs:
        execution_results = execute_dry_run_pairs(report)
    full_report = dict(report)
    full_report["execution_results"] = execution_results
    report_path = ensure_parent_directory(args.report_path)
    summary_path = ensure_parent_directory(args.summary_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(full_report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_summary(full_report))
    print(json.dumps(full_report, indent=2, ensure_ascii=False, sort_keys=True))
    all_passed = all(
        str(item.get("status", "") or "") == "passed"
        for item in execution_results
        if isinstance(item, Mapping)
    )
    return 0 if (not args.execute_dry_run_pairs or all_passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
