#!/usr/bin/env python3
"""Validate repository-safe research benchmark fixtures."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Set


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path  # noqa: E402


DEFAULT_FIXTURE_PATH = processed_data_path("benchmark_fixtures", "external_validity_cases.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "research_fixture_readiness.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "research_fixture_readiness_summary.txt")
REQUIRED_TASK_TYPES = {"qa", "negative", "partial", "contrastive", "noisy", "adversarial", "delayed"}
ALLOWED_BEHAVIORS = {"retrieve", "abstain"}


def load_fixture_cases(path: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Fixture line {line_number} must be a JSON object.")
            payload["_line_number"] = line_number
            cases.append(payload)
    return cases


def _case_errors(case: Dict[str, Any], seen_ids: Set[str]) -> List[str]:
    errors: List[str] = []
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        errors.append("case_id must be a non-empty string")
    elif case_id in seen_ids:
        errors.append(f"duplicate case_id: {case_id}")
    else:
        seen_ids.add(case_id)

    task_type = case.get("task_type")
    if task_type not in REQUIRED_TASK_TYPES:
        errors.append(f"task_type must be one of {sorted(REQUIRED_TASK_TYPES)}")

    for field in ("query", "document"):
        value = case.get(field)
        if not isinstance(value, str) or len(value.strip()) < 8:
            errors.append(f"{field} must be a meaningful string")

    expected_keywords = case.get("expected_keywords")
    if (
        not isinstance(expected_keywords, list)
        or not expected_keywords
        or not all(isinstance(item, str) and item.strip() for item in expected_keywords)
    ):
        errors.append("expected_keywords must be a non-empty list of strings")

    expected_behavior = case.get("expected_behavior")
    if expected_behavior not in ALLOWED_BEHAVIORS:
        errors.append(f"expected_behavior must be one of {sorted(ALLOWED_BEHAVIORS)}")
    if task_type in {"negative", "partial"} and expected_behavior != "abstain":
        errors.append(f"{task_type} cases must expect abstain behavior")
    if task_type in {"qa", "contrastive", "noisy", "adversarial", "delayed"} and expected_behavior != "retrieve":
        errors.append(f"{task_type} cases must expect retrieve behavior")

    return errors


def build_fixture_readiness_report(cases: Sequence[Dict[str, Any]], fixture_path: str) -> Dict[str, Any]:
    seen_ids: Set[str] = set()
    resolved_fixture_path = os.path.abspath(fixture_path)
    fixture_root = os.path.abspath(processed_data_path("benchmark_fixtures"))
    task_types = sorted({str(case.get("task_type")) for case in cases if isinstance(case.get("task_type"), str)})
    case_results: List[Dict[str, Any]] = []
    for case in cases:
        errors = _case_errors(case, seen_ids)
        case_results.append(
            {
                "case_id": case.get("case_id"),
                "line_number": case.get("_line_number"),
                "task_type": case.get("task_type"),
                "expected_behavior": case.get("expected_behavior"),
                "passed": not errors,
                "errors": errors,
            }
        )

    missing_task_types = sorted(REQUIRED_TASK_TYPES.difference(task_types))
    behavior_counts = {
        behavior: sum(1 for case in cases if case.get("expected_behavior") == behavior)
        for behavior in sorted(ALLOWED_BEHAVIORS)
    }
    coverage = {
        "has_repository_safe_fixture": resolved_fixture_path.startswith(fixture_root + os.sep),
        "has_noisy_case": "noisy" in task_types,
        "has_adversarial_case": "adversarial" in task_types,
        "has_delayed_recall_case": "delayed" in task_types,
        "has_abstention_cases": behavior_counts.get("abstain", 0) >= 2,
        "has_retrieval_cases": behavior_counts.get("retrieve", 0) >= 4,
    }
    passed = (
        len(cases) >= 8
        and not missing_task_types
        and all(item["passed"] for item in case_results)
        and all(coverage.values())
    )
    return {
        "schema": "sara-research-fixture-readiness-v1",
        "fixture_path": resolved_fixture_path,
        "case_count": len(cases),
        "task_types": task_types,
        "missing_task_types": missing_task_types,
        "behavior_counts": behavior_counts,
        "coverage": coverage,
        "case_results": case_results,
        "passed": passed,
    }


def write_outputs(report: Dict[str, Any], report_path: str, summary_path: str) -> None:
    resolved_report_path = ensure_parent_directory(report_path)
    with open(resolved_report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    resolved_summary_path = ensure_parent_directory(summary_path)
    lines = [
        f"Research fixture readiness: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Cases: {report.get('case_count')}",
        f"Task types: {', '.join(report.get('task_types', []))}",
        f"Missing task types: {', '.join(report.get('missing_task_types', [])) or 'none'}",
    ]
    for key, value in sorted(report.get("coverage", {}).items()):
        lines.append(f"- {key}: {value}")
    with open(resolved_summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate research benchmark fixture coverage.")
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cases = load_fixture_cases(args.fixture_path)
    report = build_fixture_readiness_report(cases, args.fixture_path)
    write_outputs(report, args.report_path, args.summary_path)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "case_count": report["case_count"],
                "report_path": os.path.abspath(args.report_path),
                "summary_path": os.path.abspath(args.summary_path),
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
