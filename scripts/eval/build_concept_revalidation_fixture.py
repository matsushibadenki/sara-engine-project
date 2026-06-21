#!/usr/bin/env python3
"""Build source-aware concept revalidation fixture cases from latent manifest rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from sara_engine.memory.concept_revalidation_fixture import (  # noqa: E402
    build_concept_revalidation_cases,
    summarize_case_types,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)


DEFAULT_MANIFEST_PATH = processed_data_path("autobot", "latent_manifest.jsonl")
DEFAULT_FIXTURE_PATH = processed_data_path("benchmark_fixtures", "concept_revalidation_cases.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "concept_revalidation_fixture_builder.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "concept_revalidation_fixture_builder_summary.txt")


def build_next_actions(case_type_counts: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_counts = case_type_counts if isinstance(case_type_counts, dict) else {}

    def _count(key: str) -> int:
        value = raw_counts.get(key, 0)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    blocked_source_diversity = _count("blocked_source_diversity")
    blocked_counterexample_pressure = _count("blocked_counterexample_pressure")
    blocked_attempt_budget = _count("blocked_attempt_budget")
    recoverable_revision_conflict = _count("recoverable_revision_conflict")

    actions: List[Dict[str, Any]] = []
    if blocked_source_diversity > 0:
        actions.append(
            {
                "priority": 5,
                "reason": "source_diversity",
                "action": "collect_additional_distinct_sources",
                "case_type": "blocked_source_diversity",
                "case_count": blocked_source_diversity,
            }
        )
    if blocked_counterexample_pressure > 0:
        actions.append(
            {
                "priority": 4,
                "reason": "counterexample_pressure",
                "action": "add_negative_and_contrastive_materials",
                "case_type": "blocked_counterexample_pressure",
                "case_count": blocked_counterexample_pressure,
            }
        )
    if blocked_attempt_budget > 0:
        actions.append(
            {
                "priority": 3,
                "reason": "attempt_budget",
                "action": "manual_review_high_stall_candidates",
                "case_type": "blocked_attempt_budget",
                "case_count": blocked_attempt_budget,
            }
        )
    if recoverable_revision_conflict > 0:
        actions.append(
            {
                "priority": 2,
                "reason": "revision_conflict",
                "action": "resolve_source_revision_conflicts",
                "case_type": "recoverable_revision_conflict",
                "case_count": recoverable_revision_conflict,
            }
        )
    if not actions:
        actions.append(
            {
                "priority": 1,
                "reason": "coverage",
                "action": "scale_revalidation_case_coverage",
                "case_type": "all",
                "case_count": 0,
            }
        )
    return actions


def summarize_manifest_materials(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        material_type = str(row.get("material_type", "unknown") or "unknown")
        counts[material_type] = counts.get(material_type, 0) + 1
    return dict(sorted(counts.items()))


def build_expansion_plan(
    *,
    rows: Sequence[Dict[str, Any]],
    next_actions: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    material_counts = summarize_manifest_materials(rows)
    plan: List[Dict[str, Any]] = []
    for item in next_actions:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "") or "")
        case_type = str(item.get("case_type", "") or "")
        if action == "collect_additional_distinct_sources":
            preferred_material_types = ["source_claim", "qa_pair", "transcript_segment"]
            guidance = "Increase distinct source_ref coverage for repeated relation candidates."
        elif action == "add_negative_and_contrastive_materials":
            preferred_material_types = ["contrastive_pair", "counterexample", "qa_pair"]
            guidance = "Add negative and contrastive rows that can challenge over-generalized relations."
        elif action == "manual_review_high_stall_candidates":
            preferred_material_types = ["repair_note", "source_claim", "qa_pair"]
            guidance = "Prepare review-oriented support rows so stalled candidates can be rebuilt."
        elif action == "resolve_source_revision_conflicts":
            preferred_material_types = ["source_claim", "revision_note", "qa_pair"]
            guidance = "Collect reconciled source revisions for concepts blocked by conflicting source snapshots."
        else:
            preferred_material_types = ["qa_pair", "source_claim"]
            guidance = "Expand general coverage for underrepresented concept revalidation patterns."

        available_material_types = {
            key: material_counts.get(key, 0)
            for key in preferred_material_types
            if material_counts.get(key, 0) > 0
        }
        missing_material_types = [
            key for key in preferred_material_types if material_counts.get(key, 0) == 0
        ]
        plan.append(
            {
                "action": action,
                "case_type": case_type,
                "priority": item.get("priority"),
                "target_case_count": item.get("case_count"),
                "preferred_material_types": preferred_material_types,
                "available_material_types": available_material_types,
                "missing_material_types": missing_material_types,
                "guidance": guidance,
            }
        )
    return plan


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> str:
    resolved = ensure_parent_directory(path)
    with open(resolved, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return resolved


def build_report(
    *,
    manifest_path: str,
    fixture_path: str,
    rows: Sequence[Dict[str, Any]],
    manifest_rows: Sequence[Dict[str, Any]],
    max_cases: int,
) -> Dict[str, Any]:
    case_types = summarize_case_types(rows)
    next_actions = build_next_actions(case_types)
    material_type_counts = summarize_manifest_materials(manifest_rows)
    expansion_plan = build_expansion_plan(rows=manifest_rows, next_actions=next_actions)
    expected_outcomes = sorted({str(row.get("expected_outcome", "")) for row in rows if str(row.get("expected_outcome", ""))})
    passed = bool(rows) and "admit" in expected_outcomes and "blocked" in expected_outcomes
    return {
        "schema": "sara-concept-revalidation-fixture-builder-report-v1",
        "passed": passed,
        "observed_only": True,
        "manifest_path": os.path.abspath(manifest_path),
        "fixture_path": os.path.abspath(fixture_path),
        "case_count": len(rows),
        "max_cases": int(max_cases),
        "case_type_counts": case_types,
        "manifest_material_type_counts": material_type_counts,
        "next_actions": next_actions,
        "expansion_plan": expansion_plan,
        "expected_outcomes": expected_outcomes,
        "policy_notes": [
            "Fixture cases are source-aware and derived from managed latent manifest rows.",
            "Outputs stay under data/processed/benchmark_fixtures and workspace/evaluation.",
            "Cases include both recoverable and intentionally blocked concept revalidation patterns.",
        ],
    }


def summarize_report(report: Dict[str, Any]) -> str:
    lines = [
        f"Concept revalidation fixture builder: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Observed only: {report.get('observed_only')}",
        f"Cases: {report.get('case_count')}",
        f"Max cases: {report.get('max_cases')}",
        "Case types:",
    ]
    for key, value in sorted(report.get("case_type_counts", {}).items()):
        lines.append(f"- {key}: {value}")
    material_counts = report.get("manifest_material_type_counts", {})
    if isinstance(material_counts, dict) and material_counts:
        lines.append("Manifest material types:")
        for key, value in sorted(material_counts.items()):
            lines.append(f"- {key}: {value}")
    next_actions = report.get("next_actions", [])
    if isinstance(next_actions, list) and next_actions:
        lines.append("Next actions:")
        for item in next_actions:
            if not isinstance(item, dict):
                continue
            lines.append(
                "- "
                f"{item.get('action', '')} "
                f"(reason={item.get('reason', '')}, priority={item.get('priority', '')}, "
                f"case_type={item.get('case_type', '')}, case_count={item.get('case_count', '')})"
            )
    expansion_plan = report.get("expansion_plan", [])
    if isinstance(expansion_plan, list) and expansion_plan:
        lines.append("Expansion plan:")
        for item in expansion_plan:
            if not isinstance(item, dict):
                continue
            lines.append(
                "- "
                f"{item.get('action', '')} "
                f"(preferred={','.join(item.get('preferred_material_types', []))}, "
                f"missing={','.join(item.get('missing_material_types', []))})"
            )
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build concept revalidation fixture cases.")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--fixture-path", default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--max-cases", type=int, default=4)
    args = parser.parse_args(argv)

    materials = read_jsonl(args.manifest_path)
    cases = build_concept_revalidation_cases(materials, max_cases=args.max_cases)
    resolved_fixture = write_jsonl(args.fixture_path, cases)
    report = build_report(
        manifest_path=args.manifest_path,
        fixture_path=resolved_fixture,
        rows=cases,
        manifest_rows=materials,
        max_cases=args.max_cases,
    )
    resolved_report = ensure_parent_directory(args.report_path)
    with open(resolved_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    resolved_summary = ensure_parent_directory(args.summary_path)
    with open(resolved_summary, "w", encoding="utf-8") as handle:
        handle.write(summarize_report(report))
    print(json.dumps({"case_count": len(cases), "fixture_path": resolved_fixture}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
