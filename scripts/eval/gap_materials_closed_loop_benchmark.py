#!/usr/bin/env python3
"""Measure whether deterministic gap materials reduce own-latent fixture coverage gaps."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from bot.gap_materials_builder import build_gap_materials, read_json as read_bot_json
from bot.planner import CollectionPlanner
from bot.dataset_builder import build_collection_targets
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path

import scripts.eval.own_latent_manifest_builder as own_latent_manifest_builder  # noqa: E402


DEFAULT_ACCEPTED_PATH = processed_data_path("autobot", "learning_materials.jsonl")
DEFAULT_TARGETS_PATH = workspace_path("autobot", "dataset_builder_collection_targets.json")
DEFAULT_FIXTURE_FEEDBACK_PATH = workspace_path("evaluation", "concept_revalidation_fixture_builder.json")
DEFAULT_REQUEST_PLAN_PATH = workspace_path("autobot", "fixture_material_request_plan.json")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "gap_materials_closed_loop_benchmark.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "gap_materials_closed_loop_benchmark_summary.txt")


def _build_manifest_report(
    *,
    materials: Sequence[Dict[str, Any]],
    materials_path: str,
    fixture_feedback_path: str,
    request_plan_path: str,
    width: int,
    max_events: int,
    max_terms: int,
) -> Dict[str, Any]:
    fixture_feedback = own_latent_manifest_builder.read_json(fixture_feedback_path)
    request_plan = own_latent_manifest_builder.read_json(request_plan_path)
    manifest_rows = own_latent_manifest_builder.build_latent_manifest(
        materials,
        width=width,
        max_events=max_events,
        max_terms=max_terms,
    )
    return own_latent_manifest_builder.build_report(
        materials_path=materials_path,
        material_source_paths=[materials_path],
        fallback_used=False,
        manifest_path=materials_path,
        rows=manifest_rows,
        material_count=len(materials),
        width=width,
        max_events=max_events,
        fixture_feedback_path=fixture_feedback_path,
        fixture_feedback=fixture_feedback,
        request_plan=request_plan,
    )


def build_report(
    *,
    accepted_path: str,
    targets_path: str,
    fixture_feedback_path: str,
    request_plan_path: str,
    width: int,
    max_events: int,
    max_terms: int,
) -> Dict[str, Any]:
    loaded = own_latent_manifest_builder.load_materials_with_fallback(accepted_path)
    accepted = loaded["rows"]
    targets_payload = read_bot_json(targets_path)
    request_plan_payload = read_bot_json(request_plan_path)
    if not isinstance(targets_payload, dict) or not isinstance(targets_payload.get("targets"), list) or not targets_payload.get("targets"):
        targets_payload = build_collection_targets(
            accepted=accepted,
            fixture_request_plan=request_plan_payload,
        )
    gap_rows, skipped = build_gap_materials(accepted=accepted, targets_payload=targets_payload)

    baseline_report = _build_manifest_report(
        materials=accepted,
        materials_path=accepted_path,
        fixture_feedback_path=fixture_feedback_path,
        request_plan_path=request_plan_path,
        width=width,
        max_events=max_events,
        max_terms=max_terms,
    )
    augmented_materials = list(accepted) + list(gap_rows)
    augmented_report = _build_manifest_report(
        materials=augmented_materials,
        materials_path=accepted_path,
        fixture_feedback_path=fixture_feedback_path,
        request_plan_path=request_plan_path,
        width=width,
        max_events=max_events,
        max_terms=max_terms,
    )

    built_type_counts = Counter(str(item.get("material_type", "unknown")) for item in gap_rows)
    baseline_gap_count = int(baseline_report.get("fixture_material_coverage_gap_count", 0) or 0)
    augmented_gap_count = int(augmented_report.get("fixture_material_coverage_gap_count", 0) or 0)
    planner = CollectionPlanner()
    requests = planner.material_requests_from_fixture_feedback(request_plan_payload or {})
    target_request_ids = sorted(
        {
            str(item.get("request_id", "") or "")
            for item in (
                targets_payload.get("targets", [])
                if isinstance(targets_payload, dict) and isinstance(targets_payload.get("targets", []), list)
                else []
            )
            if isinstance(item, dict) and str(item.get("request_id", "") or "")
        }
    )
    built_request_ids = sorted(
        {
            str(item.get("request_id", "") or "")
            for item in gap_rows
            if isinstance(item, dict) and str(item.get("request_id", "") or "")
        }
    )
    bundle_request_tokens = (
        "bundle",
        "source_diversity",
        "counterexample",
        "repair_support",
        "revision_conflict",
    )
    bundle_relevant_target_request_ids = [
        request_id for request_id in target_request_ids if any(token in request_id for token in bundle_request_tokens)
    ]
    bundle_relevant_built_request_ids = [
        request_id for request_id in built_request_ids if any(token in request_id for token in bundle_request_tokens)
    ]

    return {
        "schema": "sara-gap-materials-closed-loop-benchmark-report-v1",
        "passed": bool(accepted) and baseline_gap_count >= augmented_gap_count and bool(gap_rows),
        "accepted_path": os.path.abspath(accepted_path),
        "accepted_fallback_used": bool(loaded.get("fallback_used")),
        "targets_path": os.path.abspath(targets_path),
        "fixture_feedback_path": os.path.abspath(fixture_feedback_path),
        "request_plan_path": os.path.abspath(request_plan_path),
        "baseline_fixture_material_coverage_gap_count": baseline_gap_count,
        "augmented_fixture_material_coverage_gap_count": augmented_gap_count,
        "coverage_gap_reduction": baseline_gap_count - augmented_gap_count,
        "gap_material_built_count": len(gap_rows),
        "gap_material_built_type_counts": dict(sorted(built_type_counts.items())),
        "gap_material_skipped_count": len(skipped),
        "fixture_request_count": len(requests),
        "target_request_ids": target_request_ids,
        "built_request_ids": built_request_ids,
        "bundle_relevant_target_request_ids": bundle_relevant_target_request_ids,
        "bundle_relevant_built_request_ids": bundle_relevant_built_request_ids,
        "bundle_relevant_request_coverage": (
            1.0
            if not bundle_relevant_target_request_ids
            else float(len(bundle_relevant_built_request_ids)) / float(len(bundle_relevant_target_request_ids))
        ),
        "policy_notes": [
            "Closed-loop evidence compares accepted-only own-latent coverage against accepted-plus-gap-material coverage.",
            "Gap materials remain deterministic and source-backed throughout the comparison.",
            "All benchmark outputs stay under workspace/evaluation.",
        ],
    }


def summarize_report(report: Dict[str, Any]) -> str:
    lines = [
        f"Gap materials closed loop: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Baseline coverage gaps: {report.get('baseline_fixture_material_coverage_gap_count')}",
        f"Augmented coverage gaps: {report.get('augmented_fixture_material_coverage_gap_count')}",
        f"Gap reduction: {report.get('coverage_gap_reduction')}",
        f"Built gap materials: {report.get('gap_material_built_count')}",
        f"Bundle-relevant request coverage: {float(report.get('bundle_relevant_request_coverage', 0.0) or 0.0):.3f}",
        "Built gap material types:",
    ]
    for key, value in sorted(report.get("gap_material_built_type_counts", {}).items()):
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the closed-loop effect of gap materials on latent coverage gaps.")
    parser.add_argument("--accepted-path", default=DEFAULT_ACCEPTED_PATH)
    parser.add_argument("--targets-path", default=DEFAULT_TARGETS_PATH)
    parser.add_argument("--fixture-feedback-path", default=DEFAULT_FIXTURE_FEEDBACK_PATH)
    parser.add_argument("--request-plan-path", default=DEFAULT_REQUEST_PLAN_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--max-terms", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = build_report(
        accepted_path=args.accepted_path,
        targets_path=args.targets_path,
        fixture_feedback_path=args.fixture_feedback_path,
        request_plan_path=args.request_plan_path,
        width=args.width,
        max_events=args.max_events,
        max_terms=args.max_terms,
    )
    resolved_report_path = ensure_parent_directory(args.report_path)
    with open(resolved_report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    resolved_summary_path = ensure_parent_directory(args.summary_path)
    with open(resolved_summary_path, "w", encoding="utf-8") as handle:
        handle.write(summarize_report(report))
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "coverage_gap_reduction": report["coverage_gap_reduction"],
                "report_path": os.path.abspath(args.report_path),
                "summary_path": os.path.abspath(args.summary_path),
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
