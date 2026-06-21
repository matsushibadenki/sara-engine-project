#!/usr/bin/env python3
"""Build source-backed sparse own-latent manifests from autobot materials."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from bot.planner import CollectionPlanner
from sara_engine.learning.own_latent import (  # noqa: E402
    build_sparse_signature,
    stable_event_id,
    tokenize_sparse_text,
)
from sara_engine.utils.project_paths import (  # noqa: E402
    ensure_parent_directory,
    processed_data_path,
    workspace_path,
)


DEFAULT_MATERIALS_PATH = processed_data_path("autobot", "learning_materials.jsonl")
DEFAULT_MANIFEST_PATH = processed_data_path("autobot", "latent_manifest.jsonl")
DEFAULT_REPORT_PATH = workspace_path("evaluation", "own_latent_manifest_builder.json")
DEFAULT_SUMMARY_PATH = workspace_path("evaluation", "own_latent_manifest_builder_summary.txt")
DEFAULT_FIXTURE_FEEDBACK_PATH = workspace_path("evaluation", "concept_revalidation_fixture_builder.json")
DEFAULT_REQUEST_PLAN_PATH = workspace_path("autobot", "fixture_material_request_plan.json")
DEFAULT_TYPE_MATERIAL_PATHS = (
    processed_data_path("autobot", "qa_pairs.jsonl"),
    processed_data_path("autobot", "source_claims.jsonl"),
    processed_data_path("autobot", "contrastive_pairs.jsonl"),
    processed_data_path("autobot", "negative_queries.jsonl"),
    processed_data_path("autobot", "summaries.jsonl"),
    processed_data_path("autobot", "definition_cards.jsonl"),
    processed_data_path("autobot", "procedural_steps.jsonl"),
    processed_data_path("autobot", "transcript_segments.jsonl"),
    processed_data_path("autobot", "counterexamples.jsonl"),
    processed_data_path("autobot", "repair_notes.jsonl"),
    processed_data_path("autobot", "revision_notes.jsonl"),
)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payload["_line_number"] = line_number
                rows.append(payload)
    return rows


def read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> str:
    resolved = ensure_parent_directory(path)
    with open(resolved, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return resolved


def load_materials_with_fallback(materials_path: str) -> Dict[str, Any]:
    primary_rows = read_jsonl(materials_path)
    if primary_rows:
        return {
            "rows": primary_rows,
            "source_paths": [materials_path],
            "fallback_used": False,
        }

    rows: List[Dict[str, Any]] = []
    used_paths: List[str] = []
    seen_hashes = set()
    for path in DEFAULT_TYPE_MATERIAL_PATHS:
        path_rows = read_jsonl(path)
        if not path_rows:
            continue
        used_paths.append(path)
        for row in path_rows:
            material_hash = str(row.get("material_hash", "") or "")
            key = material_hash or json.dumps(row, sort_keys=True, ensure_ascii=False)
            if key in seen_hashes:
                continue
            seen_hashes.add(key)
            rows.append(row)
    return {
        "rows": rows,
        "source_paths": used_paths or [materials_path],
        "fallback_used": bool(rows),
    }


def _text_for_signature(material: Dict[str, Any]) -> str:
    return " ".join(
        str(material.get(key, "") or "")
        for key in ("material_type", "prompt", "answer", "content", "near_miss", "source_domain")
    )


def _latent_terms(material: Dict[str, Any], *, max_terms: int) -> List[str]:
    material_type = str(material.get("material_type", "unknown") or "unknown")
    source_type = str(material.get("source_type", "unknown") or "unknown")
    language = str(material.get("language", "unknown") or "unknown")
    tokens = tokenize_sparse_text(_text_for_signature(material))
    ranked = Counter(tokens)
    common_terms = [
        token
        for token, _count in sorted(ranked.items(), key=lambda item: (-item[1], item[0]))
        if token not in {"source", "prompt", "answer", "content"}
    ][: max(0, int(max_terms))]
    return [f"type:{material_type}", f"source_type:{source_type}", f"language:{language}"] + common_terms


def _cluster_id(material: Dict[str, Any], terms: Sequence[str]) -> str:
    material_type = str(material.get("material_type", "unknown") or "unknown")
    source_type = str(material.get("source_type", "unknown") or "unknown")
    basis = "|".join([material_type, source_type] + list(terms[:5]))
    return f"latent_{stable_event_id(basis, width=1_000_000):06d}"


def build_latent_manifest(
    materials: Sequence[Dict[str, Any]],
    *,
    width: int = 4096,
    max_events: int = 32,
    max_terms: int = 10,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, material in enumerate(materials):
        if material.get("accepted") is False:
            continue
        material_hash = str(material.get("material_hash", "") or "")
        if not material_hash:
            material_hash = f"line_{material.get('_line_number', index)}"
        terms = _latent_terms(material, max_terms=max_terms)
        signature = build_sparse_signature(terms, width=width, max_events=max_events)
        source_ref = (
            str(material.get("source_url", "") or "")
            or str(material.get("source_path", "") or "")
            or str(material.get("source", "") or "")
        )
        rows.append(
            {
                "schema": "sara-own-latent-manifest-row-v1",
                "manifest_id": f"latent_manifest_{index:06d}",
                "material_hash": material_hash,
                "material_type": str(material.get("material_type", "unknown") or "unknown"),
                "latent_cluster_id": _cluster_id(material, terms),
                "latent_terms": terms,
                "sparse_signature": signature,
                "signature_width": int(width),
                "source": str(material.get("source", "") or ""),
                "source_ref": source_ref,
                "source_url": str(material.get("source_url", "") or ""),
                "source_path": str(material.get("source_path", "") or ""),
                "source_type": str(material.get("source_type", "") or ""),
                "quality_score": float(material.get("quality_score", 0.0) or 0.0),
                "language": str(material.get("language", "unknown") or "unknown"),
                "license_hint": str(material.get("license_hint", "") or ""),
                "compliance_level": str(material.get("compliance_level", "") or ""),
                "event_cost": len(signature),
                "observed_only": True,
            }
        )
    return rows


def build_report(
    *,
    materials_path: str,
    material_source_paths: Sequence[str],
    fallback_used: bool,
    manifest_path: str,
    rows: Sequence[Dict[str, Any]],
    material_count: int,
    width: int,
    max_events: int,
    fixture_feedback_path: str,
    fixture_feedback: Optional[Dict[str, Any]],
    request_plan: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    type_counts = Counter(str(row.get("material_type", "unknown")) for row in rows)
    cluster_counts = Counter(str(row.get("latent_cluster_id", "")) for row in rows)
    missing_sources = sum(1 for row in rows if not row.get("source_ref"))
    max_event_cost = max((int(row.get("event_cost", 0) or 0) for row in rows), default=0)
    expansion_plan = []
    if isinstance(fixture_feedback, dict):
        raw_expansion_plan = fixture_feedback.get("expansion_plan", [])
        if isinstance(raw_expansion_plan, list):
            for item in raw_expansion_plan:
                if not isinstance(item, dict):
                    continue
                preferred = [
                    str(value)
                    for value in item.get("preferred_material_types", [])
                    if str(value)
                ]
                availability = {
                    material_type: int(type_counts.get(material_type, 0))
                    for material_type in preferred
                }
                missing_now = [
                    material_type for material_type in preferred if int(type_counts.get(material_type, 0)) <= 0
                ]
                expansion_plan.append(
                    {
                        "action": str(item.get("action", "") or ""),
                        "case_type": str(item.get("case_type", "") or ""),
                        "priority": item.get("priority"),
                        "preferred_material_types": preferred,
                        "manifest_availability": availability,
                        "missing_material_types_now": missing_now,
                        "guidance": str(item.get("guidance", "") or ""),
                    }
                )
    coverage_gap_count = sum(
        1
        for item in expansion_plan
        if isinstance(item, dict) and item.get("missing_material_types_now")
    )
    passed = bool(rows) and missing_sources == 0 and max_event_cost <= max_events
    return {
        "schema": "sara-own-latent-manifest-builder-report-v1",
        "passed": passed,
        "observed_only": True,
        "materials_path": os.path.abspath(materials_path),
        "material_source_paths": [os.path.abspath(path) for path in material_source_paths],
        "type_output_fallback_used": bool(fallback_used),
        "manifest_path": os.path.abspath(manifest_path),
        "material_count": int(material_count),
        "manifest_count": len(rows),
        "latent_cluster_count": len(cluster_counts),
        "material_type_counts": dict(sorted(type_counts.items())),
        "missing_source_ref_count": missing_sources,
        "max_event_cost": max_event_cost,
        "fixture_feedback_path": os.path.abspath(fixture_feedback_path),
        "fixture_feedback_loaded": isinstance(fixture_feedback, dict),
        "fixture_expansion_plan_count": len(expansion_plan),
        "fixture_material_coverage_gap_count": coverage_gap_count,
        "fixture_expansion_plan": expansion_plan,
        "fixture_material_request_count": 0
        if not isinstance(request_plan, dict)
        else int(request_plan.get("request_count", 0) or 0),
        "fixture_material_request_plan_path": ""
        if not isinstance(request_plan, dict)
        else str(request_plan.get("output_path", "") or ""),
        "signature_width": int(width),
        "max_events": int(max_events),
        "policy_notes": [
            "The latent manifest is source-backed and observed-only.",
            "Sparse signatures use bounded event sets, not dense embeddings.",
            "Accepted manifests are written under data/processed/autobot.",
            "Reports are written under workspace/evaluation.",
        ],
    }


def summarize_report(report: Dict[str, Any]) -> str:
    lines = [
        f"Own-latent manifest builder: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Observed only: {report.get('observed_only')}",
        f"Materials: {report.get('material_count')}",
        f"Manifest rows: {report.get('manifest_count')}",
        f"Latent clusters: {report.get('latent_cluster_count')}",
        f"Missing source refs: {report.get('missing_source_ref_count')}",
        f"Max event cost: {report.get('max_event_cost')}/{report.get('max_events')}",
        "Material types:",
    ]
    for key, value in sorted(report.get("material_type_counts", {}).items()):
        lines.append(f"- {key}: {value}")
    lines.append(f"Fixture feedback loaded: {report.get('fixture_feedback_loaded')}")
    lines.append(f"Fixture coverage gaps: {report.get('fixture_material_coverage_gap_count')}")
    lines.append(f"Fixture material requests: {report.get('fixture_material_request_count')}")
    expansion_plan = report.get("fixture_expansion_plan", [])
    if isinstance(expansion_plan, list) and expansion_plan:
        lines.append("Fixture expansion alignment:")
        for item in expansion_plan:
            if not isinstance(item, dict):
                continue
            lines.append(
                "- "
                f"{item.get('action', '')} "
                f"(missing_now={','.join(item.get('missing_material_types_now', []))}, "
                f"preferred={','.join(item.get('preferred_material_types', []))})"
            )
    return "\n".join(lines) + "\n"


def run_builder(
    *,
    materials_path: str = DEFAULT_MATERIALS_PATH,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    report_path: str = DEFAULT_REPORT_PATH,
    summary_path: str = DEFAULT_SUMMARY_PATH,
    fixture_feedback_path: str = DEFAULT_FIXTURE_FEEDBACK_PATH,
    request_plan_path: str = DEFAULT_REQUEST_PLAN_PATH,
    width: int = 4096,
    max_events: int = 32,
    max_terms: int = 10,
) -> Dict[str, Any]:
    loaded = load_materials_with_fallback(materials_path)
    materials = loaded["rows"]
    fixture_feedback = read_json(fixture_feedback_path)
    planner = CollectionPlanner()
    request_plan = (
        planner.write_fixture_material_request_plan(fixture_feedback, output_path=request_plan_path)
        if isinstance(fixture_feedback, dict)
        else None
    )
    rows = build_latent_manifest(
        materials,
        width=width,
        max_events=max_events,
        max_terms=max_terms,
    )
    resolved_manifest = write_jsonl(manifest_path, rows)
    report = build_report(
        materials_path=materials_path,
        material_source_paths=loaded["source_paths"],
        fallback_used=bool(loaded["fallback_used"]),
        manifest_path=resolved_manifest,
        rows=rows,
        material_count=len(materials),
        width=width,
        max_events=max_events,
        fixture_feedback_path=fixture_feedback_path,
        fixture_feedback=fixture_feedback,
        request_plan=request_plan,
    )
    resolved_report = ensure_parent_directory(report_path)
    with open(resolved_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    resolved_summary = ensure_parent_directory(summary_path)
    with open(resolved_summary, "w", encoding="utf-8") as handle:
        handle.write(summarize_report(report))
    report["outputs"] = {
        "manifest": resolved_manifest,
        "report": resolved_report,
        "summary": resolved_summary,
    }
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build source-backed sparse own-latent manifests.")
    parser.add_argument("--materials-path", default=DEFAULT_MATERIALS_PATH)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--fixture-feedback-path", default=DEFAULT_FIXTURE_FEEDBACK_PATH)
    parser.add_argument("--request-plan-path", default=DEFAULT_REQUEST_PLAN_PATH)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--max-terms", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_builder(
        materials_path=args.materials_path,
        manifest_path=args.manifest_path,
        report_path=args.report_path,
        summary_path=args.summary_path,
        fixture_feedback_path=args.fixture_feedback_path,
        request_plan_path=args.request_plan_path,
        width=args.width,
        max_events=args.max_events,
        max_terms=args.max_terms,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "observed_only": report["observed_only"],
                "manifest_count": report["manifest_count"],
                "latent_cluster_count": report["latent_cluster_count"],
                "report_path": os.path.abspath(args.report_path),
                "summary_path": os.path.abspath(args.summary_path),
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
