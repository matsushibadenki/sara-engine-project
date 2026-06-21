from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from bot.curriculum_manifest import build_curriculum_manifest, summarize_curriculum
from bot.learning_material_gate import material_hash, normalize_text
from sara_engine.utils.project_paths import ensure_parent_directory, processed_data_path, workspace_path


DEFAULT_ACCEPTED_PATH = processed_data_path("autobot", "learning_materials.jsonl")
DEFAULT_TARGETS_PATH = workspace_path("autobot", "dataset_builder_collection_targets.json")
DEFAULT_OUTPUT_PATH = processed_data_path("autobot", "gap_materials.jsonl")
DEFAULT_CURRICULUM_PATH = processed_data_path("autobot", "gap_curriculum_manifest.jsonl")
DEFAULT_REPORT_PATH = workspace_path("autobot", "gap_materials_builder_report.json")
DEFAULT_SUMMARY_PATH = workspace_path("autobot", "gap_materials_builder_summary.txt")

TYPE_OUTPUTS = {
    "transcript_segment": processed_data_path("autobot", "transcript_segments.jsonl"),
    "counterexample": processed_data_path("autobot", "counterexamples.jsonl"),
    "repair_note": processed_data_path("autobot", "repair_notes.jsonl"),
    "revision_note": processed_data_path("autobot", "revision_notes.jsonl"),
}


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


def write_json(path: str, payload: Dict[str, Any]) -> str:
    resolved = ensure_parent_directory(path)
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return resolved


def _sentences(text: str, limit: int = 3) -> List[str]:
    cleaned = normalize_text(text)
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", cleaned)
    sentences = [part.strip(" -\t") for part in parts if len(part.strip()) >= 12]
    if not sentences and cleaned:
        sentences = [cleaned[:240]]
    return sentences[:limit]


def _base_gap_material(source: Dict[str, Any], material_type: str, request_id: str) -> Dict[str, Any]:
    return {
        "schema": "sara-autobot-gap-material-v1",
        "material_type": material_type,
        "source": str(source.get("source", "") or ""),
        "source_url": str(source.get("source_url", "") or ""),
        "source_path": str(source.get("source_path", "") or ""),
        "source_type": str(source.get("source_type", "") or ""),
        "source_domain": str(source.get("source_domain", "") or ""),
        "collection_time": str(source.get("collection_time", "") or ""),
        "quality_score": float(source.get("quality_score", 0.5) or 0.5),
        "language": str(source.get("language", "unknown") or "unknown"),
        "license_hint": str(source.get("license_hint", "") or ""),
        "compliance_level": str(source.get("compliance_level", "") or ""),
        "source_text": str(source.get("source_text", "") or ""),
        "request_id": request_id,
        "accepted": True,
        "observed_only": True,
    }


def _select_seed_materials(
    accepted: Sequence[Dict[str, Any]],
    preferred_material_types: Sequence[str],
) -> List[Dict[str, Any]]:
    preferred = [str(item) for item in preferred_material_types if str(item)]
    selected = [
        item for item in accepted if str(item.get("material_type", "")) in preferred
    ]
    if selected:
        return selected
    return list(accepted)


def _build_transcript_segment(source: Dict[str, Any], request_id: str) -> Optional[Dict[str, Any]]:
    sentences = _sentences(str(source.get("source_text", "") or ""), limit=2)
    if not sentences:
        return None
    item = _base_gap_material(source, "transcript_segment", request_id)
    item.update(
        {
            "prompt": "Replay the supporting source segment for sparse event grounding.",
            "content": " ".join(sentences),
            "segment_role": "supporting_evidence",
        }
    )
    item["material_hash"] = material_hash(item)
    return item


def _build_counterexample(source: Dict[str, Any], request_id: str) -> Optional[Dict[str, Any]]:
    near_miss = normalize_text(str(source.get("near_miss", "") or ""))
    supported = normalize_text(str(source.get("answer", "") or str(source.get("content", "") or "")))
    if not near_miss:
        return None
    item = _base_gap_material(source, "counterexample", request_id)
    item.update(
        {
            "prompt": "Explain why the near-miss statement should not replace the supported source claim.",
            "answer": supported,
            "content": near_miss,
            "expected_behavior": "prefer_supported_claim_over_near_miss",
        }
    )
    item["material_hash"] = material_hash(item)
    return item


def _build_repair_note(source: Dict[str, Any], request_id: str) -> Optional[Dict[str, Any]]:
    supported = normalize_text(str(source.get("answer", "") or str(source.get("content", "") or "")))
    if not supported:
        return None
    item = _base_gap_material(source, "repair_note", request_id)
    item.update(
        {
            "prompt": "Record the minimal support needed to rebuild a stalled concept candidate.",
            "content": supported,
            "repair_focus": "support_rebuild",
        }
    )
    item["material_hash"] = material_hash(item)
    return item


def _build_revision_note(source: Dict[str, Any], request_id: str) -> Optional[Dict[str, Any]]:
    supported = normalize_text(str(source.get("content", "") or str(source.get("answer", "") or "")))
    if not supported:
        return None
    item = _base_gap_material(source, "revision_note", request_id)
    item.update(
        {
            "prompt": "Capture the source-backed statement that should be compared across revisions.",
            "content": supported,
            "revision_anchor": str(source.get("source_url", "") or source.get("source_path", "") or source.get("source", "")),
        }
    )
    item["material_hash"] = material_hash(item)
    return item


BUILDERS = {
    "transcript_segment": _build_transcript_segment,
    "counterexample": _build_counterexample,
    "repair_note": _build_repair_note,
    "revision_note": _build_revision_note,
}


def build_gap_materials(
    *,
    accepted: Sequence[Dict[str, Any]],
    targets_payload: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    requests = []
    if isinstance(targets_payload, dict) and isinstance(targets_payload.get("targets"), list):
        requests = targets_payload.get("targets", [])
    built: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    seen_hashes = set()
    for request in requests:
        if not isinstance(request, dict):
            continue
        request_id = str(request.get("request_id", "") or "")
        preferred_material_types = request.get("preferred_material_types", [])
        seeds = _select_seed_materials(accepted, preferred_material_types)
        missing_types = [
            str(item) for item in request.get("missing_material_types", []) if str(item)
        ]
        for missing_type in missing_types:
            builder = BUILDERS.get(missing_type)
            if builder is None:
                skipped.append({"request_id": request_id, "material_type": missing_type, "reason": "unsupported_gap_type"})
                continue
            built_one = False
            for seed in seeds:
                candidate = builder(seed, request_id)
                if not isinstance(candidate, dict):
                    continue
                candidate_hash = str(candidate.get("material_hash", "") or "")
                if not candidate_hash or candidate_hash in seen_hashes:
                    continue
                seen_hashes.add(candidate_hash)
                built.append(candidate)
                built_one = True
                break
            if not built_one:
                skipped.append({"request_id": request_id, "material_type": missing_type, "reason": "no_seed_material"})
    return built, skipped


def derive_evaluation_gaps(targets_payload: Optional[Dict[str, Any]]) -> List[str]:
    merged = set()
    if not isinstance(targets_payload, dict):
        return []
    requests = targets_payload.get("targets", [])
    if not isinstance(requests, list):
        return []
    for item in requests:
        if not isinstance(item, dict):
            continue
        for gap in item.get("evaluation_gaps", []):
            if str(gap):
                merged.add(str(gap))
    return sorted(merged)


def build_report(
    *,
    accepted_path: str,
    targets_path: str,
    output_path: str,
    built_rows: Sequence[Dict[str, Any]],
    skipped_rows: Sequence[Dict[str, Any]],
    curriculum_manifest: Sequence[Dict[str, Any]],
    evaluation_gaps: Sequence[str],
) -> Dict[str, Any]:
    type_counts = Counter(str(item.get("material_type", "unknown")) for item in built_rows)
    skipped_counts = Counter(str(item.get("material_type", "unknown")) for item in skipped_rows)
    curriculum_summary = summarize_curriculum(curriculum_manifest)
    return {
        "schema": "sara-autobot-gap-materials-builder-report-v1",
        "passed": bool(built_rows),
        "accepted_path": os.path.abspath(accepted_path),
        "targets_path": os.path.abspath(targets_path),
        "output_path": os.path.abspath(output_path),
        "built_count": len(built_rows),
        "skipped_count": len(skipped_rows),
        "built_material_type_counts": dict(sorted(type_counts.items())),
        "skipped_material_type_counts": dict(sorted(skipped_counts.items())),
        "evaluation_gaps": list(evaluation_gaps),
        "curriculum_distribution": curriculum_summary["curriculum_distribution"],
        "curriculum_material_type_counts": curriculum_summary["material_type_counts"],
        "policy_notes": [
            "Gap materials are deterministic source-backed supplements derived from accepted autobot materials.",
            "Supplement outputs stay under data/processed/autobot and workspace/autobot.",
            "Unsupported or seedless gap targets are preserved in the report instead of being silently dropped.",
        ],
    }


def summarize_report(report: Dict[str, Any]) -> str:
    lines = [
        f"Gap materials builder: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Built: {report.get('built_count')}",
        f"Skipped: {report.get('skipped_count')}",
        f"Evaluation gaps: {','.join(report.get('evaluation_gaps', []))}",
        "Built material types:",
    ]
    for key, value in sorted(report.get("built_material_type_counts", {}).items()):
        lines.append(f"- {key}: {value}")
    lines.append("Curriculum distribution:")
    for key, value in sorted(report.get("curriculum_distribution", {}).items()):
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def run_builder(
    *,
    accepted_path: str = DEFAULT_ACCEPTED_PATH,
    targets_path: str = DEFAULT_TARGETS_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    curriculum_path: str = DEFAULT_CURRICULUM_PATH,
    report_path: str = DEFAULT_REPORT_PATH,
    summary_path: str = DEFAULT_SUMMARY_PATH,
) -> Dict[str, Any]:
    accepted = read_jsonl(accepted_path)
    targets_payload = read_json(targets_path)
    built_rows, skipped_rows = build_gap_materials(accepted=accepted, targets_payload=targets_payload)
    evaluation_gaps = derive_evaluation_gaps(targets_payload)
    curriculum_manifest = build_curriculum_manifest(built_rows, evaluation_gaps=evaluation_gaps)
    outputs = {"all_gap_materials": write_jsonl(output_path, built_rows)}
    outputs["curriculum_manifest"] = write_jsonl(curriculum_path, curriculum_manifest)
    for material_type, path in TYPE_OUTPUTS.items():
        rows = [item for item in built_rows if item.get("material_type") == material_type]
        outputs[material_type] = write_jsonl(path, rows)
    report = build_report(
        accepted_path=accepted_path,
        targets_path=targets_path,
        output_path=output_path,
        built_rows=built_rows,
        skipped_rows=skipped_rows,
        curriculum_manifest=curriculum_manifest,
        evaluation_gaps=evaluation_gaps,
    )
    report["outputs"] = outputs
    report["skipped"] = skipped_rows
    report["report_path"] = write_json(report_path, report)
    report["summary_path"] = write_json(summary_path, {"summary": summarize_report(report)})
    with open(report["summary_path"], "w", encoding="utf-8") as handle:
        handle.write(summarize_report(report))
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build source-backed gap materials from collection targets.")
    parser.add_argument("--accepted-path", default=DEFAULT_ACCEPTED_PATH)
    parser.add_argument("--targets-path", default=DEFAULT_TARGETS_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--curriculum-path", default=DEFAULT_CURRICULUM_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_builder(
        accepted_path=args.accepted_path,
        targets_path=args.targets_path,
        output_path=args.output_path,
        curriculum_path=args.curriculum_path,
        report_path=args.report_path,
        summary_path=args.summary_path,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "built_count": report["built_count"],
                "skipped_count": report["skipped_count"],
                "report_path": report["report_path"],
                "summary_path": report["summary_path"],
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
