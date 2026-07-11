from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

from bot.curriculum_manifest import build_curriculum_manifest, summarize_curriculum
from bot.learning_material_gate import normalize_text, split_accepted_rejected
from sara_engine.utils.project_paths import (
    ensure_parent_directory,
    interim_data_path,
    processed_data_path,
    workspace_path,
)


DEFAULT_RECORDS_PATH = processed_data_path("autobot", "multimodal_records.jsonl")
DEFAULT_CANDIDATE_PATH = interim_data_path("autobot", "candidate_learning_materials.jsonl")
DEFAULT_REJECTED_PATH = interim_data_path("autobot", "rejected_learning_materials.jsonl")
DEFAULT_ACCEPTED_PATH = processed_data_path("autobot", "learning_materials.jsonl")
DEFAULT_CURRICULUM_PATH = processed_data_path("autobot", "curriculum_manifest.jsonl")
DEFAULT_REPORT_PATH = workspace_path("autobot", "dataset_builder_report.json")
DEFAULT_SUMMARY_PATH = workspace_path("autobot", "dataset_builder_summary.txt")
DEFAULT_FIXTURE_REQUEST_PLAN_PATH = workspace_path("autobot", "fixture_material_request_plan.json")
DEFAULT_COLLECTION_TARGETS_PATH = workspace_path("autobot", "dataset_builder_collection_targets.json")

TYPE_OUTPUTS = {
    "qa_pair": processed_data_path("autobot", "qa_pairs.jsonl"),
    "contrastive_pair": processed_data_path("autobot", "contrastive_pairs.jsonl"),
    "negative_query": processed_data_path("autobot", "negative_queries.jsonl"),
    "summary": processed_data_path("autobot", "summaries.jsonl"),
    "definition_card": processed_data_path("autobot", "definition_cards.jsonl"),
    "procedural_steps": processed_data_path("autobot", "procedural_steps.jsonl"),
    "source_claim": processed_data_path("autobot", "source_claims.jsonl"),
}


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


def write_json(path: str, payload: Dict[str, Any]) -> str:
    resolved = ensure_parent_directory(path)
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return resolved


def _sentences(text: str, limit: int = 8) -> List[str]:
    cleaned = normalize_text(text)
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", cleaned)
    sentences = [part.strip(" -\t") for part in parts if len(part.strip()) >= 20]
    if not sentences and cleaned:
        sentences = [cleaned[:320]]
    return sentences[:limit]


def _language(text: str, meta: Dict[str, Any]) -> str:
    raw = str(meta.get("language", meta.get("lang", ""))).strip().lower()
    if raw:
        return raw
    jp = len(re.findall(r"[\u3040-\u30FF\u4E00-\u9FFF]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    if jp > en:
        return "jp"
    if en > 0:
        return "en"
    return "unknown"


def _source_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _near_duplicate_signature(text: str) -> str:
    """Build a bounded SimHash-style signature without dense vector operations."""
    tokens = re.findall(r"[\w-]+", normalize_text(text).lower())[:256]
    if not tokens:
        return ""
    weights = [0] * 64
    for token in tokens:
        token_hash = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
        for bit in range(64):
            weights[bit] += 1 if token_hash & (1 << bit) else -1
    signature = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            signature |= 1 << bit
    return f"{signature:016x}"


def _record_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    meta = record.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    source = str(record.get("source", meta.get("source", "autobot")) or "autobot")
    source_url = str(meta.get("url", meta.get("source_url", "")) or "")
    source_path = str(meta.get("path", meta.get("source_path", "")) or "")
    source_type = str(meta.get("source_type", source) or source)
    domain = urlparse(source_url).netloc.lower() if source_url else ""
    source_text = normalize_text(str(record.get("record_text", record.get("text", "")) or ""))
    source_hash = str(meta.get("source_hash", record.get("source_hash", "")) or "")
    if not source_hash:
        source_hash = _source_hash(source_text)
    source_revision = str(
        meta.get("source_revision", meta.get("revision", record.get("source_revision", ""))) or ""
    )
    return {
        "source": source,
        "source_url": source_url,
        "source_path": source_path,
        "source_type": source_type,
        "source_domain": domain,
        "collection_time": str(record.get("ts", meta.get("collection_time", "")) or ""),
        "source_hash": source_hash,
        "source_revision": source_revision or source_hash,
        "near_duplicate_signature": str(
            meta.get("near_duplicate_signature", record.get("near_duplicate_signature", ""))
            or _near_duplicate_signature(source_text)
        ),
        "quality_score": float(meta.get("quality", meta.get("quality_score", 0.5)) or 0.5),
        "license_hint": str(meta.get("license_hint", meta.get("license", "")) or ""),
        "compliance_level": str(meta.get("compliance_level", meta.get("compliance", "unknown")) or "unknown"),
    }


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    text = normalize_text(str(record.get("record_text", record.get("text", "")) or ""))
    meta = _record_metadata(record)
    meta["language"] = _language(text, record.get("meta", {}) if isinstance(record.get("meta", {}), dict) else {})
    meta["record_id"] = str(record.get("record_id", record.get("_line_number", "")) or "")
    meta["source_text"] = text
    return meta


def _base_material(record: Dict[str, Any], material_type: str) -> Dict[str, Any]:
    return {
        "schema": "sara-autobot-learning-material-v1",
        "material_type": material_type,
        "source": record.get("source", ""),
        "source_url": record.get("source_url", ""),
        "source_path": record.get("source_path", ""),
        "source_type": record.get("source_type", ""),
        "source_domain": record.get("source_domain", ""),
        "collection_time": record.get("collection_time", ""),
        "source_hash": record.get("source_hash", ""),
        "source_revision": record.get("source_revision", ""),
        "near_duplicate_signature": record.get("near_duplicate_signature", ""),
        "quality_score": record.get("quality_score", 0.5),
        "language": record.get("language", "unknown"),
        "license_hint": record.get("license_hint", ""),
        "compliance_level": record.get("compliance_level", ""),
        "source_text": record.get("source_text", ""),
    }


def build_materials_for_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = str(record.get("source_text", ""))
    sentences = _sentences(text)
    if not sentences:
        return []

    materials: List[Dict[str, Any]] = []
    summary = " ".join(sentences[:2])[:420]
    item = _base_material(record, "summary")
    item.update({"content": summary, "prompt": "Summarize the source in one compact note."})
    materials.append(item)

    answer = sentences[0]
    topic_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", answer)
    topic = " ".join(topic_tokens[:3]) if topic_tokens else "the source"
    item = _base_material(record, "qa_pair")
    item.update({"prompt": f"What does the source say about {topic}?", "answer": answer})
    materials.append(item)

    item = _base_material(record, "source_claim")
    item.update({"content": answer, "prompt": "Verify this claim against the source."})
    materials.append(item)

    definition_sentence = next(
        (sentence for sentence in sentences if re.search(r"\b(is|are|means|refers to)\b", sentence, re.IGNORECASE)),
        "",
    )
    if definition_sentence:
        term = re.split(r"\b(?:is|are|means|refers to)\b", definition_sentence, maxsplit=1, flags=re.IGNORECASE)[0]
        item = _base_material(record, "definition_card")
        item.update({"prompt": f"Define {normalize_text(term)[:80]}", "answer": definition_sentence})
        materials.append(item)

    negative_anchor = re.sub(r"[^a-zA-Z0-9_]+", "_", str(record.get("record_id", "record")) or "record")
    item = _base_material(record, "negative_query")
    item.update(
        {
            "prompt": f"absent_decoy_{negative_anchor}_not_in_source",
            "answer": "",
            "expected_behavior": "abstain",
        }
    )
    materials.append(item)

    step_candidates = [
        sentence
        for sentence in sentences
        if re.search(r"\b(step|first|then|next|finally|install|run|use|write)\b", sentence, re.IGNORECASE)
    ]
    if len(step_candidates) >= 2:
        item = _base_material(record, "procedural_steps")
        item.update({"prompt": "Extract the procedure from the source.", "content": " ".join(step_candidates[:4])})
        materials.append(item)

    return materials


def build_contrastive_pairs(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    normalized = [record for record in records if len(str(record.get("source_text", ""))) >= 24]
    for left, right in zip(normalized, normalized[1:]):
        left_sentence = _sentences(str(left.get("source_text", "")), limit=1)
        right_sentence = _sentences(str(right.get("source_text", "")), limit=1)
        if not left_sentence or not right_sentence or left_sentence[0] == right_sentence[0]:
            continue
        item = _base_material(left, "contrastive_pair")
        item.update(
            {
                "prompt": "Choose the claim supported by the first source, not the near-miss source.",
                "answer": left_sentence[0],
                "near_miss": right_sentence[0],
                "near_miss_source": right.get("source_url") or right.get("source_path") or right.get("source"),
            }
        )
        pairs.append(item)
    return pairs


def build_candidate_materials(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_records = [normalize_record(record) for record in records]
    candidates: List[Dict[str, Any]] = []
    for record in normalized_records:
        candidates.extend(build_materials_for_record(record))
    candidates.extend(build_contrastive_pairs(normalized_records))
    return candidates


def derive_evaluation_gaps(
    explicit_gaps: Sequence[str],
    fixture_request_plan: Optional[Dict[str, Any]],
) -> List[str]:
    merged = {str(item) for item in explicit_gaps if str(item)}
    if not isinstance(fixture_request_plan, dict):
        return sorted(merged)
    requests = fixture_request_plan.get("requests", [])
    if not isinstance(requests, list):
        return sorted(merged)
    for item in requests:
        if not isinstance(item, dict):
            continue
        for gap in item.get("evaluation_gaps", []):
            if str(gap):
                merged.add(str(gap))
    return sorted(merged)


def build_collection_targets(
    *,
    accepted: Sequence[Dict[str, Any]],
    fixture_request_plan: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    accepted_types = Counter(str(item.get("material_type", "unknown")) for item in accepted)
    source_domains = sorted(
        {
            str(item.get("source_domain", "") or "local")
            for item in accepted
            if str(item.get("source_domain", "") or "local")
        }
    )
    targets: List[Dict[str, Any]] = []
    requests = []
    if isinstance(fixture_request_plan, dict) and isinstance(fixture_request_plan.get("requests"), list):
        requests = fixture_request_plan.get("requests", [])
    for item in requests:
        if not isinstance(item, dict):
            continue
        missing_material_types = [
            str(value) for value in item.get("missing_material_types", []) if str(value)
        ]
        targets.append(
            {
                "request_id": str(item.get("request_id", "") or ""),
                "reason": str(item.get("reason", "") or ""),
                "priority": float(item.get("priority", 0.0) or 0.0),
                "evaluation_gaps": [
                    str(value) for value in item.get("evaluation_gaps", []) if str(value)
                ],
                "missing_material_types": missing_material_types,
                "preferred_material_types": [
                    str(value) for value in item.get("material_types", []) if str(value)
                ],
                "current_material_counts": {
                    material_type: int(accepted_types.get(material_type, 0))
                    for material_type in missing_material_types
                },
                "candidate_source_domains": source_domains,
                "guidance": str(item.get("guidance", "") or ""),
            }
        )
    return {
        "schema": "sara-autobot-collection-targets-v1",
        "target_count": len(targets),
        "targets": targets,
    }


def build_report(
    records: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    accepted: Sequence[Dict[str, Any]],
    rejected: Sequence[Dict[str, Any]],
    manifest: Sequence[Dict[str, Any]],
    outputs: Dict[str, str],
    evaluation_gaps: Sequence[str],
    fixture_request_plan: Optional[Dict[str, Any]],
    collection_targets: Dict[str, Any],
) -> Dict[str, Any]:
    accepted_types = Counter(str(item.get("material_type", "unknown")) for item in accepted)
    rejected_reasons = Counter(str(item.get("gate_reason", "unknown")) for item in rejected)
    languages = Counter(str(item.get("language", "unknown")) for item in accepted)
    domains = Counter(str(item.get("source_domain", "")) or "local" for item in accepted)
    curriculum_summary = summarize_curriculum(manifest)
    return {
        "schema": "sara-autobot-dataset-builder-report-v1",
        "generated_at": datetime.utcnow().isoformat(),
        "passed": bool(accepted),
        "record_count": len(records),
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "duplicate_rejection_count": int(rejected_reasons.get("duplicate_material", 0)),
        "accepted_material_type_counts": dict(sorted(accepted_types.items())),
        "rejected_reason_counts": dict(sorted(rejected_reasons.items())),
        "language_balance": dict(sorted(languages.items())),
        "source_domain_counts": dict(sorted(domains.items())),
        "curriculum_distribution": curriculum_summary["curriculum_distribution"],
        "curriculum_material_type_counts": curriculum_summary["material_type_counts"],
        "evaluation_gaps": list(evaluation_gaps),
        "fixture_request_plan_loaded": isinstance(fixture_request_plan, dict),
        "fixture_request_count": 0
        if not isinstance(fixture_request_plan, dict)
        else int(fixture_request_plan.get("request_count", 0) or 0),
        "collection_target_count": int(collection_targets.get("target_count", 0) or 0),
        "outputs": outputs,
        "policy_notes": [
            "Generated materials are deterministic source-backed extracts.",
            "Rejected materials remain under data/interim/autobot for audit.",
            "Accepted final materials and curriculum manifests stay under data/processed/autobot.",
            "The operator report stays under workspace/autobot.",
        ],
    }


def write_summary(report: Dict[str, Any], summary_path: str) -> str:
    resolved = ensure_parent_directory(summary_path)
    lines = [
        f"Autobot dataset builder: {'PASS' if report.get('passed') else 'FAIL'}",
        f"Records: {report.get('record_count')}",
        f"Candidates: {report.get('candidate_count')}",
        f"Accepted: {report.get('accepted_count')}",
        f"Rejected: {report.get('rejected_count')}",
        f"Fixture request plan loaded: {report.get('fixture_request_plan_loaded')}",
        f"Fixture requests: {report.get('fixture_request_count')}",
        f"Collection targets: {report.get('collection_target_count')}",
        "Accepted material types:",
    ]
    for key, value in sorted(report.get("accepted_material_type_counts", {}).items()):
        lines.append(f"- {key}: {value}")
    lines.append("Curriculum distribution:")
    for key, value in sorted(report.get("curriculum_distribution", {}).items()):
        lines.append(f"- {key}: {value}")
    with open(resolved, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return resolved


def run_dataset_builder(
    records_path: str = DEFAULT_RECORDS_PATH,
    candidate_path: str = DEFAULT_CANDIDATE_PATH,
    rejected_path: str = DEFAULT_REJECTED_PATH,
    accepted_path: str = DEFAULT_ACCEPTED_PATH,
    curriculum_path: str = DEFAULT_CURRICULUM_PATH,
    report_path: str = DEFAULT_REPORT_PATH,
    summary_path: str = DEFAULT_SUMMARY_PATH,
    fixture_request_plan_path: str = DEFAULT_FIXTURE_REQUEST_PLAN_PATH,
    collection_targets_path: str = DEFAULT_COLLECTION_TARGETS_PATH,
    evaluation_gaps: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    records = read_jsonl(records_path)
    candidates = build_candidate_materials(records)
    split = split_accepted_rejected(candidates)
    accepted = split["accepted"]
    rejected = split["rejected"]
    fixture_request_plan = read_json(fixture_request_plan_path)
    merged_evaluation_gaps = derive_evaluation_gaps(evaluation_gaps or (), fixture_request_plan)
    manifest = build_curriculum_manifest(accepted, evaluation_gaps=merged_evaluation_gaps)
    collection_targets = build_collection_targets(
        accepted=accepted,
        fixture_request_plan=fixture_request_plan,
    )

    outputs: Dict[str, str] = {}
    outputs["candidate_materials"] = write_jsonl(candidate_path, candidates)
    outputs["rejected_materials"] = write_jsonl(rejected_path, rejected)
    outputs["accepted_materials"] = write_jsonl(accepted_path, accepted)
    outputs["curriculum_manifest"] = write_jsonl(curriculum_path, manifest)
    outputs["collection_targets"] = write_json(collection_targets_path, collection_targets)
    for material_type, output_path in TYPE_OUTPUTS.items():
        rows = [item for item in accepted if item.get("material_type") == material_type]
        outputs[material_type] = write_jsonl(output_path, rows)

    report = build_report(
        records,
        candidates,
        accepted,
        rejected,
        manifest,
        outputs,
        merged_evaluation_gaps,
        fixture_request_plan,
        collection_targets,
    )
    resolved_report = ensure_parent_directory(report_path)
    with open(resolved_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    outputs["report"] = resolved_report
    outputs["summary"] = write_summary(report, summary_path)
    report["outputs"] = outputs
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build source-aware autobot learning materials.")
    parser.add_argument("--records-path", default=DEFAULT_RECORDS_PATH)
    parser.add_argument("--candidate-path", default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--rejected-path", default=DEFAULT_REJECTED_PATH)
    parser.add_argument("--accepted-path", default=DEFAULT_ACCEPTED_PATH)
    parser.add_argument("--curriculum-path", default=DEFAULT_CURRICULUM_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--fixture-request-plan-path", default=DEFAULT_FIXTURE_REQUEST_PLAN_PATH)
    parser.add_argument("--collection-targets-path", default=DEFAULT_COLLECTION_TARGETS_PATH)
    parser.add_argument("--evaluation-gap", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_dataset_builder(
        records_path=args.records_path,
        candidate_path=args.candidate_path,
        rejected_path=args.rejected_path,
        accepted_path=args.accepted_path,
        curriculum_path=args.curriculum_path,
        report_path=args.report_path,
        summary_path=args.summary_path,
        fixture_request_plan_path=args.fixture_request_plan_path,
        collection_targets_path=args.collection_targets_path,
        evaluation_gaps=args.evaluation_gap or (),
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "record_count": report["record_count"],
                "accepted_count": report["accepted_count"],
                "rejected_count": report["rejected_count"],
                "report_path": os.path.abspath(args.report_path),
                "summary_path": os.path.abspath(args.summary_path),
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
