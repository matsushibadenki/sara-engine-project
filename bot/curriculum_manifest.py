from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence


REPAIR_TYPES = {"negative_query", "contrastive_pair", "counterexample", "repair_note", "revision_note"}
REPLAY_TYPES = {"source_claim", "summary", "transcript_segment"}


def assign_curriculum_stage(material: Dict[str, object], evaluation_gaps: Sequence[str] = ()) -> str:
    material_type = str(material.get("material_type", ""))
    quality = float(material.get("quality_score", 0.5) or 0.5)
    gaps = {str(item) for item in evaluation_gaps}
    if material_type == "negative_query" and "negative_control" in gaps:
        return "repair"
    if material_type == "contrastive_pair" and "contrastive_control" in gaps:
        return "repair"
    if material_type in {"counterexample", "repair_note", "revision_note"}:
        return "repair"
    if material_type == "transcript_segment" and "retrieval_grounding" in gaps:
        return "replay"
    if material_type in REPLAY_TYPES and quality >= 0.75:
        return "replay"
    if material_type in {"definition_card", "summary"} or quality < 0.45:
        return "easy"
    if material_type in {"qa_pair", "source_claim"}:
        return "medium"
    return "hard"


def compute_material_priority(material: Dict[str, object], evaluation_gaps: Sequence[str] = ()) -> float:
    quality = max(0.0, min(1.0, float(material.get("quality_score", 0.5) or 0.5)))
    source_type = str(material.get("source_type", "unknown"))
    material_type = str(material.get("material_type", ""))
    source_bonus = 0.12 if source_type in {"hot_inbox", "official_docs", "offline_batch"} else 0.04
    gap_bonus = 0.0
    gaps = {str(item) for item in evaluation_gaps}
    if material_type == "negative_query" and "negative_control" in gaps:
        gap_bonus += 0.22
    if material_type == "contrastive_pair" and "contrastive_control" in gaps:
        gap_bonus += 0.22
    if material_type == "summary" and "summary_coverage" in gaps:
        gap_bonus += 0.18
    if material_type == "qa_pair" and "retrieval_grounding" in gaps:
        gap_bonus += 0.18
    if material_type == "transcript_segment" and "retrieval_grounding" in gaps:
        gap_bonus += 0.22
    if material_type in {"counterexample", "repair_note", "revision_note"}:
        gap_bonus += 0.22
    rarity_bonus = 0.08 if material_type in REPAIR_TYPES else 0.02
    return round(max(0.0, min(1.5, quality * 0.7 + source_bonus + gap_bonus + rarity_bonus)), 4)


def build_curriculum_manifest(
    materials: Iterable[Dict[str, object]],
    evaluation_gaps: Sequence[str] = (),
) -> List[Dict[str, object]]:
    manifest: List[Dict[str, object]] = []
    for index, material in enumerate(materials):
        item = dict(material)
        stage = assign_curriculum_stage(item, evaluation_gaps=evaluation_gaps)
        priority = compute_material_priority(item, evaluation_gaps=evaluation_gaps)
        manifest.append(
            {
                "manifest_id": f"autobot-material-{index:06d}",
                "material_hash": item.get("material_hash", ""),
                "material_type": item.get("material_type", ""),
                "curriculum_stage": stage,
                "priority": priority,
                "source": item.get("source", ""),
                "source_url": item.get("source_url", ""),
                "source_path": item.get("source_path", ""),
                "language": item.get("language", "unknown"),
                "quality_score": float(item.get("quality_score", 0.0) or 0.0),
                "license_hint": item.get("license_hint", ""),
                "compliance_level": item.get("compliance_level", ""),
            }
        )
    return manifest


def summarize_curriculum(manifest: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, int]]:
    stage_counts = Counter(str(item.get("curriculum_stage", "unknown")) for item in manifest)
    type_counts = Counter(str(item.get("material_type", "unknown")) for item in manifest)
    return {
        "curriculum_distribution": dict(sorted(stage_counts.items())),
        "material_type_counts": dict(sorted(type_counts.items())),
    }
