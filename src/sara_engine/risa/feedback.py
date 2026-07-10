from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from sara_engine.ingest.candidate_proposals import CandidateRelation, make_candidate_relation
from sara_engine.memory.concept_admission import ConceptRevalidationEntry

from .kernel import SARAAlignedRisaKernel


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class RisaFeedbackPackage:
    candidate_relations: Tuple[CandidateRelation, ...]
    revalidation_entries: Tuple[ConceptRevalidationEntry, ...]
    trace: Dict[str, Any]
    schema: str = "sara-risa-feedback-package-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "candidate_relations": [item.to_dict() for item in self.candidate_relations],
            "revalidation_entries": [item.to_dict() for item in self.revalidation_entries],
            "trace": dict(self.trace),
        }


def build_feedback_package(
    kernel: SARAAlignedRisaKernel,
    *,
    current_segment: int,
    min_support: int = 2,
    skip_dormant: bool = True,
) -> RisaFeedbackPackage:
    relation_records: List[CandidateRelation] = []
    queue_entries: List[ConceptRevalidationEntry] = []
    exported_concepts: List[str] = []

    for concept_id, lineage in sorted(kernel.state.concept_lineage.items()):
        node = kernel.state.graph.get_node(concept_id)
        if node is None:
            continue
        if skip_dormant and bool(node.dormant):
            continue
        support = int(lineage.get("support", 0) or 0)
        if support < max(1, int(min_support)):
            continue
        action = str(getattr(node, "attributes", {}).get("shared_action", "") or "")
        effect = str(getattr(node, "attributes", {}).get("shared_effect", "") or "")
        actors = tuple(str(item) for item in lineage.get("actors", []) or ())
        if not action or not effect or not actors:
            continue

        source_refs: List[str] = []
        source_hashes: List[str] = []
        relation_key = f"predicts:process:{action}->state:{effect}"
        confidence = _clamp01(max(float(node.stability), float(node.energy)))
        prediction_gain = round(_clamp01((float(node.stability) * 0.6) + (float(node.energy) * 0.4)), 6)

        for actor in sorted(set(actors)):
            source_ref = f"risa::{concept_id}::actor::{actor}"
            source_hash = f"risa-hash::{_stable_hash(source_ref)}"
            source_refs.append(source_ref)
            source_hashes.append(source_hash)
            relation_records.append(
                make_candidate_relation(
                    {
                        "record_id": f"risa-rel::{concept_id}::{actor}",
                        "relation": "predicts",
                        "source_event_id": f"process:{action}",
                        "target_event_id": f"state:{effect}",
                        "delay_lower_ms": 0,
                        "delay_upper_ms": 0,
                        "confidence": confidence,
                        "source_ref": source_ref,
                        "source_hash": source_hash,
                        "extractor_name": "risa_feedback",
                        "extractor_version": "v1",
                        "parent_ids": (concept_id,),
                        "observed_anchor_ids": (f"entity:{actor}", f"state:{effect}"),
                        "evidence_count": support,
                        "counterexample_count": 0,
                        "prediction_gain": prediction_gain,
                    }
                )
            )

        queue_entries.append(
            ConceptRevalidationEntry(
                concept_key=relation_key,
                decision="reject_missing_support",
                supporting_relation_ids=(relation_key,),
                source_refs=tuple(sorted(set(source_refs))),
                source_hashes=tuple(sorted(set(source_hashes))),
                revision_conflict_count=0,
                contradiction_score=0.0,
                next_action="rebuild_supporting_relations",
                attempt_count=0,
                blocked_at_segment=int(current_segment),
                last_review_segment=max(0, int(current_segment) - 1),
                retry_after_segment=int(current_segment),
            )
        )
        exported_concepts.append(concept_id)

    return RisaFeedbackPackage(
        candidate_relations=tuple(relation_records),
        revalidation_entries=tuple(queue_entries),
        trace={
            "exported_concept_ids": exported_concepts,
            "exported_relation_count": len(relation_records),
            "exported_queue_entry_count": len(queue_entries),
            "current_segment": int(current_segment),
            "skip_dormant": bool(skip_dormant),
            "min_support": int(min_support),
        },
    )


def merge_revalidation_entries(
    existing: Sequence[ConceptRevalidationEntry],
    incoming: Sequence[ConceptRevalidationEntry],
) -> Tuple[ConceptRevalidationEntry, ...]:
    merged: Dict[str, ConceptRevalidationEntry] = {
        entry.concept_key: entry for entry in existing
    }
    for entry in incoming:
        previous = merged.get(entry.concept_key)
        if previous is None:
            merged[entry.concept_key] = entry
            continue
        merged[entry.concept_key] = ConceptRevalidationEntry(
            concept_key=entry.concept_key,
            decision=previous.decision,
            supporting_relation_ids=tuple(
                sorted(set(previous.supporting_relation_ids) | set(entry.supporting_relation_ids))
            ),
            source_refs=tuple(sorted(set(previous.source_refs) | set(entry.source_refs))),
            source_hashes=tuple(sorted(set(previous.source_hashes) | set(entry.source_hashes))),
            revision_conflict_count=min(previous.revision_conflict_count, entry.revision_conflict_count),
            contradiction_score=min(previous.contradiction_score, entry.contradiction_score),
            next_action=previous.next_action or entry.next_action,
            attempt_count=min(previous.attempt_count, entry.attempt_count),
            blocked_at_segment=min(previous.blocked_at_segment, entry.blocked_at_segment),
            last_review_segment=max(previous.last_review_segment, entry.last_review_segment),
            retry_after_segment=min(previous.retry_after_segment, entry.retry_after_segment),
        )
    return tuple(sorted(merged.values(), key=lambda item: (item.retry_after_segment, item.concept_key)))
