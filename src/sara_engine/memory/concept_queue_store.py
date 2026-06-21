from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence, Tuple

from sara_engine.memory.concept_admission import ConceptRevalidationEntry
from sara_engine.memory.concept_review_loop import ConceptReviewLoop, ConceptReviewLoopResult
from sara_engine.ingest import CandidateRelation
from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


QUEUE_SCHEMA = "sara-concept-revalidation-queue-v1"
REVIEW_REPORT_SCHEMA = "sara-concept-review-report-v1"


def default_queue_path() -> str:
    return workspace_path("memory", "concept_revalidation_queue.json")


def load_revalidation_queue(path: str | None = None) -> Tuple[ConceptRevalidationEntry, ...]:
    resolved = ensure_parent_directory(path or default_queue_path())
    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return ()
    if not isinstance(payload, Mapping):
        raise ValueError("revalidation queue file must contain a JSON object")
    if str(payload.get("schema", "")) != QUEUE_SCHEMA:
        raise ValueError("revalidation queue schema is not supported")
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("revalidation queue entries must be a list")
    return tuple(_entry_from_dict(item) for item in entries)


def save_revalidation_queue(
    entries: Sequence[ConceptRevalidationEntry],
    path: str | None = None,
) -> str:
    resolved = ensure_parent_directory(path or default_queue_path())
    payload = {
        "schema": QUEUE_SCHEMA,
        "entry_count": len(entries),
        "entries": [entry.to_dict() for entry in entries],
    }
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return resolved


def run_persisted_concept_review_cycle(
    relations: Sequence[CandidateRelation],
    *,
    current_segment: int,
    queue_path: str | None = None,
    report_path: str | None = None,
    loop: ConceptReviewLoop | None = None,
) -> ConceptReviewLoopResult:
    active_loop = loop or ConceptReviewLoop()
    loaded_queue = load_revalidation_queue(queue_path)
    result = active_loop.run(
        loaded_queue,
        relations,
        current_segment=int(current_segment),
    )
    resolved_queue_path = save_revalidation_queue(
        result.next_revalidation_queue,
        queue_path,
    )
    if report_path:
        save_review_report(
            result,
            queue_path=resolved_queue_path,
            report_path=report_path,
            current_segment=current_segment,
        )
    return result


def save_review_report(
    result: ConceptReviewLoopResult,
    *,
    queue_path: str,
    report_path: str,
    current_segment: int,
) -> str:
    resolved = ensure_parent_directory(report_path)
    payload: Dict[str, Any] = {
        "schema": REVIEW_REPORT_SCHEMA,
        "current_segment": int(current_segment),
        "queue_path": str(queue_path),
        "ready_count": len(result.schedule.ready_queue),
        "blocked_count": len(result.schedule.blocked_queue),
        "admitted_candidate_count": len(result.admission_plan.admitted_candidates),
        "revalidation_queue_count": len(result.next_revalidation_queue),
        "result": result.to_dict(),
    }
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return resolved


def _entry_from_dict(payload: Mapping[str, Any]) -> ConceptRevalidationEntry:
    return ConceptRevalidationEntry(
        concept_key=str(payload.get("concept_key", "")),
        decision=str(payload.get("decision", "")),
        supporting_relation_ids=tuple(
            str(item) for item in payload.get("supporting_relation_ids", ()) if str(item)
        ),
        source_refs=tuple(
            str(item) for item in payload.get("source_refs", ()) if str(item)
        ),
        source_hashes=tuple(
            str(item) for item in payload.get("source_hashes", ()) if str(item)
        ),
        revision_conflict_count=int(payload.get("revision_conflict_count", 0) or 0),
        contradiction_score=float(payload.get("contradiction_score", 0.0) or 0.0),
        next_action=str(payload.get("next_action", "")),
        attempt_count=int(payload.get("attempt_count", 0) or 0),
        blocked_at_segment=int(payload.get("blocked_at_segment", 0) or 0),
        last_review_segment=int(payload.get("last_review_segment", 0) or 0),
        retry_after_segment=int(payload.get("retry_after_segment", 0) or 0),
    )

