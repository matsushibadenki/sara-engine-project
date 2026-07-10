from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from sara_engine.dynamics import PersistentSelfStateController

from sara_engine.ingest.candidate_proposals import CandidateRelation
from sara_engine.ingest.frequent_sequence import FrequentSequence
from sara_engine.ingest.relation_stability import RelationStabilityAssessor
from sara_engine.memory.concept_admission import (
    ConceptAdmissionPlan,
    ConceptAdmissionPlanner,
    ConceptRevalidationEntry,
)
from sara_engine.memory.concept_revalidation import (
    ConceptRevalidationSchedule,
    ConceptRevalidationScheduler,
)


def _bundle_summary_from_candidates(candidates: Sequence[Any]) -> Dict[str, Any]:
    bundle_candidates = [
        candidate
        for candidate in candidates
        if str(getattr(candidate, "entry_id", "")).startswith("bundle:")
        or str(getattr(candidate, "own_latent_id", "")).startswith("bundle:")
    ]
    bundle_keys = tuple(
        sorted(
            {
                str(getattr(candidate, "own_latent_id", "") or getattr(candidate, "entry_id", ""))
                for candidate in bundle_candidates
                if str(getattr(candidate, "own_latent_id", "") or getattr(candidate, "entry_id", ""))
            }
        )
    )
    return {
        "bundle_candidate_count": len(bundle_candidates),
        "bundle_candidate_keys": list(bundle_keys),
        "bundle_candidate_ratio": float(len(bundle_candidates)) / float(max(1, len(candidates))),
    }


@dataclass(frozen=True)
class ConceptReviewLoopResult:
    schedule: ConceptRevalidationSchedule
    admission_plan: ConceptAdmissionPlan
    next_revalidation_queue: Tuple[ConceptRevalidationEntry, ...]
    schema: str = "sara-concept-review-loop-result-v1"

    def to_dict(self) -> Dict[str, Any]:
        bundle_summary = _bundle_summary_from_candidates(self.admission_plan.admitted_candidates)
        return {
            "schema": self.schema,
            "schedule": self.schedule.to_dict(),
            "admission_plan": self.admission_plan.to_dict(),
            "next_revalidation_queue": [item.to_dict() for item in self.next_revalidation_queue],
            "multimodal_bundle_summary": bundle_summary,
        }


class ConceptReviewLoop:
    """Runs the deterministic revalidation -> rebuild -> admission loop for concept candidates."""

    def __init__(
        self,
        *,
        scheduler: ConceptRevalidationScheduler | None = None,
        stability_assessor: RelationStabilityAssessor | None = None,
        admission_planner: ConceptAdmissionPlanner | None = None,
    ) -> None:
        self.scheduler = scheduler or ConceptRevalidationScheduler()
        self.stability_assessor = stability_assessor or RelationStabilityAssessor()
        self.admission_planner = admission_planner or ConceptAdmissionPlanner(
            guard=self.scheduler.guard
        )

    def run(
        self,
        queue_entries: Sequence[ConceptRevalidationEntry],
        relations: Sequence[CandidateRelation],
        *,
        current_segment: int,
        frequent_sequences: Sequence[FrequentSequence] = (),
        persistent_self_state: PersistentSelfStateController | None = None,
    ) -> ConceptReviewLoopResult:
        self_state_ids = (
            tuple(persistent_self_state.self_state_ids())
            if persistent_self_state is not None
            else ()
        )
        schedule = self.scheduler.build_schedule(
            queue_entries,
            relations,
            current_segment=current_segment,
            frequent_sequences=frequent_sequences,
            self_state_ids=self_state_ids,
        )
        entry_by_key = {entry.concept_key: entry for entry in queue_entries}
        ready_keys = {item.concept_key for item in schedule.ready_queue}

        rebuilt_candidates = [
            candidate
            for candidate in self.stability_assessor.crystallization_candidates(relations)
            if candidate.concept_key in ready_keys
        ]
        rebuilt_keys = {candidate.concept_key for candidate in rebuilt_candidates}
        rebuilt_relations = [
            relation
            for relation in relations
            if self._relation_key(relation) in rebuilt_keys
        ]

        if rebuilt_candidates:
            admission_plan = self.admission_planner.build_plan(
                rebuilt_relations,
                rebuilt_candidates,
                time_segment=current_segment,
                frequent_sequences=frequent_sequences,
            )
        else:
            admission_plan = ConceptAdmissionPlan(
                admitted_candidates=(),
                revalidation_queue=(),
                audits=(),
            )

        next_queue: List[ConceptRevalidationEntry] = []
        for decision in schedule.blocked_queue:
            original = entry_by_key.get(decision.concept_key)
            if original is None:
                continue
            next_queue.append(
                self._updated_blocked_entry(
                    original,
                    decision=decision,
                    current_segment=current_segment,
                )
            )

        for decision in schedule.ready_queue:
            if decision.concept_key in rebuilt_keys:
                continue
            original = entry_by_key.get(decision.concept_key)
            if original is None:
                continue
            next_queue.append(
                ConceptRevalidationEntry(
                    concept_key=original.concept_key,
                    decision="reject_missing_support",
                    supporting_relation_ids=original.supporting_relation_ids,
                    source_refs=original.source_refs,
                    source_hashes=original.source_hashes,
                    revision_conflict_count=original.revision_conflict_count,
                    contradiction_score=original.contradiction_score,
                    next_action="rebuild_supporting_relations",
                    attempt_count=original.attempt_count + 1,
                    blocked_at_segment=original.blocked_at_segment,
                    last_review_segment=int(current_segment),
                    retry_after_segment=int(current_segment) + self.scheduler.min_cooldown_segments,
                )
            )

        next_queue.extend(admission_plan.revalidation_queue)
        next_queue = self._rank_next_queue(
            next_queue,
            blocked_decisions=schedule.blocked_queue,
        )

        return ConceptReviewLoopResult(
            schedule=schedule,
            admission_plan=admission_plan,
            next_revalidation_queue=tuple(next_queue),
        )

    def _updated_blocked_entry(
        self,
        original: ConceptRevalidationEntry,
        *,
        decision: Any,
        current_segment: int,
    ) -> ConceptRevalidationEntry:
        next_decision = original.decision
        if str(decision.decision) == "blocked_attempt_budget":
            next_decision = "blocked_attempt_budget"
        return ConceptRevalidationEntry(
            concept_key=original.concept_key,
            decision=next_decision,
            supporting_relation_ids=original.supporting_relation_ids,
            source_refs=original.source_refs,
            source_hashes=original.source_hashes,
            revision_conflict_count=int(decision.revision_conflict_count),
            contradiction_score=float(decision.contradiction_score),
            next_action=str(decision.next_action),
            attempt_count=original.attempt_count,
            blocked_at_segment=original.blocked_at_segment,
            last_review_segment=int(current_segment),
            retry_after_segment=int(decision.retry_after_segment),
        )

    def _relation_key(self, relation: CandidateRelation) -> str:
        return f"{relation.relation}:{relation.source_event_id}->{relation.target_event_id}"

    def _rank_next_queue(
        self,
        entries: Sequence[ConceptRevalidationEntry],
        *,
        blocked_decisions: Sequence[Any],
    ) -> List[ConceptRevalidationEntry]:
        decision_by_key = {
            str(item.concept_key): item
            for item in blocked_decisions
        }
        return sorted(
            entries,
            key=lambda entry: self._queue_sort_key(
                entry,
                decision=decision_by_key.get(entry.concept_key),
            ),
        )

    def _queue_sort_key(
        self,
        entry: ConceptRevalidationEntry,
        *,
        decision: Any | None,
    ) -> tuple[float, float, int, str]:
        manual_review = float(
            bool(decision is not None and str(getattr(decision, "next_action", "")) == "manual_review")
            or str(entry.next_action) == "manual_review"
        )
        priority = float(getattr(decision, "priority_score", 0.0) or 0.0)
        contradiction = float(getattr(decision, "contradiction_score", entry.contradiction_score) or 0.0)
        retry_after = int(getattr(decision, "retry_after_segment", entry.retry_after_segment) or 0)
        return (
            -manual_review,
            -priority,
            retry_after,
            f"{contradiction:.6f}:{entry.concept_key}",
        )
