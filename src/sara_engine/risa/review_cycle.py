from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

from sara_engine.dynamics import PersistentSelfStateController
from sara_engine.ingest.candidate_proposals import CandidateRelation
from sara_engine.ingest.frequent_sequence import FrequentSequence
from sara_engine.memory.concept_queue_store import (
    load_revalidation_queue,
    save_review_report,
    save_revalidation_queue,
)
from sara_engine.memory.concept_review_loop import ConceptReviewLoop, ConceptReviewLoopResult

from .feedback import RisaFeedbackPackage, build_feedback_package, merge_revalidation_entries
from .kernel import SARAAlignedRisaKernel


@dataclass(frozen=True)
class RisaReviewCycleResult:
    feedback_package: RisaFeedbackPackage
    merged_queue_path: str
    review_result: ConceptReviewLoopResult
    report_path: str | None
    schema: str = "sara-risa-review-cycle-result-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "feedback_package": self.feedback_package.to_dict(),
            "merged_queue_path": self.merged_queue_path,
            "report_path": self.report_path,
            "review_result": self.review_result.to_dict(),
        }


def run_risa_feedback_review_cycle(
    kernel: SARAAlignedRisaKernel,
    *,
    current_segment: int,
    queue_path: str | None = None,
    report_path: str | None = None,
    relations: Sequence[CandidateRelation] = (),
    frequent_sequences: Sequence[FrequentSequence] = (),
    persistent_self_state: PersistentSelfStateController | None = None,
    loop: ConceptReviewLoop | None = None,
    min_support: int = 2,
    skip_dormant: bool = True,
) -> RisaReviewCycleResult:
    feedback = build_feedback_package(
        kernel,
        current_segment=int(current_segment),
        min_support=int(min_support),
        skip_dormant=bool(skip_dormant),
    )
    existing_queue = load_revalidation_queue(queue_path)
    merged_queue = merge_revalidation_entries(
        existing_queue,
        feedback.revalidation_entries,
    )
    resolved_queue_path = save_revalidation_queue(merged_queue, queue_path)

    active_loop = loop or ConceptReviewLoop()
    review_relations = tuple(relations) + tuple(feedback.candidate_relations)
    review_result = active_loop.run(
        merged_queue,
        review_relations,
        current_segment=int(current_segment),
        frequent_sequences=frequent_sequences,
        persistent_self_state=persistent_self_state,
    )
    save_revalidation_queue(review_result.next_revalidation_queue, resolved_queue_path)
    resolved_report_path = None
    if report_path:
        resolved_report_path = save_review_report(
            review_result,
            queue_path=resolved_queue_path,
            report_path=report_path,
            current_segment=int(current_segment),
        )
    return RisaReviewCycleResult(
        feedback_package=feedback,
        merged_queue_path=resolved_queue_path,
        review_result=review_result,
        report_path=resolved_report_path,
    )
