import json
import os

from sara_engine.risa import (
    RisaObservation,
    SARAAlignedRisaKernel,
    run_risa_feedback_review_cycle,
)
from sara_engine.memory.concept_admission import ConceptRevalidationEntry
from sara_engine.memory.concept_queue_store import (
    load_revalidation_queue,
    save_revalidation_queue,
)
from sara_engine.utils.project_paths import workspace_path


def _toy_observations():
    return [
        RisaObservation(timestamp=1, event_id="e1", actor="dog", action="run", observed_effects=["fatigue_up"], verified=True, resonance_score=0.7),
        RisaObservation(timestamp=2, event_id="e2", actor="human", action="run", observed_effects=["fatigue_up"], verified=True, resonance_score=0.7),
        RisaObservation(timestamp=3, event_id="e3", actor="horse", action="run", observed_effects=["fatigue_up"], verified=True, resonance_score=0.8),
    ]


def test_risa_review_cycle_merges_queue_runs_review_and_persists_outputs() -> None:
    kernel = SARAAlignedRisaKernel(min_support=2, min_distinct_actors=2)
    kernel.ingest_observations(_toy_observations())

    queue_path = workspace_path("test_risa_review_cycle_queue.json")
    report_path = workspace_path("test_risa_review_cycle_report.json")
    save_revalidation_queue(
        (
            ConceptRevalidationEntry(
                concept_key="predicts:process:run->state:fatigue_up",
                decision="reject_missing_support",
                supporting_relation_ids=("predicts:process:run->state:fatigue_up",),
                source_refs=("legacy-source",),
                source_hashes=("legacy-hash",),
                revision_conflict_count=0,
                contradiction_score=0.0,
                next_action="rebuild_supporting_relations",
                attempt_count=0,
                blocked_at_segment=2,
                last_review_segment=2,
                retry_after_segment=2,
            ),
        ),
        queue_path,
    )

    result = run_risa_feedback_review_cycle(
        kernel,
        current_segment=9,
        queue_path=queue_path,
        report_path=report_path,
    )

    assert result.feedback_package.trace["exported_relation_count"] >= 3
    assert len(result.review_result.admission_plan.admitted_candidates) == 1
    assert result.review_result.admission_plan.admitted_candidates[0].own_latent_id == "predicts:process:run->state:fatigue_up"

    stored_queue = load_revalidation_queue(queue_path)
    assert stored_queue == ()

    with open(report_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["ready_count"] == 1
    assert payload["admitted_candidate_count"] == 1
    assert payload["result"]["admission_plan"]["admitted_candidates"]


def test_risa_review_cycle_keeps_queue_when_concepts_are_dormant() -> None:
    kernel = SARAAlignedRisaKernel(
        min_support=2,
        min_distinct_actors=2,
        dormancy_energy_threshold=0.2,
        dormancy_idle_threshold=4,
        connection_cost_rate=0.05,
    )
    kernel.ingest_observations(_toy_observations())
    kernel.apply_metabolism(current_timestamp=20)

    queue_path = workspace_path("test_risa_review_cycle_dormant_queue.json")
    initial = (
        ConceptRevalidationEntry(
            concept_key="predicts:legacy->legacy",
            decision="reject_missing_support",
            supporting_relation_ids=("predicts:legacy->legacy",),
            source_refs=("legacy-source",),
            source_hashes=("legacy-hash",),
            revision_conflict_count=0,
            contradiction_score=0.0,
            next_action="rebuild_supporting_relations",
            attempt_count=0,
            blocked_at_segment=2,
            last_review_segment=2,
            retry_after_segment=2,
        ),
    )
    save_revalidation_queue(initial, queue_path)

    result = run_risa_feedback_review_cycle(
        kernel,
        current_segment=20,
        queue_path=queue_path,
        skip_dormant=True,
    )

    assert result.feedback_package.candidate_relations == ()
    stored_queue = load_revalidation_queue(queue_path)
    assert len(stored_queue) == 1
    assert stored_queue[0].concept_key == "predicts:legacy->legacy"
