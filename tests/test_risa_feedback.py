from sara_engine.risa import (
    RisaObservation,
    SARAAlignedRisaKernel,
    build_feedback_package,
    merge_revalidation_entries,
)
from sara_engine.memory.concept_admission import ConceptRevalidationEntry


def _toy_observations():
    return [
        RisaObservation(timestamp=1, event_id="e1", actor="dog", action="run", observed_effects=["fatigue_up"], verified=True, resonance_score=0.7),
        RisaObservation(timestamp=2, event_id="e2", actor="human", action="run", observed_effects=["fatigue_up"], verified=True, resonance_score=0.7),
        RisaObservation(timestamp=3, event_id="e3", actor="horse", action="run", observed_effects=["fatigue_up"], verified=True, resonance_score=0.8),
        RisaObservation(timestamp=4, event_id="e4", actor="dog", action="rest", observed_effects=["fatigue_down"], verified=True, resonance_score=0.6),
    ]


def test_risa_feedback_exports_candidate_relations_and_queue_entries() -> None:
    kernel = SARAAlignedRisaKernel(min_support=2, min_distinct_actors=2)
    kernel.ingest_observations(_toy_observations())

    package = build_feedback_package(kernel, current_segment=9)

    assert package.trace["exported_queue_entry_count"] >= 1
    assert package.trace["exported_relation_count"] >= 3
    first_entry = package.revalidation_entries[0]
    assert first_entry.concept_key == "predicts:process:run->state:fatigue_up"
    assert first_entry.decision == "reject_missing_support"
    assert first_entry.next_action == "rebuild_supporting_relations"
    source_refs = {relation.lineage.source_ref for relation in package.candidate_relations}
    assert any("risa::concept:shared_run_fatigue_up::actor::dog" == item for item in source_refs)


def test_risa_feedback_skips_dormant_concepts_when_requested() -> None:
    kernel = SARAAlignedRisaKernel(
        min_support=2,
        min_distinct_actors=2,
        dormancy_energy_threshold=0.2,
        dormancy_idle_threshold=4,
        connection_cost_rate=0.05,
    )
    kernel.ingest_observations(_toy_observations()[:3])
    kernel.apply_metabolism(current_timestamp=20)

    package = build_feedback_package(kernel, current_segment=20, skip_dormant=True)

    assert package.candidate_relations == ()
    assert package.revalidation_entries == ()


def test_risa_feedback_can_merge_into_existing_revalidation_queue() -> None:
    existing = (
        ConceptRevalidationEntry(
            concept_key="predicts:process:run->state:fatigue_up",
            decision="reject_missing_support",
            supporting_relation_ids=("predicts:process:run->state:fatigue_up",),
            source_refs=("legacy-source",),
            source_hashes=("legacy-hash",),
            revision_conflict_count=1,
            contradiction_score=0.2,
            next_action="rebuild_supporting_relations",
            attempt_count=1,
            blocked_at_segment=3,
            last_review_segment=4,
            retry_after_segment=5,
        ),
    )
    kernel = SARAAlignedRisaKernel(min_support=2, min_distinct_actors=2)
    kernel.ingest_observations(_toy_observations())
    package = build_feedback_package(kernel, current_segment=9)

    merged = merge_revalidation_entries(existing, package.revalidation_entries)

    assert len(merged) >= 1
    merged_entry = next(item for item in merged if item.concept_key == "predicts:process:run->state:fatigue_up")
    assert "legacy-source" in merged_entry.source_refs
    assert any(item.startswith("risa::concept:shared_run_fatigue_up::actor::") for item in merged_entry.source_refs)
    assert merged_entry.retry_after_segment == 5
