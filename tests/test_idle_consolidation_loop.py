from sara_engine.dynamics import PersistentSelfStateController, stable_self_state_id
from sara_engine.ingest import FrequentSequence, make_candidate_relation
from sara_engine.learning.astro_modulator import AstroReplayModulator
from sara_engine.learning.idle_replay import IdleReplayConfig
from sara_engine.learning.sleep_consolidation import SleepConsolidationConfig
from sara_engine.memory.concept_admission import ConceptRevalidationEntry
from sara_engine.memory.event_state_cache import (
    EventStateCandidate,
    VerifiedHierarchicalEventStateCache,
)
from sara_engine.memory.idle_consolidation_loop import IdleConsolidationLoop


def _relation(
    *,
    record_id: str,
    source_ref: str,
    source_hash: str,
    source_event_id: str = "vision:visual_cluster_018",
    target_event_id: str = "audio:audio_cluster_044",
    evidence_count: int = 5,
    counterexample_count: int = 0,
    prediction_gain: float = 0.18,
):
    return make_candidate_relation(
        {
            "record_id": record_id,
            "relation": "predicts",
            "source_event_id": source_event_id,
            "target_event_id": target_event_id,
            "delay_lower_ms": 60,
            "delay_upper_ms": 140,
            "confidence": 0.88,
            "source_ref": source_ref,
            "source_hash": source_hash,
            "extractor_name": "prediction_gain",
            "extractor_version": "v1",
            "evidence_count": evidence_count,
            "counterexample_count": counterexample_count,
            "prediction_gain": prediction_gain,
        }
    )


def _entry(decision: str, **overrides):
    values = {
        "concept_key": "predicts:vision:visual_cluster_018->audio:audio_cluster_044",
        "decision": decision,
        "supporting_relation_ids": ("predicts:vision:visual_cluster_018->audio:audio_cluster_044",),
        "source_refs": ("episode-1",),
        "source_hashes": ("hash-a",),
        "revision_conflict_count": 1,
        "contradiction_score": 0.2,
        "next_action": "wait",
        "attempt_count": 0,
        "blocked_at_segment": 3,
        "last_review_segment": 3,
        "retry_after_segment": 4,
    }
    values.update(overrides)
    return ConceptRevalidationEntry(**values)


def _candidate(entry_id: str, **overrides) -> EventStateCandidate:
    values = {
        "entry_id": entry_id,
        "signature": (1, 3, 5),
        "source_ref": f"source:{entry_id}",
        "time_segment": 1,
        "own_latent_id": f"latent:{entry_id}",
        "confidence": 0.9,
        "uncertainty": 0.1,
        "source_reliability": 0.9,
        "resonance_score": 0.9,
        "sequence_support_score": 0.0,
        "sequence_support_count": 0,
        "metabolic_headroom": 0.8,
        "observed": True,
        "source_backed": True,
        "verified": True,
    }
    values.update(overrides)
    return EventStateCandidate(**values)


def test_idle_consolidation_loop_connects_replay_sleep_and_concept_review():
    cache = VerifiedHierarchicalEventStateCache()
    cache.admit(
        _candidate(
            "concept-memory",
            signature=(21, 23, 27),
            own_latent_id="predicts:vision:visual_cluster_018->audio:audio_cluster_044",
            source_ref="concept:aligned",
            sequence_support_score=0.4,
            sequence_support_count=2,
        )
    )
    queue = [_entry("quarantine_source_revision_conflict")]
    relations = [
        _relation(record_id="rel-1", source_ref="episode-1", source_hash="hash-a"),
        _relation(record_id="rel-2", source_ref="episode-2", source_hash="hash-b"),
    ]
    sequences = [
        FrequentSequence(
            sequence_key="visual_cluster_018 -> audio_cluster_044",
            labels=("visual_cluster_018", "audio_cluster_044"),
            support_episode_count=2,
            occurrence_count=2,
            source_count=2,
            mean_span_ms=50.0,
            parent_episode_ids=("episode-1", "episode-2"),
        )
    ]
    controller = PersistentSelfStateController(
        core_event_ids=(
            stable_self_state_id("vision:visual_cluster_018"),
            stable_self_state_id("audio:audio_cluster_044"),
        )
    )

    result = IdleConsolidationLoop().run(
        cache,
        queue,
        relations,
        current_segment=6,
        frequent_sequences=sequences,
        persistent_self_state=controller,
        replay_config=IdleReplayConfig(max_candidates=2, event_budget=12, min_replay_score=0.3),
        sleep_config=SleepConsolidationConfig(event_budget=12.0),
    )

    assert result.idle_replay_report["selected"]
    assert result.prioritized_concept_keys == (
        "predicts:vision:visual_cluster_018->audio:audio_cluster_044",
    )
    assert len(result.concept_review_result.admission_plan.admitted_candidates) == 1
    assert result.sleep_consolidation_report["event_budget_ok"] is True
    assert result.sleep_consolidation_report["metrics"]["sleep_consolidation_retention_observed"] == 1.0
    assert result.memory_phase_report["observed_only"] is True
    assert result.memory_phase_report["phase_tracks"][0]["final_phase"] in {"glass", "crystal"}
    assert result.delta_retention_policy_report["observed_only"] is True
    assert "delta_memory_policy_state_budget_observed" in result.delta_retention_policy_report["metrics"]
    assert result.cache_refresh
    assert result.cache_refresh[0]["new_utility"] >= result.cache_refresh[0]["previous_utility"]


def test_idle_consolidation_loop_keeps_nonprioritized_queue_entries_after_replay():
    cache = VerifiedHierarchicalEventStateCache()
    cache.admit(_candidate("plain-memory", signature=(7, 11, 13), own_latent_id="latent:plain"))
    queue = [
        _entry(
            "reject_insufficient_source_diversity",
            concept_key="predicts:other_source->other_target",
            retry_after_segment=8,
            last_review_segment=7,
        )
    ]

    result = IdleConsolidationLoop().run(
        cache,
        queue,
        [],
        current_segment=7,
        replay_config=IdleReplayConfig(max_candidates=1, event_budget=8, min_replay_score=0.2),
        sleep_config=SleepConsolidationConfig(event_budget=8.0),
    )

    assert result.prioritized_concept_keys == ()
    assert len(result.concept_review_result.next_revalidation_queue) == 1
    assert result.concept_review_result.next_revalidation_queue[0].concept_key == "predicts:other_source->other_target"


def test_idle_consolidation_loop_reflects_astro_pressure_in_sleep_report():
    cache = VerifiedHierarchicalEventStateCache()
    cache.admit(
        _candidate(
            "candidate",
            signature=(31, 37, 41),
            own_latent_id="latent:candidate",
            sequence_support_score=0.5,
            sequence_support_count=3,
        )
    )
    stressed_modulator = AstroReplayModulator()
    stressed_modulator.update(interference_ratio=0.9, replay_recovery_signal=0.0)
    calm_modulator = AstroReplayModulator()

    stressed = IdleConsolidationLoop().run(
        cache,
        [],
        [],
        current_segment=5,
        astro_modulator=stressed_modulator,
        replay_config=IdleReplayConfig(max_candidates=1, event_budget=8, min_replay_score=0.2),
        sleep_config=SleepConsolidationConfig(event_budget=8.0),
    )
    calm = IdleConsolidationLoop().run(
        cache,
        [],
        [],
        current_segment=5,
        astro_modulator=calm_modulator,
        replay_config=IdleReplayConfig(max_candidates=1, event_budget=8, min_replay_score=0.2),
        sleep_config=SleepConsolidationConfig(event_budget=8.0),
    )

    assert stressed.idle_replay_report["selected"][0]["replay_score"] < calm.idle_replay_report["selected"][0]["replay_score"]
    assert stressed.sleep_consolidation_report["traces"][0]["post_retention"] < calm.sleep_consolidation_report["traces"][0]["post_retention"]
    assert stressed.memory_phase_report["phase_tracks"][0]["final_phase"] in {"liquid", "glass", "crystal"}
    assert stressed.delta_retention_policy_report["observed_only"] is True


def test_idle_consolidation_loop_can_skip_cache_refresh():
    cache = VerifiedHierarchicalEventStateCache()
    cache.admit(_candidate("memory", signature=(5, 7, 11), own_latent_id="latent:memory"))
    original_utility = cache.entries["memory"].utility

    result = IdleConsolidationLoop().run(
        cache,
        [],
        [],
        current_segment=4,
        replay_config=IdleReplayConfig(max_candidates=1, event_budget=8, min_replay_score=0.2),
        sleep_config=SleepConsolidationConfig(event_budget=8.0),
        apply_cache_refresh=False,
    )

    assert result.cache_refresh == ()
    assert cache.entries["memory"].utility == original_utility
    assert result.delta_retention_policy_report["observed_only"] is True
