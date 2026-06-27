from sara_engine.dynamics import PersistentSelfStateController, stable_self_state_id
from sara_engine.learning import IdleReplayConfig, plan_idle_replay
from sara_engine.learning.astro_modulator import AstroReplayModulator
from sara_engine.memory.event_state_cache import (
    EventStateCandidate,
    VerifiedHierarchicalEventStateCache,
)


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


def test_idle_replay_prefers_self_state_aligned_verified_memory():
    cache = VerifiedHierarchicalEventStateCache()
    cache.admit(
        _candidate(
            "plain",
            signature=(11, 13, 17),
            own_latent_id="latent:plain",
            source_ref="source:plain",
            sequence_support_score=0.2,
        )
    )
    cache.admit(
        _candidate(
            "aligned",
            signature=(21, 23, 27),
            own_latent_id="predicts:vision:visual_cluster_018->audio:audio_cluster_044",
            source_ref="source:aligned",
            sequence_support_score=0.2,
        )
    )
    controller = PersistentSelfStateController(core_event_ids=(101,))
    controller.step(external_event_ids=(stable_self_state_id("vision:visual_cluster_018"),))
    controller.step(external_event_ids=(stable_self_state_id("audio:audio_cluster_044"),))

    report = plan_idle_replay(cache, persistent_self_state=controller)

    assert report["observed_only"] is True
    assert report["selected"]
    assert report["selected"][0]["entry_id"] == "aligned"
    assert report["metrics"]["idle_replay_self_state_alignment_observed"] == 1.0


def test_idle_replay_uses_reactivation_hints_and_respects_budget():
    cache = VerifiedHierarchicalEventStateCache()
    cache.admit(
        _candidate(
            "anchor",
            signature=tuple(range(10)),
            source_ref="source:anchor",
            sequence_support_score=0.5,
            sequence_support_count=2,
        )
    )
    cache.admit(
        _candidate(
            "small",
            signature=(41, 43),
            source_ref="source:small",
            confidence=0.7,
        )
    )
    controller = PersistentSelfStateController(core_event_ids=(101,))

    report = plan_idle_replay(
        cache,
        persistent_self_state=controller,
        reactivation_hints=(
            {
                "entry_id": "anchor",
                "activation": 0.95,
                "mutates_durable_state": False,
            },
        ),
        config=IdleReplayConfig(max_candidates=2, event_budget=4, min_replay_score=0.3),
    )

    assert report["event_budget_ok"] is True
    assert all(item["entry_id"] != "anchor" for item in report["selected"])
    assert report["selected"][0]["entry_id"] == "small"
    assert report["metrics"]["idle_replay_memory_reactivation_observed"] == 1.0


def test_idle_replay_can_be_modulated_by_astro_state():
    cache = VerifiedHierarchicalEventStateCache()
    cache.admit(
        _candidate(
            "candidate",
            signature=(7, 11, 13),
            source_ref="source:candidate",
            sequence_support_score=0.6,
            sequence_support_count=3,
        )
    )
    modulator = AstroReplayModulator()
    modulator.update(interference_ratio=0.9, replay_recovery_signal=0.0)
    stressed = plan_idle_replay(cache, astro_modulator=modulator)

    calmer = AstroReplayModulator()
    calm = plan_idle_replay(cache, astro_modulator=calmer)

    assert stressed["candidates"][0]["replay_score"] < calm["candidates"][0]["replay_score"]


def test_idle_replay_reports_state_continuity_during_idle_period():
    cache = VerifiedHierarchicalEventStateCache()
    cache.admit(
        _candidate(
            "memory",
            signature=(31, 37, 41),
            source_ref="source:memory",
        )
    )
    controller = PersistentSelfStateController(core_event_ids=(101, 202))
    controller.step(external_event_ids=(101,))

    report = plan_idle_replay(cache, persistent_self_state=controller)

    assert report["self_state_trace"]["current_active_ids"]
    assert report["metrics"]["idle_replay_state_continuity_observed"] == 1.0
