from sara_engine.memory.architecture_migration import (
    ArchitectureMigrationCoordinator,
    ArchitectureMigrationPolicy,
)
from sara_engine.memory.event_state_cache import (
    EventStateCandidate,
    VerifiedHierarchicalEventStateCache,
)


def _candidate(entry_id: str, **overrides) -> EventStateCandidate:
    values = {
        "entry_id": entry_id,
        "signature": (3, 5, 7),
        "source_ref": f"source:{entry_id}",
        "time_segment": 4,
        "own_latent_id": f"predicts:{entry_id}->outcome",
        "confidence": 0.9,
        "uncertainty": 0.1,
        "source_reliability": 0.9,
        "resonance_score": 0.9,
        "metabolic_headroom": 0.9,
        "observed": True,
        "source_backed": True,
        "verified": True,
        "architecture_version": "sara-architecture-v1",
    }
    values.update(overrides)
    return EventStateCandidate.from_verified_evidence(
        verifier_id="test-architecture-migration",
        evidence={"entry_id": entry_id, "signature": list(values["signature"])},
        **values,
    )


def test_architecture_migration_replays_verified_memory_without_mutating_legacy_cache():
    legacy = VerifiedHierarchicalEventStateCache()
    legacy.admit(_candidate("verified"))
    legacy_before = legacy.state_dict()
    target = VerifiedHierarchicalEventStateCache()
    coordinator = ArchitectureMigrationCoordinator(
        ArchitectureMigrationPolicy(
            source_architecture_version="sara-architecture-v1",
            target_architecture_version="sara-architecture-v2",
        )
    )

    result = coordinator.migrate(legacy, target)

    assert legacy.state_dict() == legacy_before
    assert result.to_dict()["legacy_cache_mutated"] is False
    assert result.to_dict()["admitted_count"] == 1
    migrated = target.entries["migration:sara-architecture-v2:verified"]
    assert migrated.architecture_version == "sara-architecture-v2"
    assert migrated.migration_source_architecture_version == "sara-architecture-v1"
    assert migrated.own_latent_id == "predicts:verified->outcome"


def test_architecture_migration_holds_incompatible_or_noncanonical_entries():
    legacy = VerifiedHierarchicalEventStateCache()
    legacy.admit(_candidate("wrong-version", architecture_version="sara-architecture-v0"))
    legacy.admit(_candidate("missing-key", own_latent_id=""))
    coordinator = ArchitectureMigrationCoordinator(
        ArchitectureMigrationPolicy(
            source_architecture_version="sara-architecture-v1",
            target_architecture_version="sara-architecture-v2",
        )
    )

    plan = coordinator.build_plan(legacy)

    assert plan.replay_candidates == ()
    assert {item.reason for item in plan.held_entries} == {
        "hold_architecture_version_mismatch",
        "hold_missing_canonical_concept_key",
    }


def test_event_state_cache_loads_pre_migration_state_with_default_architecture_version():
    cache = VerifiedHierarchicalEventStateCache()
    cache.admit(_candidate("legacy-shape"))
    state = cache.state_dict()
    state["entries"][0].pop("architecture_version")
    state["entries"][0].pop("migration_source_architecture_version")

    restored = VerifiedHierarchicalEventStateCache.from_state_dict(state)

    assert restored.entries["legacy-shape"].architecture_version == "sara-architecture-v1"
    assert restored.entries["legacy-shape"].migration_source_architecture_version == ""
