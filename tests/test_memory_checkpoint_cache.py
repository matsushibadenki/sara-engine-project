from __future__ import annotations

import copy

import pytest

from sara_engine.memory.memory_checkpoint_cache import (
    BoundedSparseMemoryCheckpointCache,
    MemoryCheckpointCandidate,
)


def _candidate(index: int, **overrides):
    values = {
        "event_start": index * 4,
        "event_end": index * 4 + 4,
        "summary_ids": (f"key-{index}", f"value-{index}"),
        "source_refs": (f"source:{index}",),
        "source_revision": "revision-v1",
        "state_group_id": "state:main",
        "parent_digest": f"parent-{index}",
        "runtime_fingerprint": "runtime-v1",
        "schema_fingerprint": "schema-v1",
    }
    values.update(overrides)
    return MemoryCheckpointCandidate(**values)


def _retrieve(cache, query, **overrides):
    values = {
        "source_revision": "revision-v1",
        "runtime_fingerprint": "runtime-v1",
        "schema_fingerprint": "schema-v1",
    }
    values.update(overrides)
    return cache.retrieve(query, **values)


def test_cache_is_default_off_and_never_mutates_durable_state():
    cache = BoundedSparseMemoryCheckpointCache()

    admission = cache.admit(_candidate(0))
    retrieval = _retrieve(cache, ("key-0",))

    assert admission.accepted is False
    assert admission.decision == "checkpoint_cache_disabled"
    assert retrieval.abstained is True
    assert retrieval.durable_mutation_allowed is False
    assert cache.state_dict()["checkpoints"] == []


def test_verified_semantic_checkpoint_returns_only_evidence_references():
    cache = BoundedSparseMemoryCheckpointCache(enabled=True)
    admission = cache.admit(_candidate(0), current_event=4)

    result = _retrieve(cache, ("key-0",), current_event=8)

    assert admission.accepted is True
    assert result.abstained is False
    assert result.decision == "retrieve_evidence_references"
    assert result.evidence[0]["source_refs"] == ["source:0"]
    assert result.evidence[0]["restores_mutable_state"] is False
    assert "state" not in result.evidence[0]


@pytest.mark.parametrize(
    ("overrides", "decision"),
    [
        ({"semantic_boundary": False}, "not_semantic_boundary"),
        ({"observed": False}, "unverified_checkpoint"),
        ({"verified": False}, "unverified_checkpoint"),
        ({"contradicted": True}, "contradicted_checkpoint"),
        ({"source_refs": ()}, "missing_checkpoint_provenance"),
        ({"event_start": 5, "event_end": 5}, "invalid_event_interval"),
        ({"summary_ids": tuple(f"id-{i}" for i in range(9))}, "summary_width_exceeded"),
    ],
)
def test_admission_rejects_untrusted_or_over_budget_checkpoint(overrides, decision):
    cache = BoundedSparseMemoryCheckpointCache(enabled=True)

    result = cache.admit(_candidate(0, **overrides))

    assert result.accepted is False
    assert result.decision == decision
    assert cache.checkpoints == ()


def test_sparse_selection_excludes_stale_checkpoint_and_uses_fixed_topk():
    cache = BoundedSparseMemoryCheckpointCache(enabled=True, selected_k=2)
    cache.admit(_candidate(0, summary_ids=("shared", "old")))
    cache.admit(_candidate(1, summary_ids=("shared", "new")))
    cache.admit(
        _candidate(
            2,
            summary_ids=("shared", "stale"),
            source_revision="revision-v0",
        )
    )

    result = _retrieve(cache, ("shared",))

    assert result.abstained is False
    assert len(result.selected_checkpoint_ids) == 2
    assert all(item["source_revision"] == "revision-v1" for item in result.evidence)
    assert result.scanned_checkpoints == 3


def test_only_stale_matching_checkpoint_rejects_instead_of_restoring():
    cache = BoundedSparseMemoryCheckpointCache(enabled=True)
    cache.admit(_candidate(0, source_revision="revision-v0"))

    result = _retrieve(cache, ("key-0",))

    assert result.abstained is True
    assert result.decision == "reject_stale_checkpoint"


def test_equal_retention_evicts_oldest_deterministically():
    cache = BoundedSparseMemoryCheckpointCache(
        enabled=True,
        retention_profile="equal",
        max_checkpoints=2,
    )
    first = cache.admit(_candidate(0))
    cache.admit(_candidate(1))
    third = cache.admit(_candidate(2))

    assert first.checkpoint_id in third.evicted_ids
    assert [item.event_start for item in cache.checkpoints] == [4, 8]
    assert cache.eviction_count == 1


def test_logarithmic_retention_merges_oldest_compatible_with_provenance():
    cache = BoundedSparseMemoryCheckpointCache(
        enabled=True,
        retention_profile="logarithmic",
        max_checkpoints=2,
        max_summary_ids=8,
    )
    first = cache.admit(_candidate(0))
    second = cache.admit(_candidate(1))
    third = cache.admit(_candidate(2))

    assert third.merged_ids == (first.checkpoint_id, second.checkpoint_id)
    assert cache.merge_count == 1
    assert cache.eviction_count == 0
    merged = cache.checkpoints[0]
    assert merged.event_start == 0
    assert merged.event_end == 8
    assert merged.parent_digests == ("parent-0", "parent-1")
    assert merged.source_refs == ("source:0", "source:1")
    assert merged.is_valid() is True


def test_expiry_and_revision_invalidation_are_deterministic():
    cache = BoundedSparseMemoryCheckpointCache(enabled=True)
    expiring = cache.admit(_candidate(0, expires_at=8))
    revised = cache.admit(_candidate(1, source_revision="revision-v2"))

    assert cache.expire(8) == (expiring.checkpoint_id,)
    assert cache.invalidate(source_revision="revision-v2") == (revised.checkpoint_id,)
    assert cache.checkpoints == ()


def test_state_round_trip_preserves_replay_and_rejects_tampering():
    cache = BoundedSparseMemoryCheckpointCache(enabled=True)
    cache.admit(_candidate(0))
    state = cache.state_dict()

    restored = BoundedSparseMemoryCheckpointCache.from_state_dict(state)
    assert restored.state_dict() == state
    assert _retrieve(restored, ("key-0",)).selected_checkpoint_ids == (
        cache.checkpoints[0].checkpoint_id,
    )

    tampered = copy.deepcopy(state)
    tampered["checkpoints"][0]["event_end"] = 99
    with pytest.raises(ValueError, match="identity_mismatch"):
        BoundedSparseMemoryCheckpointCache.from_state_dict(tampered)


def test_state_byte_budget_failure_rolls_back_without_eviction_or_merge():
    cache = BoundedSparseMemoryCheckpointCache(
        enabled=True,
        retention_profile="logarithmic",
        max_checkpoints=2,
        max_state_bytes=1,
    )

    result = cache.admit(_candidate(0))

    assert result.accepted is False
    assert result.decision == "state_byte_budget_exceeded"
    assert cache.checkpoints == ()
    assert cache.eviction_count == 0
    assert cache.merge_count == 0
