from sara_engine.memory.event_state_cache import (
    EventStateCandidate,
    VerifiedHierarchicalEventStateCache,
)
from sara_engine.dynamics import stable_self_state_id
from sara_engine.memory.verification_receipt import issue_verification_receipt


def _candidate(entry_id: str, **overrides):
    values = {
        "entry_id": entry_id,
        "signature": (1, 3, 5),
        "source_ref": f"source:{entry_id}",
        "time_segment": 1,
        "own_latent_id": "latent:memory",
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
    if "verification_receipt" not in overrides:
        values["verification_receipt"] = issue_verification_receipt(
            verifier_id="test-event-state-candidate",
            verifier_version="v1",
            decision="verified_test_fixture",
            evidence={"entry_id": values["entry_id"], "signature": list(values["signature"])},
            source_refs=(values["source_ref"],),
            source_revision=values.get("source_revision", ""),
            observed=values["observed"],
            source_backed=values["source_backed"],
            verified=values["verified"],
            contradicted=values.get("contradicted", False),
            abstained=values.get("abstained", False),
        )
    return EventStateCandidate(**values)


def test_cache_admits_and_retrieves_verified_sparse_state():
    cache = VerifiedHierarchicalEventStateCache(retention_profile="logarithmic")

    admission = cache.admit(_candidate("verified"))
    retrieval = cache.retrieve((1, 3, 5), own_latent_id="latent:memory")

    assert admission.accepted is True
    assert admission.tier == "durable"
    assert retrieval.abstained is False
    assert retrieval.matches[0]["entry_id"] == "verified"
    assert retrieval.scanned_entries == 1


def test_cache_blocks_unverified_predicted_and_contradicted_states():
    cache = VerifiedHierarchicalEventStateCache()

    unverified = cache.admit(_candidate("unverified", source_backed=False))
    predicted = cache.admit(_candidate("predicted", observed=False))
    contradicted = cache.admit(_candidate("contradicted", contradicted=True))

    assert unverified.decision == "block_unverified_source"
    assert predicted.decision == "block_predicted_only"
    assert contradicted.decision == "block_contradiction"
    assert cache.state_dict()["entry_count"] == 0


def test_cache_merges_verified_duplicates_without_state_growth():
    cache = VerifiedHierarchicalEventStateCache()

    first = cache.admit(_candidate("first"))
    second = cache.admit(
        _candidate(
            "second",
            signature=(1, 3, 5, 7),
            confidence=0.95,
        )
    )

    assert first.accepted is True
    assert second.decision == "merge_verified_duplicate"
    assert cache.state_dict()["entry_count"] == 1
    assert cache.state_dict()["merge_count"] == 1


def test_logarithmic_cache_is_bounded_and_preserves_high_utility_state():
    cache = VerifiedHierarchicalEventStateCache(
        retention_profile="logarithmic",
        max_entries=12,
    )
    cache.admit(_candidate("target", signature=(11, 13, 17), resonance_score=0.98))
    for index in range(20):
        cache.admit(
            _candidate(
                f"distractor-{index}",
                signature=(100 + index, 200 + index),
                own_latent_id=f"latent:{index}",
                time_segment=index + 2,
                resonance_score=0.68,
                confidence=0.65,
                source_reliability=0.7,
                uncertainty=0.3,
            )
        )

    state = cache.state_dict()
    retrieval = cache.retrieve((11, 13, 17))

    assert state["entry_count"] <= 8
    assert state["eviction_count"] > 0
    assert retrieval.matches[0]["entry_id"] == "target"


def test_cache_expires_entries_and_abstains_on_weak_query():
    cache = VerifiedHierarchicalEventStateCache()
    cache.admit(_candidate("expiring", expires_at=5))

    expired = cache.expire(5)
    retrieval = cache.retrieve((99, 101))

    assert expired == ["expiring"]
    assert retrieval.abstained is True
    assert retrieval.decision == "abstain_insufficient_evidence"


def test_cache_state_round_trip_preserves_retrieval_and_source_revision():
    cache = VerifiedHierarchicalEventStateCache()
    cache.admit(_candidate("round-trip", source_revision="hash-v1", sequence_support_score=0.42, sequence_support_count=2))

    restored = VerifiedHierarchicalEventStateCache.from_state_dict(
        cache.state_dict()
    )
    result = restored.retrieve((1, 3, 5))

    assert result.matches[0]["entry_id"] == "round-trip"
    assert restored.state_dict()["entries"][0]["source_revision"] == "hash-v1"
    assert restored.state_dict()["entries"][0]["sequence_support_score"] == 0.42
    assert restored.state_dict()["entries"][0]["sequence_support_count"] == 2
    assert result.reactivation_hints[0]["mutates_durable_state"] is False


def test_cache_state_rejects_bad_schema_and_unverified_entry():
    cache = VerifiedHierarchicalEventStateCache()
    cache.admit(_candidate("valid"))
    bad_schema = cache.state_dict()
    bad_schema["schema"] = "unknown"

    try:
        VerifiedHierarchicalEventStateCache.from_state_dict(bad_schema)
    except ValueError as exc:
        assert "schema" in str(exc)
    else:
        raise AssertionError("bad schema must be rejected")

    unverified = cache.state_dict()
    unverified["entries"][0]["verified"] = False
    try:
        VerifiedHierarchicalEventStateCache.from_state_dict(unverified)
    except ValueError as exc:
        assert "observed and verified" in str(exc)
    else:
        raise AssertionError("unverified durable state must be rejected")


def test_cache_retrieval_exposes_sequence_support_component():
    cache = VerifiedHierarchicalEventStateCache(retrieval_threshold=0.1)
    cache.admit(_candidate("supported", sequence_support_score=0.7, sequence_support_count=3))

    result = cache.retrieve((1, 3, 5), own_latent_id="latent:memory")

    assert result.abstained is False
    assert result.matches[0]["components"]["sequence_support"] == 0.7


def test_cache_credit_support_improves_utility_for_equal_candidates():
    cache = VerifiedHierarchicalEventStateCache(retention_profile="logarithmic", max_entries=1)
    cache.admit(
        _candidate(
            "plain-credit",
            signature=(81, 83, 89),
            own_latent_id="latent:plain-credit",
            source_ref="source:plain-credit",
            resonance_score=0.82,
            confidence=0.82,
            source_reliability=0.82,
            uncertainty=0.18,
            sequence_support_score=0.0,
            credit_score=0.0,
        )
    )
    cache.admit(
        _candidate(
            "supported-credit",
            signature=(91, 97, 101),
            own_latent_id="latent:supported-credit",
            source_ref="source:supported-credit",
            resonance_score=0.82,
            confidence=0.82,
            source_reliability=0.82,
            uncertainty=0.18,
            sequence_support_score=0.0,
            credit_score=0.9,
            credit_responsibility=0.92,
            credit_confidence=0.88,
            credit_longevity=0.84,
        )
    )

    state = cache.state_dict()

    assert state["entry_count"] == 1
    assert state["entries"][0]["entry_id"] == "supported-credit"
    assert state["entries"][0]["credit_score"] == 0.9


def test_cache_multimodal_bundle_affinity_improves_utility_for_equal_candidates():
    cache = VerifiedHierarchicalEventStateCache(retention_profile="logarithmic", max_entries=1)
    cache.admit(
        _candidate(
            "plain-affinity",
            signature=(41, 43, 47),
            own_latent_id="latent:plain-affinity",
            source_ref="source:plain-affinity",
            resonance_score=0.82,
            confidence=0.82,
            source_reliability=0.82,
            uncertainty=0.18,
            credit_score=0.0,
            credit_longevity=0.58,
        )
    )
    cache.admit(
        _candidate(
            "bundle:0:123456",
            signature=(51, 53, 59),
            own_latent_id="bundle:0:123456",
            source_ref="bundle::fixture-supported",
            resonance_score=0.82,
            confidence=0.82,
            source_reliability=0.82,
            uncertainty=0.18,
            credit_score=0.0,
            credit_longevity=0.58,
        )
    )

    state = cache.state_dict()

    assert state["entry_count"] == 1
    assert state["entries"][0]["entry_id"] == "bundle:0:123456"


def test_cache_sequence_support_improves_utility_for_equal_candidates():
    cache = VerifiedHierarchicalEventStateCache(retention_profile="logarithmic", max_entries=1)
    cache.admit(
        _candidate(
            "plain",
            signature=(11, 13, 17),
            own_latent_id="latent:plain",
            source_ref="source:plain",
            resonance_score=0.82,
            confidence=0.82,
            source_reliability=0.82,
            uncertainty=0.18,
            sequence_support_score=0.0,
        )
    )
    cache.admit(
        _candidate(
            "supported",
            signature=(21, 23, 27),
            own_latent_id="latent:supported",
            source_ref="source:supported",
            resonance_score=0.82,
            confidence=0.82,
            source_reliability=0.82,
            uncertainty=0.18,
            sequence_support_score=0.8,
            sequence_support_count=3,
        )
    )

    state = cache.state_dict()

    assert state["entry_count"] == 1
    assert state["entries"][0]["entry_id"] == "supported"


def test_cache_retrieval_exposes_self_state_alignment_component():
    cache = VerifiedHierarchicalEventStateCache(retrieval_threshold=0.1)
    cache.admit(
        _candidate(
            "aligned",
            own_latent_id="predicts:vision:visual_cluster_018->audio:audio_cluster_044",
            source_ref="concept:aligned",
        )
    )

    result = cache.retrieve(
        (1, 3, 5),
        self_state_ids=(
            stable_self_state_id("vision:visual_cluster_018"),
            stable_self_state_id("audio:audio_cluster_044"),
        ),
    )

    assert result.abstained is False
    assert result.matches[0]["components"]["self_state_alignment"] == 1.0


def test_cache_retrieval_exposes_multimodal_bundle_affinity_component():
    cache = VerifiedHierarchicalEventStateCache(retrieval_threshold=0.1)
    cache.admit(
        _candidate(
            "bundle:0:123456",
            source_ref="bundle::fixture-supported",
            own_latent_id="bundle:0:123456",
        )
    )

    result = cache.retrieve((1, 3, 5), own_latent_id="bundle:0:123456")

    assert result.abstained is False
    assert result.matches[0]["components"]["multimodal_bundle_affinity"] == 1.0


def test_cache_self_state_alignment_breaks_tie_between_equal_entries():
    cache = VerifiedHierarchicalEventStateCache(retrieval_threshold=0.1, top_k=1)
    cache.admit(
        _candidate(
            "plain",
            signature=(41, 43, 47),
            own_latent_id="latent:plain",
            source_ref="source:plain",
            confidence=0.8,
            source_reliability=0.8,
            resonance_score=0.8,
            uncertainty=0.2,
            sequence_support_score=0.0,
        )
    )
    cache.admit(
        _candidate(
            "aligned",
            signature=(41, 43, 47),
            own_latent_id="predicts:vision:visual_cluster_018->audio:audio_cluster_044",
            source_ref="source:aligned",
            confidence=0.8,
            source_reliability=0.8,
            resonance_score=0.8,
            uncertainty=0.2,
            sequence_support_score=0.0,
        )
    )

    plain = cache.retrieve((41, 43, 47))
    aligned = cache.retrieve(
        (41, 43, 47),
        self_state_ids=(
            stable_self_state_id("vision:visual_cluster_018"),
            stable_self_state_id("audio:audio_cluster_044"),
        ),
    )

    assert plain.matches[0]["entry_id"] in {"plain", "aligned"}
    assert aligned.matches[0]["entry_id"] == "aligned"


def test_cache_refresh_from_consolidation_can_promote_recent_entry():
    cache = VerifiedHierarchicalEventStateCache(retention_profile="logarithmic")
    cache.admit(
        _candidate(
            "recent-memory",
            resonance_score=0.68,
            confidence=0.72,
            source_reliability=0.74,
            uncertainty=0.22,
            sequence_support_score=0.4,
        )
    )
    entry = cache.entries["recent-memory"]
    assert entry.tier == "recent"

    results = cache.refresh_from_consolidation(
        (
            {
                "memory_id": "recent-memory",
                "baseline_retention": 0.58,
                "post_retention": 0.76,
                "baseline_noise": 0.34,
                "post_noise": 0.22,
                "health_before": 0.61,
                "health_after": 0.78,
                "phase": "glass",
                "latent_branch_count": 2,
                "selected_branch": "recent:self_state",
            },
        )
    )

    assert results[0].updated is True
    assert results[0].new_tier == "consolidated"
    assert cache.entries["recent-memory"].utility > entry.utility


def test_cache_refresh_from_consolidation_can_defer_weak_entry_to_recent():
    cache = VerifiedHierarchicalEventStateCache(retention_profile="logarithmic")
    cache.admit(
        _candidate(
            "durable-memory",
            resonance_score=0.95,
            confidence=0.92,
            source_reliability=0.92,
            uncertainty=0.08,
            sequence_support_score=0.5,
        )
    )
    entry = cache.entries["durable-memory"]
    assert entry.tier == "durable"

    results = cache.refresh_from_consolidation(
        (
            {
                "memory_id": "durable-memory",
                "baseline_retention": 0.72,
                "post_retention": 0.48,
                "baseline_noise": 0.26,
                "post_noise": 0.39,
                "health_before": 0.73,
                "health_after": 0.49,
                "phase": "liquid",
                "latent_branch_count": 1,
                "selected_branch": "durable:memory",
            },
        )
    )

    assert results[0].updated is True
    assert results[0].new_tier == "recent"
    assert cache.entries["durable-memory"].tier == "recent"


def test_cache_refresh_from_consolidation_can_promote_crystal_memory_to_durable():
    cache = VerifiedHierarchicalEventStateCache(retention_profile="logarithmic")
    cache.admit(
        _candidate(
            "candidate",
            resonance_score=0.76,
            confidence=0.8,
            source_reliability=0.82,
            uncertainty=0.14,
            sequence_support_score=0.6,
            sequence_support_count=3,
        )
    )
    entry = cache.entries["candidate"]
    assert entry.tier == "consolidated"

    results = cache.refresh_from_consolidation(
        (
            {
                "memory_id": "candidate",
                "phase": "crystal",
                "baseline_retention": 0.74,
                "post_retention": 0.80,
                "baseline_noise": 0.18,
                "post_noise": 0.16,
                "health_before": 0.72,
                "health_after": 0.79,
                "multimodal_bundle_affinity": 1.0,
                "latent_branch_count": 3,
                "selected_branch": "consolidated:self_state",
            },
        )
    )

    assert results[0].updated is True
    assert results[0].new_tier == "durable"
    assert results[0].trace["multimodal_bundle_affinity"] == 1.0
