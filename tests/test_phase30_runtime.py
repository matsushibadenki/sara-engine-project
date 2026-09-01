from __future__ import annotations

import copy

import pytest

from sara_engine.evaluation.phase30_fixtures import build_fixtures
from sara_engine.evaluation.phase30_preregistration import ARMS
from sara_engine.evaluation.phase30_runtime import TemporalControlRuntime, run_control


def _case(family: str, partition: str = "evaluation"):
    inputs, _, _ = build_fixtures()
    return next(case for case in inputs if case["case_family"] == family and case["partition"] == partition)


@pytest.mark.parametrize("arm", ARMS)
def test_phase30_control_is_deterministic_bounded_and_default_isolated(arm):
    case = _case("delayed_response")
    first = run_control(case, arm)
    second = run_control(copy.deepcopy(case), arm)
    assert first == second
    assert first["event_count"] == 256
    assert first["active_edge_count"] <= 64
    assert first["cached_interaction_count"] <= 32
    assert first["event_cost"] <= 4096
    assert first["production_mutation"] is False
    assert first["durable_knowledge_mutation"] is False
    assert len(first["replay_digest"]) == 64


def test_phase30_only_cache_arm_materializes_interactions():
    case = _case("phase_synchrony_discrimination")
    results = {arm: run_control(case, arm) for arm in ARMS}
    assert results["temporal_state_bounded_effective_interaction"]["cache_builds"] > 0
    assert results["temporal_state_bounded_effective_interaction"]["cache_hits"] > 0
    for arm in ARMS[:-1]:
        assert results[arm]["cache_builds"] == 0
        assert results[arm]["cached_interaction_count"] == 0


@pytest.mark.parametrize("family,reason", (("context_revision", "context_revision"), ("stale_cache", "expiry"), ("contradiction", "contradiction")))
def test_phase30_cache_invalidation_is_exact_and_replayable(family, reason):
    result = run_control(_case(family), "temporal_state_bounded_effective_interaction")
    matching = [entry for entry in result["invalidation_trace"] if entry["reason"] == reason]
    assert matching
    assert any(entry["cache_entry_removed"] for entry in matching)
    assert result == run_control(_case(family), "temporal_state_bounded_effective_interaction")


def test_phase30_no_reuse_stays_within_edge_and_cache_capacity():
    result = run_control(_case("no_reuse"), "temporal_state_bounded_effective_interaction")
    assert result["active_edge_count"] == 64
    assert result["cache_builds"] == 0
    assert result["cached_interaction_count"] == 0
    assert any(entry["reason"] == "active_edge_eviction" for entry in result["invalidation_trace"])


def test_phase30_unseen_context_and_shuffled_time_abstain_without_labels():
    unseen = run_control(_case("unseen_context"), "temporal_state_only")
    shuffled = run_control(_case("shuffled_time"), "temporal_state_only")
    assert unseen["decision"] == "abstain"
    assert shuffled["decision"] == "abstain"
    assert unseen["unknown_context_events"] > 0
    assert shuffled["nonmonotonic_events"] > 0


def test_phase30_runtime_rejects_unknown_arm_and_malformed_event():
    with pytest.raises(ValueError, match="unknown_phase30_arm"):
        TemporalControlRuntime(arm="unknown")
    runtime = TemporalControlRuntime(arm="fixed_sparse_snn")
    with pytest.raises(ValueError, match="phase30_event_contract_incomplete"):
        runtime.observe({"edge_id": "edge"})
