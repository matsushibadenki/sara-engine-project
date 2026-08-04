from __future__ import annotations

import json
from pathlib import Path

from sara_engine.evaluation.phase34_memory_cache_preregistration import ARMS
from sara_engine.evaluation.phase34_separation_preregistration import REPLICATE_SEEDS
from sara_engine.memory.memory_checkpoint_separation import (
    MemoryCacheSeparationRuntime,
    SeparationLimits,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rows():
    path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "benchmark_fixtures"
        / "phase34_memory_cache_separation_cases.jsonl"
    )
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_all_240_conditions_are_deterministic_bounded_and_read_only():
    limits = SeparationLimits()
    condition_count = 0
    for arm in ARMS:
        for seed in REPLICATE_SEEDS:
            runtime = MemoryCacheSeparationRuntime(arm, limits)
            replay = MemoryCacheSeparationRuntime(arm, limits)
            for row in _rows():
                first = runtime.evaluate(row, seed=seed)
                second = replay.evaluate(row, seed=seed)
                assert first == second
                assert first["bounded"] is True
                assert first["durable_mutation"] is False
                assert first["production_path_changed"] is False
                assert first["retained_count"] <= limits.max_checkpoints
                assert first["state_bytes"] <= limits.max_state_bytes
                assert first["event_cost"] <= limits.max_event_cost
                condition_count += 1
    assert condition_count == 240


def test_logarithmic_retention_preserves_old_target_after_equal_eviction():
    row = next(
        item for item in _rows() if item["family"] == "old_target_after_overflow"
    )
    limits = SeparationLimits()
    equal = MemoryCacheSeparationRuntime(ARMS[1], limits).evaluate(row, seed=107)
    logarithmic = MemoryCacheSeparationRuntime(ARMS[2], limits).evaluate(
        row, seed=107
    )

    assert equal["recall"] == 0.0
    assert logarithmic["recall"] == 1.0
    assert logarithmic["merge_count"] > 0


def test_logarithmic_retention_lowers_aggregate_temporal_resolution():
    row = next(
        item for item in _rows() if item["family"] == "boundary_burst_recent"
    )
    limits = SeparationLimits()
    equal = MemoryCacheSeparationRuntime(ARMS[1], limits).evaluate(row, seed=223)
    logarithmic = MemoryCacheSeparationRuntime(ARMS[2], limits).evaluate(
        row, seed=223
    )

    assert equal["retained_temporal_resolution"] == 1.0
    assert logarithmic["retained_temporal_resolution"] < 1.0
    assert equal["recall"] == logarithmic["recall"] == 1.0


def test_pollution_fixture_exposes_retention_confound_in_topk_arm():
    row = next(item for item in _rows() if item["family"] == "relevance_pollution")
    limits = SeparationLimits()
    retrieve_all = MemoryCacheSeparationRuntime(ARMS[1], limits).evaluate(
        row, seed=311
    )
    topk = MemoryCacheSeparationRuntime(ARMS[3], limits).evaluate(row, seed=311)

    assert retrieve_all["recall"] == 0.0
    assert topk["recall"] == 0.0
    assert topk["selection_precision"] == 0.0


def test_revision_contradiction_stale_and_missing_cases_fail_closed():
    expected = {
        "revision_after_merge": "retrieve",
        "contradiction_after_merge": "reject_contradiction",
        "stale_digest_after_merge": "reject_stale",
        "missing_target": "abstain",
    }
    rows = {row["family"]: row for row in _rows()}
    for arm in ARMS:
        runtime = MemoryCacheSeparationRuntime(arm, SeparationLimits())
        for family, decision in expected.items():
            result = runtime.evaluate(rows[family], seed=419)
            assert result["decision"] == decision
            assert result["safety_integrity"] == 1.0
