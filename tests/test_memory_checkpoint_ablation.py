from __future__ import annotations

import json
from pathlib import Path

from sara_engine.memory.memory_checkpoint_ablation import (
    ARMS,
    Phase34MemoryCacheLimits,
    Phase34MemoryCheckpointRuntime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rows():
    path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "benchmark_fixtures"
        / "phase34_memory_checkpoint_cache_cases.jsonl"
    )
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_all_registered_arms_are_bounded_deterministic_and_read_only():
    rows = _rows()
    limits = Phase34MemoryCacheLimits()
    for arm in ARMS:
        runtime = Phase34MemoryCheckpointRuntime(arm, limits)
        replay = Phase34MemoryCheckpointRuntime(arm, limits)
        first = [runtime.evaluate(row) for row in rows]
        second = [replay.evaluate(row) for row in rows]

        assert first == second
        assert all(result["durable_mutation"] is False for result in first)
        assert all(result["production_path_changed"] is False for result in first)
        assert all(result["state_bytes"] <= limits.max_state_bytes for result in first)
        assert all(result["event_cost"] <= limits.max_event_cost for result in first)
        assert all(result["checkpoint_count"] <= limits.max_checkpoints for result in first)
        assert all(result["selected_count"] <= limits.selected_k for result in first)


def test_checkpoint_arm_improves_long_gap_recall_over_recurrent_control():
    case = next(row for row in _rows() if row["family"] == "long_irrelevant_interval")
    limits = Phase34MemoryCacheLimits()

    control = Phase34MemoryCheckpointRuntime(ARMS[0], limits).evaluate(case)
    cache = Phase34MemoryCheckpointRuntime(ARMS[3], limits).evaluate(case)

    assert control["decision"] == "abstain"
    assert control["target_match"] is False
    assert cache["decision"] == "retrieve"
    assert cache["target_match"] is True


def test_negative_cases_fail_closed_for_every_cache_arm():
    rows = {row["family"]: row for row in _rows()}
    expected = {
        "contradiction": "reject_contradiction",
        "missing_segment": "abstain",
        "stale_runtime_digest": "reject_stale",
        "stale_schema_digest": "reject_stale",
        "reordered_replay": "abstain",
        "cache_overflow": "evict",
    }
    for arm in ARMS[1:]:
        runtime = Phase34MemoryCheckpointRuntime(arm, Phase34MemoryCacheLimits())
        for family, decision in expected.items():
            result = runtime.evaluate(rows[family])
            assert result["decision"] == decision
            assert result["target_match"] is True


def test_logarithmic_arm_preserves_parent_provenance_under_overflow():
    case = next(row for row in _rows() if row["family"] == "cache_overflow")
    result = Phase34MemoryCheckpointRuntime(
        "logarithmic_segments_retrieve_all",
        Phase34MemoryCacheLimits(),
    ).evaluate(case)

    assert result["decision"] == "evict"
    assert result["checkpoint_count"] <= 8
    assert result["merge_count"] <= 2
