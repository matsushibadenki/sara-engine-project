from __future__ import annotations

import json
from pathlib import Path

from sara_engine.evaluation.phase34_factorial_preregistration import (
    ARMS,
    REPLICATE_SEEDS,
)
from sara_engine.memory.memory_checkpoint_factorial import (
    FactorialLimits,
    MemoryCacheFactorialRuntime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rows():
    path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "benchmark_fixtures"
        / "phase34_memory_cache_factorial_cases.jsonl"
    )
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_all_300_conditions_are_deterministic_bounded_and_query_blind_in_retention():
    limits = FactorialLimits()
    count = 0
    for arm in ARMS:
        for seed in REPLICATE_SEEDS:
            runtime = MemoryCacheFactorialRuntime(arm, limits)
            replay = MemoryCacheFactorialRuntime(arm, limits)
            for row in _rows():
                first = runtime.evaluate(row, seed=seed)
                second = replay.evaluate(row, seed=seed)
                assert first == second
                assert first["bounded"] is True
                assert first["query_visible_during_retention"] is False
                assert first["durable_mutation"] is False
                assert first["production_path_changed"] is False
                assert first["retained_count"] <= limits.max_checkpoints
                assert first["total_state_bytes"] <= limits.max_state_bytes
                assert first["event_cost"] <= limits.max_event_cost
                count += 1
    assert count == 300


def test_selection_pairs_receive_byte_identical_retained_sets():
    limits = FactorialLimits()
    for seed in REPLICATE_SEEDS:
        for row in _rows():
            values = {
                arm: MemoryCacheFactorialRuntime(arm, limits).evaluate(
                    row, seed=seed
                )
                for arm in ARMS[1:]
            }
            assert (
                values[ARMS[1]]["retained_set_digest"]
                == values[ARMS[2]]["retained_set_digest"]
            )
            assert (
                values[ARMS[3]]["retained_set_digest"]
                == values[ARMS[4]]["retained_set_digest"]
            )
            assert values[ARMS[1]]["retention_bytes"] == values[ARMS[2]]["retention_bytes"]
            assert values[ARMS[3]]["retention_bytes"] == values[ARMS[4]]["retention_bytes"]


def test_topk_improves_precision_without_recall_loss_on_retained_targets():
    row = next(
        item
        for item in _rows()
        if item["family"] == "retained_exact_target_pollution"
    )
    limits = FactorialLimits()
    equal_all = MemoryCacheFactorialRuntime(ARMS[1], limits).evaluate(row, seed=109)
    equal_topk = MemoryCacheFactorialRuntime(ARMS[2], limits).evaluate(row, seed=109)
    log_all = MemoryCacheFactorialRuntime(ARMS[3], limits).evaluate(row, seed=109)
    log_topk = MemoryCacheFactorialRuntime(ARMS[4], limits).evaluate(row, seed=109)

    assert equal_all["recall"] == equal_topk["recall"] == 1.0
    assert log_all["recall"] == log_topk["recall"] == 1.0
    assert equal_topk["selection_precision"] > equal_all["selection_precision"]
    assert log_topk["selection_precision"] > log_all["selection_precision"]


def test_retention_main_effect_preserves_old_target_with_resolution_cost():
    row = next(
        item for item in _rows() if item["family"] == "old_target_retention_pressure"
    )
    limits = FactorialLimits()
    equal = MemoryCacheFactorialRuntime(ARMS[1], limits).evaluate(row, seed=227)
    logarithmic = MemoryCacheFactorialRuntime(ARMS[3], limits).evaluate(row, seed=227)

    assert equal["recall"] == 0.0
    assert logarithmic["recall"] == 1.0
    assert equal["retained_temporal_resolution"] > logarithmic[
        "retained_temporal_resolution"
    ]


def test_factorial_safety_cases_fail_closed_for_all_arms():
    expected = {
        "revision_factorial_control": "retrieve",
        "contradiction_factorial_control": "reject_contradiction",
        "stale_digest_factorial_control": "reject_stale",
        "missing_target_factorial_control": "abstain",
    }
    rows = {row["family"]: row for row in _rows()}
    for arm in ARMS:
        runtime = MemoryCacheFactorialRuntime(arm, FactorialLimits())
        for family, decision in expected.items():
            result = runtime.evaluate(rows[family], seed=313)
            assert result["decision"] == decision
            assert result["safety_integrity"] == 1.0
