from __future__ import annotations

import json
from pathlib import Path

from sara_engine.neuro.twinprop_ablation import (
    TwinPropAblationLimits,
    TwinPropAblationRuntime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _limits():
    return TwinPropAblationLimits(
        max_contacts=8,
        max_branches=4,
        max_slow_state_slots=4,
        max_events=64,
        max_interactions=128,
        max_state_bytes=4096,
        decision_window_ticks=4,
        readout_threshold=2,
    )


def _cases():
    path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "benchmark_fixtures"
        / "phase33_twinprop_ablation_cases.jsonl"
    )
    with path.open("r", encoding="utf-8") as handle:
        return {
            row["family"]: row
            for row in (json.loads(line) for line in handle if line.strip())
        }


def test_intact_slow_state_separates_bridge_from_decay_control():
    cases = _cases()
    intact = TwinPropAblationRuntime("intact_bounded_branches", _limits())
    no_slow = TwinPropAblationRuntime(
        "no_slow_coincidence_state",
        _limits(),
    )

    bridge = intact.evaluate(cases["slow_state_bridge"])
    bridge_without_slow = no_slow.evaluate(cases["slow_state_bridge"])
    decay = intact.evaluate(cases["slow_state_decay_control"])

    assert bridge["prediction"] is True
    assert bridge["target_match"] is True
    assert bridge["slow_state_slot_count"] == 1
    assert bridge_without_slow["prediction"] is False
    assert bridge_without_slow["target_match"] is False
    assert decay["prediction"] is False
    assert decay["target_match"] is True


def test_intact_placement_differs_from_topology_collapse():
    cases = _cases()
    intact = TwinPropAblationRuntime("intact_bounded_branches", _limits())
    collapsed = TwinPropAblationRuntime(
        "topology_collapsed_aggregation",
        _limits(),
    )

    structured = intact.evaluate(cases["deterministic_contact_placement"])
    shuffled = intact.evaluate(cases["shuffled_contact_placement"])
    collapsed_shuffle = collapsed.evaluate(cases["shuffled_contact_placement"])

    assert structured["readout_count"] == 2
    assert structured["prediction"] is True
    assert shuffled["readout_count"] == 0
    assert shuffled["prediction"] is False
    assert collapsed_shuffle["prediction"] is True
    assert collapsed_shuffle["target_match"] is False


def test_local_topology_preserves_opposing_subunits():
    cases = _cases()
    intact = TwinPropAblationRuntime("intact_bounded_branches", _limits())
    collapsed = TwinPropAblationRuntime(
        "topology_collapsed_aggregation",
        _limits(),
    )

    intact_result = intact.evaluate(cases["topology_collapse_control"])
    collapsed_result = collapsed.evaluate(cases["topology_collapse_control"])

    assert intact_result["prediction"] is True
    assert intact_result["active_branch_count"] == 1
    assert collapsed_result["prediction"] is False


def test_missing_and_stale_contacts_abstain_without_mutation():
    cases = _cases()
    runtime = TwinPropAblationRuntime("intact_bounded_branches", _limits())

    for family, reason in (
        ("missing_contact", "missing_contact"),
        ("stale_source_revision", "stale_source_revision"),
    ):
        result = runtime.evaluate(cases[family])
        assert result["status"] == "abstained"
        assert result["reason"] == reason
        assert result["target_match"] is True
        assert result["durable_mutation"] is False


def test_runtime_is_deterministic_and_bounded():
    case = _cases()["interaction_order_4"]
    runtime = TwinPropAblationRuntime("intact_bounded_branches", _limits())

    first = runtime.evaluate(case)
    second = runtime.evaluate(case)

    assert first == second
    assert first["active_branch_count"] == 2
    assert first["state_bytes"] <= 4096
    assert first["event_cost"] <= 128
