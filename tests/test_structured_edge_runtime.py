from __future__ import annotations

from sara_engine.neuro.structured_edge import (
    StructuredEdgeLimits,
    StructuredEdgeRuntime,
)


def _case(family, contacts, events, revision="fixture-r1"):
    return {
        "case_id": f"case-{family}",
        "family": family,
        "source_revision": revision,
        "contacts": contacts,
        "events": events,
    }


def test_branch_local_runtime_detects_only_local_coincidence():
    limits = StructuredEdgeLimits()
    same_branch = _case(
        "branch_local_coincidence",
        [
            {"contact_id": "a", "branch": 1, "polarity": "excitatory"},
            {"contact_id": "b", "branch": 1, "polarity": "excitatory"},
            {"contact_id": "c", "branch": 2, "polarity": "excitatory"},
        ],
        [
            {"contact_id": "a", "tick": 0},
            {"contact_id": "b", "tick": 0},
            {"contact_id": "c", "tick": 0},
        ],
    )

    typed = StructuredEdgeRuntime("typed_independent_contacts", limits).evaluate(
        same_branch
    )
    branched = StructuredEdgeRuntime("branch_local_contacts", limits).evaluate(
        same_branch
    )

    assert typed["branch_interaction_count"] == 0
    assert typed["behavior_satisfied"] is False
    assert branched["branch_interaction_count"] == 1
    assert branched["behavior_satisfied"] is True
    assert branched["output_signal"] > typed["output_signal"]


def test_runtime_abstains_safely_on_duplicate_missing_and_stale_contacts():
    runtime = StructuredEdgeRuntime(
        "branch_local_contacts_with_add_prune",
        StructuredEdgeLimits(),
    )
    duplicate = _case(
        "duplicated_contact",
        [
            {"contact_id": "a", "branch": 0},
            {"contact_id": "a", "branch": 1},
        ],
        [{"contact_id": "a", "tick": 0}],
    )
    missing = _case(
        "missing_contact",
        [{"contact_id": "a", "branch": 0}],
        [{"contact_id": "b", "tick": 0}],
    )
    stale = _case(
        "stale_source_revision",
        [{"contact_id": "a", "branch": 0}],
        [{"contact_id": "a", "tick": 0, "source_revision": "fixture-r0"}],
        revision="fixture-r1",
    )

    for case, reason in (
        (duplicate, "duplicate_contact"),
        (missing, "missing_contact"),
        (stale, "stale_source_revision"),
    ):
        result = runtime.evaluate(case)
        assert result["status"] == "abstained"
        assert result["reason"] == reason
        assert result["behavior_satisfied"] is True
        assert result["durable_mutation"] is False


def test_runtime_replay_is_canonical_and_bounded():
    runtime = StructuredEdgeRuntime(
        "typed_independent_contacts",
        StructuredEdgeLimits(max_state_bytes=4096),
    )
    case = _case(
        "same_count_different_order",
        [
            {
                "contact_id": "a",
                "branch": 0,
                "delay_bucket": 0,
                "polarity": "excitatory",
                "role": "first",
            },
            {
                "contact_id": "b",
                "branch": 0,
                "delay_bucket": 0,
                "polarity": "excitatory",
                "role": "second",
            },
        ],
        [
            {"contact_id": "b", "tick": 0},
            {"contact_id": "a", "tick": 1},
        ],
    )

    first = runtime.evaluate(case)
    second = runtime.evaluate(case)

    assert first == second
    assert first["ordered_roles"] == ["second", "first"]
    assert first["behavior_satisfied"] is True
    assert first["state_bytes"] <= 4096
    assert first["event_cost"] == 2
