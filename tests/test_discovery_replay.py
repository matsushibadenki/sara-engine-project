"""Observed-only research replay must never manufacture or leak outcomes."""
from dataclasses import replace

import pytest

from sara_engine.research.discovery_replay import DiscoveryReplayWorld, ReplayRecord
from sara_engine.research.discovery_audit import audit_replay


HASH = "a" * 64


def record(sequence, node_id, parent_id, *, status="valid", score=1.0):
    return ReplayRecord(
        sequence=sequence,
        node_id=node_id,
        parent_id=parent_id,
        hypothesis_id="hypothesis-1",
        policy_id="fixed-policy-v1",
        candidate_sha256=HASH,
        source_sha256=HASH,
        preregistration_sha256=HASH,
        evaluator_id="frozen-evaluator-v1",
        split="development",
        status=status,
        score=score,
        cpu_ms=0 if sequence == 0 else 10,
        event_count=0 if sequence == 0 else 2,
        state_bytes=0 if sequence == 0 else 32,
    )


def tree():
    return (
        record(0, "root", None, status="root", score=None),
        record(1, "branch-a", "root"),
        record(2, "child-a", "branch-a", status="negative", score=0.25),
        record(3, "branch-b", "root", score=2.0),
    )


def test_replay_reveals_only_recorded_children_in_order():
    world = DiscoveryReplayWorld(tree(), max_reveals=3)
    assert [row.node_id for row in world.view.revealed] == ["root"]
    with pytest.raises(ValueError, match="not visible"):
        world.reveal(["branch-a"])
    assert [row.node_id for row in world.view.revealed] == ["root"]
    assert [row.node_id for row in world.reveal(["root"]).revealed] == ["root", "branch-a"]
    assert [row.node_id for row in world.reveal(["branch-a", "root"]).revealed] == [
        "root", "branch-a", "child-a", "branch-b",
    ]
    with pytest.raises(ValueError, match="no recorded continuation"):
        world.reveal(["child-a"])
    assert world.view.reveals == 3
    assert world.view.rounds == 2
    assert world.reveal([]).stopped is True
    with pytest.raises(ValueError, match="stopped"):
        world.reveal(["root"])


def test_invalid_batches_are_atomic_and_bounded():
    world = DiscoveryReplayWorld(tree(), max_batch=1, max_reveals=1)
    with pytest.raises(ValueError, match="too large"):
        world.reveal(["root", "root"])
    assert world.view.rounds == world.view.reveals == 0
    world.reveal(["root"])
    with pytest.raises(ValueError, match="budget exceeded"):
        world.reveal(["branch-a"])
    assert [row.node_id for row in world.view.revealed] == ["root", "branch-a"]
    assert world.view.rounds == world.view.reveals == 1


@pytest.mark.parametrize("changed", [
    lambda rows: (rows[0], replace(rows[1], split="heldout"), *rows[2:]),
    lambda rows: (rows[0], replace(rows[1], evaluator_id="changed-evaluator"), *rows[2:]),
    lambda rows: (rows[0], replace(rows[1], score=float("nan")), *rows[2:]),
    lambda rows: (rows[0], replace(rows[1], parent_id="missing"), *rows[2:]),
    lambda rows: (rows[0], replace(rows[1], candidate_sha256="not-a-hash"), *rows[2:]),
    lambda rows: (rows[0], replace(rows[1], sequence=0), *rows[2:]),
    lambda rows: (rows[0], replace(rows[1], status="failed", score=1.0), *rows[2:]),
    lambda rows: (rows[0], replace(rows[1], status="negative", score=None), *rows[2:]),
    lambda rows: (*rows, record(4, "sibling-a", "branch-a")),
])
def test_invalid_or_leaky_trees_fail_closed(changed):
    with pytest.raises(ValueError):
        DiscoveryReplayWorld(changed(tree()))


def test_record_schema_and_tree_digest_are_deterministic():
    rows = tree()
    assert ReplayRecord.from_mapping(rows[1].__dict__) == rows[1]
    with pytest.raises(ValueError, match="fields"):
        ReplayRecord.from_mapping({**rows[1].__dict__, "secret": "raw evidence"})
    assert DiscoveryReplayWorld(rows).tree_sha256 == DiscoveryReplayWorld(rows).tree_sha256
    with pytest.raises(ValueError, match="node budget"):
        DiscoveryReplayWorld(rows, max_nodes=3)


def test_postrun_audit_separates_missing_and_budget_coverage():
    rows = (
        record(0, "root", None, status="root", score=None),
        record(1, "branch-a", "root"),
        record(2, "child-a", "branch-a", status="missing", score=None),
        record(3, "branch-b", "root", status="failed", score=None),
        record(4, "branch-c", "root"),
    )
    world = DiscoveryReplayWorld(rows, max_reveals=2)
    before = audit_replay(world)
    assert before.recorded_nonroot == 4
    assert before.revealed_nonroot == 0
    assert before.budget_unreachable == 1
    assert before.missing_recorded == 1
    assert before.missing_revealed == 0
    assert before.failed_recorded == 1
    world.reveal(["root"])
    world.reveal(["branch-a"])
    after = audit_replay(world)
    assert after.coverage == 0.5
    assert after.unrevealed_recorded == 2
    assert after.missing_revealed == 1
    assert after.budget_unreachable == 1
