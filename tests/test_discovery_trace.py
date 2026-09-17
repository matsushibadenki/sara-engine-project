"""Observable replay transcripts must be reproducible and tamper-evident."""
from dataclasses import replace

import pytest

from sara_engine.research import (
    DiscoveryReplayWorld, ReplayRecord, run_traced_replay_policy,
    verify_replay_transcript,
)


HASH = "a" * 64


def record(sequence, node_id, parent_id, *, score=1.0):
    return ReplayRecord(
        sequence=sequence, node_id=node_id, parent_id=parent_id,
        hypothesis_id="hypothesis-1", policy_id="fixed-policy-v1",
        candidate_sha256=HASH, source_sha256=HASH,
        preregistration_sha256=HASH, evaluator_id="frozen-evaluator-v1",
        split="development", status="root" if sequence == 0 else "valid",
        score=None if sequence == 0 else score,
        cpu_ms=0 if sequence == 0 else 10,
        event_count=0 if sequence == 0 else 2,
        state_bytes=0 if sequence == 0 else 32,
    )


def tree():
    return (
        record(0, "root", None),
        record(1, "branch-a", "root"),
        record(2, "child-a", "branch-a", score=0.25),
        record(3, "branch-b", "root", score=2.0),
    )


def test_transcript_rebuilds_only_observed_decisions():
    rows = tree()
    observed = []

    def policy(view):
        observed.append(tuple(record.node_id for record in view.revealed))
        if len(observed) == 1:
            return ("root",)
        if len(observed) == 2:
            return ("branch-a", "root")
        return ()

    world = DiscoveryReplayWorld(rows)
    transcript = run_traced_replay_policy(world, policy)
    assert observed == [("root",), ("root", "branch-a"),
                        ("root", "branch-a", "child-a", "branch-b")]
    assert transcript.decisions[0].revealed_node_ids == ("branch-a",)
    assert transcript.decisions[1].revealed_node_ids == ("child-a", "branch-b")
    assert verify_replay_transcript(rows, transcript) == world.view


def test_transcript_rejects_hidden_action_and_tampering():
    rows = tree()
    with pytest.raises(ValueError, match="not visible"):
        run_traced_replay_policy(DiscoveryReplayWorld(rows), lambda view: ("child-a",))
    transcript = run_traced_replay_policy(
        DiscoveryReplayWorld(rows), lambda view: ("root",) if view.rounds == 0 else (),
    )
    altered = replace(transcript.decisions[0], actions=("child-a",))
    with pytest.raises(ValueError, match="unavailable action"):
        verify_replay_transcript(rows, replace(transcript, decisions=(altered, *transcript.decisions[1:])))
    altered = replace(transcript.decisions[0], revealed_node_ids=("branch-b",))
    with pytest.raises(ValueError, match="reveal mismatch"):
        verify_replay_transcript(rows, replace(transcript, decisions=(altered, *transcript.decisions[1:])))
    with pytest.raises(ValueError, match="tree mismatch"):
        verify_replay_transcript((*rows[:2], replace(rows[2], score=0.5), rows[3]), transcript)


def test_trace_requires_fresh_world_and_bounded_rounds():
    world = DiscoveryReplayWorld(tree())
    world.reveal(("root",))
    with pytest.raises(ValueError, match="fresh world"):
        run_traced_replay_policy(world, lambda view: ())
    with pytest.raises(ValueError, match="round budget"):
        run_traced_replay_policy(DiscoveryReplayWorld(tree()), lambda view: (), max_rounds=0)


def test_policy_can_follow_available_support_without_hidden_node_names():
    rows = tree()
    seen = []

    def policy(view):
        seen.append(view.available_actions)
        return view.available_actions[:1]

    world = DiscoveryReplayWorld(rows, max_reveals=3)
    transcript = run_traced_replay_policy(world, policy, max_rounds=4)
    assert seen == [("root",), ("root", "branch-a"),
                    ("branch-a",), ()]
    assert world.view.reveals == 3
    assert verify_replay_transcript(rows, transcript) == world.view
