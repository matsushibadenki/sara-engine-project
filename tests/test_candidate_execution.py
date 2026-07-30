from __future__ import annotations

import pytest

from sara_engine.agent.candidate_execution import (
    CandidateExecution,
    CandidateExecutionError,
)


def _execution() -> CandidateExecution:
    return CandidateExecution.create(
        execution_id="candidate-1",
        goal="open_door",
        plan={"steps": ["inspect", "open"], "rollback": "restore_door"},
        source_revision="door-source-r3",
        state={"door": "closed", "attempts": 0},
        event_budget=8,
        sandbox_checkpoint_identity="sandbox-checkpoint-7",
        max_state_bytes=512,
    )


def _resume(paused: CandidateExecution, **overrides) -> CandidateExecution:
    values = {
        "goal": paused.goal,
        "plan": paused.plan,
        "source_revision": paused.source_revision,
        "state_digest": paused.state_digest,
        "event_budget_remaining": paused.event_budget_remaining,
        "sandbox_checkpoint_identity": paused.sandbox_checkpoint_identity,
    }
    values.update(overrides)
    return paused.resume(**values)


def test_candidate_execution_pauses_snapshots_and_resumes_exact_context():
    active = _execution()
    paused = active.pause()
    snapshot = paused.snapshot()
    resumed = _resume(paused)

    assert active.status == "active"
    assert paused.status == "paused"
    assert resumed.status == "active"
    assert snapshot["goal"] == "open_door"
    assert snapshot["plan"] == active.plan
    assert snapshot["source_revision"] == "door-source-r3"
    assert snapshot["state_digest"] == active.state_digest
    assert snapshot["event_budget_remaining"] == 8
    assert snapshot["sandbox_checkpoint_identity"] == "sandbox-checkpoint-7"
    assert snapshot["snapshot_digest"]


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"goal": "close_door"}, "stale_execution_goal"),
        ({"plan": {"steps": ["force"]}}, "stale_execution_plan"),
        ({"source_revision": "door-source-r4"}, "stale_source_revision"),
        ({"state_digest": "forged"}, "stale_execution_state"),
        ({"event_budget_remaining": 7}, "stale_event_budget"),
        ({"event_budget_remaining": "8"}, "stale_event_budget"),
        ({"sandbox_checkpoint_identity": "other"}, "stale_sandbox_checkpoint"),
    ],
)
def test_candidate_execution_rejects_stale_resume_context(override, error):
    paused = _execution().pause()

    with pytest.raises(CandidateExecutionError, match=error):
        _resume(paused, **override)


def test_judging_fork_is_read_only_and_cannot_mutate_source():
    source = _execution().pause()
    fork = source.fork_for_judging(execution_id="candidate-1-judge")
    detached = fork.state
    detached["door"] = "open"

    assert fork.read_only is True
    assert fork.parent_execution_id == source.execution_id
    assert fork.state["door"] == "closed"
    assert source.state["door"] == "closed"
    assert source.snapshot()["snapshot_digest"] != fork.snapshot()["snapshot_digest"]
    with pytest.raises(CandidateExecutionError, match="read_only_judging_fork"):
        fork.apply_state(
            {"door": "open"},
            event_cost=1,
            expected_state_digest=fork.state_digest,
        )


def test_candidate_state_updates_are_copy_on_write_and_budgeted():
    source = _execution()
    updated = source.apply_state(
        {"door": "open", "attempts": 1},
        event_cost=3,
        expected_state_digest=source.state_digest,
    )

    assert source.state == {"door": "closed", "attempts": 0}
    assert source.event_budget_remaining == 8
    assert updated.state == {"door": "open", "attempts": 1}
    assert updated.event_budget_remaining == 5
    assert updated.state_digest != source.state_digest
    with pytest.raises(CandidateExecutionError, match="candidate_event_budget_exceeded"):
        updated.apply_state(
            {"door": "open", "attempts": 2},
            event_cost=6,
            expected_state_digest=updated.state_digest,
        )
    with pytest.raises(CandidateExecutionError, match="candidate_event_budget_exceeded"):
        updated.apply_state(
            {"door": "open", "attempts": 2},
            event_cost=1.0,
            expected_state_digest=updated.state_digest,
        )
