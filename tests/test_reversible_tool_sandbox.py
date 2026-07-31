from __future__ import annotations

from dataclasses import replace

import pytest

from sara_engine.agent.bounded_agent_loop import BoundedAgentLoop
from sara_engine.agent.reversible_tool_sandbox import (
    IsolatedReversibleToolSandbox,
    ReversibleToolSandboxError,
)
from sara_engine.agent.transactional_tools import (
    ToolStateEdit,
    TransactionalToolRequest,
)


def _plan():
    return BoundedAgentLoop().evaluate_plan(
        goal="open_door",
        structural_prediction="key_matches_lock",
        expected_outcome="door_open",
        rollback_action="restore_door_state",
        risk=0.2,
        active_goal="open_door",
        plan_case={
            "initial_state": ["door_closed", "key_available"],
            "goal": ["door_open"],
            "actions": {
                "open": {
                    "pre": ["door_closed", "key_available"],
                    "add": ["door_open"],
                    "del": ["door_closed"],
                }
            },
            "plan": [{"action": "open"}],
        },
    )


def _request():
    return TransactionalToolRequest(
        request_id="sandbox-request",
        tool_name="bounded_state_edit",
        goal="open_door",
        expected_outcome="door_open",
        rollback_action="restore_door_state",
        source_ref="fixture:sandbox",
        edits=(
            ToolStateEdit("set", "door_state", "open"),
            ToolStateEdit("set", "attempts", 1),
        ),
        event_cost=2,
    )


def _sandbox(initial_state, **overrides):
    values = {
        "sandbox_id": "sandbox-1",
        "allowed_tools": ("bounded_state_edit",),
        "max_operations": 4,
        "max_state_bytes": 1024,
    }
    values.update(overrides)
    return IsolatedReversibleToolSandbox(initial_state, **values)


def test_sandbox_commit_is_private_and_checkpoint_rollback_is_exact():
    caller_state = {"door_state": "closed", "attempts": 0}
    sandbox = _sandbox(caller_state)
    checkpoint = sandbox.checkpoint()

    execution = sandbox.execute(
        plan=_plan(),
        request=_request(),
        observed_outcome="door_open",
    )

    assert execution.tool_result.committed is True
    assert execution.sandbox_revision_advanced is True
    assert execution.external_side_effects_executed is False
    assert caller_state == {"door_state": "closed", "attempts": 0}
    assert sandbox.state == {"door_state": "open", "attempts": 1}

    rollback = sandbox.rollback(checkpoint)

    assert rollback.restored is True
    assert rollback.byte_equivalent_restoration is True
    assert rollback.restored_digest == checkpoint.state_digest
    assert sandbox.state == caller_state
    assert caller_state == {"door_state": "closed", "attempts": 0}


def test_sandbox_does_not_share_nested_state_references():
    caller_state = {
        "door_state": "closed",
        "metadata": {"attempts": 0},
    }
    sandbox = _sandbox(caller_state)
    caller_state["metadata"]["attempts"] = 99
    detached = sandbox.state
    detached["metadata"]["attempts"] = 42

    assert sandbox.state["metadata"]["attempts"] == 0
    assert caller_state["metadata"]["attempts"] == 99


def test_unexpected_outcome_rolls_back_inside_sandbox_without_revision_change():
    sandbox = _sandbox({"door_state": "closed", "attempts": 0})
    before = sandbox.checkpoint()
    execution = sandbox.execute(
        plan=_plan(),
        request=_request(),
        observed_outcome="alarm_triggered",
    )

    assert execution.tool_result.rolled_back is True
    assert execution.sandbox_revision_advanced is False
    assert execution.after_checkpoint.state_digest == before.state_digest
    assert execution.after_checkpoint.revision == before.revision
    assert sandbox.state == before.state


def test_sandbox_rejects_foreign_and_tampered_checkpoints():
    sandbox = _sandbox({"door_state": "closed"})
    checkpoint = sandbox.checkpoint()
    foreign = replace(checkpoint, sandbox_id="sandbox-2")
    tampered = replace(checkpoint, state_digest="forged")

    with pytest.raises(ReversibleToolSandboxError, match="foreign"):
        sandbox.rollback(foreign)
    with pytest.raises(ReversibleToolSandboxError, match="invalid"):
        sandbox.rollback(tampered)

    assert sandbox.state == {"door_state": "closed"}


def test_sandbox_operation_budget_is_hard_bounded():
    sandbox = _sandbox(
        {"door_state": "closed", "attempts": 0},
        max_operations=1,
    )
    sandbox.execute(
        plan=_plan(),
        request=_request(),
        observed_outcome="alarm_triggered",
    )

    with pytest.raises(
        ReversibleToolSandboxError,
        match="sandbox_operation_budget_exceeded",
    ):
        sandbox.execute(
            plan=_plan(),
            request=_request(),
            observed_outcome="alarm_triggered",
        )
