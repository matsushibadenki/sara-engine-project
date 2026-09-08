from __future__ import annotations

import pytest

from sara_engine.agent.bounded_agent_loop import BoundedAgentLoop
from sara_engine.agent.sara_agent import SaraAgent
from sara_engine.agent.tool_result_pairing import IndexedToolCall, IndexedToolResult
from sara_engine.agent.transactional_tools import (
    BoundedTransactionalToolAdapter,
    ToolStateEdit,
    TransactionalToolRequest,
)
from sara_engine.memory.verification_receipt import VerificationReceipt


def _agent() -> SaraAgent:
    agent = object.__new__(SaraAgent)
    agent.transactional_tool_adapter = BoundedTransactionalToolAdapter(
        allowed_tools=("bounded_state_edit",)
    )
    return agent


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


def _request() -> TransactionalToolRequest:
    return TransactionalToolRequest(
        request_id="request-1",
        tool_name="bounded_state_edit",
        goal="open_door",
        expected_outcome="door_open",
        rollback_action="restore_door_state",
        source_ref="fixture:door-controller",
        edits=(ToolStateEdit("set", "door_state", "open"),),
        event_cost=1,
    )


def test_agent_commits_paired_tool_state_and_issues_valid_receipt():
    state = {"door_state": "closed"}
    result = _agent().commit_verified_tool_state(
        state,
        plan=_plan(),
        request=_request(),
        observed_outcome="door_open",
        calls=(IndexedToolCall(0, "call-1", "bounded_state_edit", {}, "str"),),
        results=(IndexedToolResult(0, "call-1", "bounded_state_edit", "door_open"),),
    )

    transaction = result["transaction"]
    receipt = VerificationReceipt.from_dict(result["verification_receipt"])
    assert transaction["committed"] is True
    assert transaction["trace"]["pairing"]["commit_allowed"] is True
    assert state == {"door_state": "open"}
    assert receipt.is_valid() is True
    assert result["durable_memory_mutation"] is False


def test_agent_rolls_back_without_issuing_receipt_on_unexpected_outcome():
    state = {"door_state": "closed"}
    result = _agent().commit_verified_tool_state(
        state,
        plan=_plan(),
        request=_request(),
        observed_outcome="alarm_triggered",
    )

    assert result["transaction"]["rolled_back"] is True
    assert result["verification_receipt"] is None
    assert state == {"door_state": "closed"}


def test_agent_rejects_unpaired_call_or_result_input_before_mutation():
    state = {"door_state": "closed"}
    with pytest.raises(ValueError, match="supplied together"):
        _agent().commit_verified_tool_state(
            state,
            plan=_plan(),
            request=_request(),
            observed_outcome="door_open",
            calls=(IndexedToolCall(0, "call-1", "bounded_state_edit", {}, "str"),),
        )
    assert state == {"door_state": "closed"}
