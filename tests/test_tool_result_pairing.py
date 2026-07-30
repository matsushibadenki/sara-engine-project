from __future__ import annotations

import pytest

from sara_engine.agent.bounded_agent_loop import BoundedAgentLoop
from sara_engine.agent.tool_result_pairing import (
    IndexedToolCall,
    IndexedToolResult,
    IndexedToolResultPairingGate,
)
from sara_engine.agent.transactional_tools import (
    BoundedTransactionalToolAdapter,
    ToolStateEdit,
    TransactionalToolRequest,
)


def _calls():
    return (
        IndexedToolCall(0, "call-0", "inspect_lock", {"door": "front"}, "object"),
        IndexedToolCall(1, "call-1", "inspect_key", {"key": "brass"}, "bool"),
    )


def _results():
    return (
        IndexedToolResult(0, "call-0", "inspect_lock", {"state": "closed"}),
        IndexedToolResult(1, "call-1", "inspect_key", True),
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
        request_id="paired-tool-request",
        tool_name="bounded_state_edit",
        goal="open_door",
        expected_outcome="door_open",
        rollback_action="restore_door_state",
        source_ref="fixture:paired-tools",
        edits=(ToolStateEdit("set", "door_state", "open"),),
        event_cost=2,
    )


def test_indexed_typed_pairing_accepts_exact_parallel_results():
    validation = IndexedToolResultPairingGate().validate(_calls(), _results())

    assert validation.commit_allowed is True
    assert validation.errors == ()
    assert validation.call_count == validation.result_count == 2
    assert validation.event_cost == 2
    assert validation.pairing_digest
    assert validation.trace["side_effects_executed"] is False


@pytest.mark.parametrize(
    ("results", "error"),
    [
        ((_results()[0],), "missing_tool_result"),
        ((_results()[0], _results()[0]), "duplicate_tool_result_id"),
        (tuple(reversed(_results())), "reordered_tool_results"),
        (
            (_results()[0], IndexedToolResult(1, "call-1", "inspect_key", 1)),
            "tool_result_type_mismatch",
        ),
        (
            (
                _results()[0],
                IndexedToolResult(1, "call-1", "different_tool", True),
            ),
            "reordered_tool_results",
        ),
    ],
)
def test_indexed_typed_pairing_rejects_invalid_result_batches(results, error):
    validation = IndexedToolResultPairingGate().validate(_calls(), results)

    assert validation.commit_allowed is False
    assert error in validation.errors
    assert validation.pairing_digest == ""


def test_invalid_pairing_blocks_transactional_commit_without_state_change():
    state = {"door_state": "closed"}
    adapter = BoundedTransactionalToolAdapter(
        allowed_tools=("bounded_state_edit",),
        max_edits=4,
        max_event_cost=8,
        max_state_bytes=1024,
    )
    invalid_results = tuple(reversed(_results()))
    rejected = adapter.execute_paired(
        state,
        plan=_plan(),
        request=_request(),
        observed_outcome="door_open",
        calls=_calls(),
        results=invalid_results,
    )

    assert rejected.committed is False
    assert rejected.executed is False
    assert rejected.decision == "reject_tool_result_pairing"
    assert "reordered_tool_results" in rejected.trace["errors"]
    assert state == {"door_state": "closed"}
    assert rejected.before_digest == rejected.after_digest


def test_valid_pairing_enters_existing_transactional_commit_path():
    state = {"door_state": "closed"}
    adapter = BoundedTransactionalToolAdapter(
        allowed_tools=("bounded_state_edit",),
        max_edits=4,
        max_event_cost=8,
        max_state_bytes=1024,
    )
    committed = adapter.execute_paired(
        state,
        plan=_plan(),
        request=_request(),
        observed_outcome="door_open",
        calls=_calls(),
        results=_results(),
    )

    assert committed.committed is True
    assert state == {"door_state": "open"}
    assert committed.trace["pairing"]["commit_allowed"] is True
    assert committed.trace["pairing"]["pairing_digest"]
