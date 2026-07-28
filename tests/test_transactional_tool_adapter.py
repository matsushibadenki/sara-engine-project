from __future__ import annotations

from sara_engine.agent.bounded_agent_loop import BoundedAgentLoop
from sara_engine.agent.transactional_tools import (
    BoundedTransactionalToolAdapter,
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


def _request(*edits, verified=True, event_cost=3):
    return TransactionalToolRequest(
        request_id="tool-request-1",
        tool_name="bounded_state_edit",
        goal="open_door",
        expected_outcome="door_open",
        rollback_action="restore_door_state",
        source_ref="fixture:door-controller",
        edits=tuple(edits),
        observed=True,
        verified=verified,
        event_cost=event_cost,
    )


def test_transactional_tool_adapter_commits_verified_expected_outcome():
    state = {"door_state": "closed", "attempts": 0}
    adapter = BoundedTransactionalToolAdapter(allowed_tools=("bounded_state_edit",))
    result = adapter.execute(
        state,
        plan=_plan(),
        request=_request(
            ToolStateEdit("set", "door_state", "open"),
            ToolStateEdit("set", "attempts", 1),
        ),
        observed_outcome="door_open",
    )

    assert result.committed is True
    assert result.rolled_back is False
    assert state == {"door_state": "open", "attempts": 1}
    assert result.after_digest != result.before_digest
    assert result.durable_mutation_allowed is False
    assert result.trace["side_effects_executed"] is False


def test_transactional_tool_adapter_rolls_back_unexpected_outcome_exactly():
    state = {"door_state": "closed", "attempts": 0}
    before = dict(state)
    adapter = BoundedTransactionalToolAdapter(allowed_tools=("bounded_state_edit",))
    result = adapter.execute(
        state,
        plan=_plan(),
        request=_request(ToolStateEdit("set", "door_state", "open")),
        observed_outcome="alarm_triggered",
    )

    assert result.executed is True
    assert result.committed is False
    assert result.rolled_back is True
    assert result.decision == "rollback_unexpected_tool_outcome"
    assert state == before
    assert result.restored_digest == result.before_digest
    assert result.trace["byte_equivalent_restoration"] is True


def test_transactional_tool_adapter_rolls_back_after_late_staged_failure():
    state = {"door_state": "closed"}
    before = dict(state)
    adapter = BoundedTransactionalToolAdapter(allowed_tools=("bounded_state_edit",))
    result = adapter.execute(
        state,
        plan=_plan(),
        request=_request(
            ToolStateEdit("set", "door_state", "open"),
            ToolStateEdit("delete", "missing_key"),
        ),
        observed_outcome="door_open",
    )

    assert result.rolled_back is True
    assert result.decision == "rollback_staged_tool_error"
    assert result.after_digest != result.before_digest
    assert result.restored_digest == result.before_digest
    assert state == before


def test_transactional_tool_adapter_rejects_unverified_and_budgeted_requests():
    state = {"door_state": "closed"}
    adapter = BoundedTransactionalToolAdapter(
        allowed_tools=("bounded_state_edit",),
        max_edits=1,
        max_event_cost=2,
    )
    unverified = adapter.execute(
        state,
        plan=_plan(),
        request=_request(
            ToolStateEdit("set", "door_state", "open"),
            verified=False,
        ),
        observed_outcome="door_open",
    )
    over_budget = adapter.execute(
        state,
        plan=_plan(),
        request=_request(
            ToolStateEdit("set", "door_state", "open"),
            ToolStateEdit("set", "attempts", 1),
            event_cost=3,
        ),
        observed_outcome="door_open",
    )
    non_finite = adapter.execute(
        state,
        plan=_plan(),
        request=_request(
            ToolStateEdit("set", "confidence", float("nan")),
            event_cost=1,
        ),
        observed_outcome="door_open",
    )

    assert unverified.executed is False
    assert "tool_evidence_not_verified" in unverified.trace["errors"]
    assert over_budget.executed is False
    assert "tool_edit_budget_exceeded" in over_budget.trace["errors"]
    assert "tool_event_budget_exceeded" in over_budget.trace["errors"]
    assert non_finite.executed is False
    assert "malformed_tool_edit" in non_finite.trace["errors"]
    assert state == {"door_state": "closed"}
