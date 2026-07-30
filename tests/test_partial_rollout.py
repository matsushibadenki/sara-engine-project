from __future__ import annotations

import pytest

from sara_engine.agent.candidate_execution import CandidateExecution
from sara_engine.agent.partial_rollout import (
    BoundedPartialRolloutScheduler,
    PartialRolloutError,
    RolloutResumeContext,
)


def _execution(execution_id: str, *, event_budget: int = 4) -> CandidateExecution:
    return CandidateExecution.create(
        execution_id=execution_id,
        goal="inspect_door",
        plan={"steps": ["inspect", "compare"]},
        source_revision="door-r1",
        state={"observations": 0},
        event_budget=event_budget,
        sandbox_checkpoint_identity=f"checkpoint:{execution_id}",
        max_state_bytes=512,
    )


def _contexts(scheduler, *execution_ids):
    return {
        execution_id: RolloutResumeContext.from_execution(
            scheduler.execution(execution_id)
        )
        for execution_id in execution_ids
    }


def test_partial_rollout_scheduler_is_round_robin_and_bounded_stale():
    scheduler = BoundedPartialRolloutScheduler(
        max_trajectories=2,
        max_slice_events=1,
        max_staleness_ticks=2,
        max_total_state_bytes=1024,
    )
    scheduler.register(_execution("candidate-a"))
    scheduler.register(_execution("candidate-b"))

    first = scheduler.dispatch_next({})
    first_result = scheduler.complete_slice(
        dispatch_token=first.dispatch_token,
        state={"observations": 1},
        event_cost=1,
    )
    second = scheduler.dispatch_next({})
    second_result = scheduler.complete_slice(
        dispatch_token=second.dispatch_token,
        state={"observations": 1},
        event_cost=1,
    )
    third = scheduler.dispatch_next(_contexts(scheduler, "candidate-a"))
    third_result = scheduler.complete_slice(
        dispatch_token=third.dispatch_token,
        state={"observations": 2},
        event_cost=1,
    )

    assert [first.execution.execution_id, second.execution.execution_id] == [
        "candidate-a",
        "candidate-b",
    ]
    assert third.execution.execution_id == "candidate-a"
    assert first_result.status == second_result.status == "paused"
    assert third_result.staleness_ticks == 2
    assert third_result.staleness_ticks <= scheduler.max_staleness_ticks
    assert scheduler.snapshot()["total_state_bytes"] <= 1024


def test_partial_rollout_completes_exactly_when_event_budget_is_spent():
    scheduler = BoundedPartialRolloutScheduler(
        max_trajectories=1,
        max_slice_events=2,
        max_staleness_ticks=1,
    )
    scheduler.register(_execution("candidate-a", event_budget=2))
    dispatch = scheduler.dispatch_next({})
    result = scheduler.complete_slice(
        dispatch_token=dispatch.dispatch_token,
        state={"observations": 2},
        event_cost=2,
    )

    assert result.status == "completed"
    assert result.event_budget_remaining == 0
    assert scheduler.execution("candidate-a").status == "completed"
    with pytest.raises(PartialRolloutError, match="no_schedulable_rollout"):
        scheduler.dispatch_next({})


def test_partial_rollout_rejects_stale_resume_revision_without_advancing():
    scheduler = BoundedPartialRolloutScheduler(
        max_trajectories=1,
        max_slice_events=1,
        max_staleness_ticks=1,
    )
    scheduler.register(_execution("candidate-a"))
    dispatch = scheduler.dispatch_next({})
    scheduler.complete_slice(
        dispatch_token=dispatch.dispatch_token,
        state={"observations": 1},
        event_cost=1,
    )
    paused = scheduler.execution("candidate-a")
    stale_context = RolloutResumeContext(
        goal=paused.goal,
        plan=paused.plan,
        source_revision="door-r2",
        state_digest=paused.state_digest,
        event_budget_remaining=paused.event_budget_remaining,
        sandbox_checkpoint_identity=paused.sandbox_checkpoint_identity,
    )

    with pytest.raises(PartialRolloutError, match="stale_source_revision"):
        scheduler.dispatch_next({"candidate-a": stale_context})

    snapshot = scheduler.snapshot()
    assert snapshot["scheduler_tick"] == 1
    assert snapshot["in_flight"] is None
    assert scheduler.execution("candidate-a").status == "paused"


def test_partial_rollout_requires_a_queue_coverable_staleness_bound():
    with pytest.raises(
        PartialRolloutError,
        match="staleness_bound_cannot_cover_queue",
    ):
        BoundedPartialRolloutScheduler(
            max_trajectories=3,
            max_staleness_ticks=2,
        )


def test_partial_rollout_rejects_bad_dispatch_and_slice_budgets():
    scheduler = BoundedPartialRolloutScheduler(
        max_trajectories=1,
        max_slice_events=1,
        max_staleness_ticks=1,
    )
    scheduler.register(_execution("candidate-a"))
    dispatch = scheduler.dispatch_next({})

    with pytest.raises(PartialRolloutError, match="invalid_rollout_dispatch_token"):
        scheduler.complete_slice(
            dispatch_token="forged",
            state={"observations": 1},
            event_cost=1,
        )
    with pytest.raises(
        PartialRolloutError,
        match="rollout_slice_event_budget_exceeded",
    ):
        scheduler.complete_slice(
            dispatch_token=dispatch.dispatch_token,
            state={"observations": 2},
            event_cost=2,
        )
