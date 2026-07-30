"""Deterministic bounded scheduling for resumable partial agent rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

from sara_engine.memory.verification_receipt import evidence_digest

from .candidate_execution import CandidateExecution, CandidateExecutionError


class PartialRolloutError(ValueError):
    """Raised when a partial-rollout scheduling contract is violated."""


@dataclass(frozen=True)
class RolloutResumeContext:
    goal: str
    plan: Mapping[str, Any]
    source_revision: str
    state_digest: str
    event_budget_remaining: int
    sandbox_checkpoint_identity: str

    @classmethod
    def from_execution(cls, execution: CandidateExecution) -> "RolloutResumeContext":
        return cls(
            goal=execution.goal,
            plan=execution.plan,
            source_revision=execution.source_revision,
            state_digest=execution.state_digest,
            event_budget_remaining=execution.event_budget_remaining,
            sandbox_checkpoint_identity=execution.sandbox_checkpoint_identity,
        )


@dataclass(frozen=True)
class PartialRolloutDispatch:
    dispatch_token: str
    scheduler_tick: int
    staleness_ticks: int
    execution: CandidateExecution

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dispatch_token": self.dispatch_token,
            "scheduler_tick": self.scheduler_tick,
            "staleness_ticks": self.staleness_ticks,
            "execution": self.execution.snapshot(),
        }


@dataclass(frozen=True)
class PartialRolloutSliceResult:
    execution_id: str
    scheduler_tick: int
    event_cost: int
    status: str
    event_budget_remaining: int
    state_digest: str
    staleness_ticks: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class _RolloutSlot:
    execution: CandidateExecution
    registration_order: int
    slices_run: int = 0
    paused_at_tick: Optional[int] = None
    last_staleness_ticks: int = 0


class BoundedPartialRolloutScheduler:
    """Run sparse candidate trajectories in deterministic equal-turn slices."""

    def __init__(
        self,
        *,
        max_trajectories: int = 4,
        max_slice_events: int = 4,
        max_staleness_ticks: int = 4,
        max_total_state_bytes: int = 16384,
    ) -> None:
        self.max_trajectories = max(1, int(max_trajectories))
        self.max_slice_events = max(1, int(max_slice_events))
        self.max_staleness_ticks = max(1, int(max_staleness_ticks))
        self.max_total_state_bytes = max(128, int(max_total_state_bytes))
        if self.max_trajectories > self.max_staleness_ticks:
            raise PartialRolloutError("staleness_bound_cannot_cover_queue")
        self._slots: Dict[str, _RolloutSlot] = {}
        self._tick = 0
        self._in_flight: Optional[PartialRolloutDispatch] = None

    def register(self, execution: CandidateExecution) -> None:
        if execution.status != "active" or execution.read_only:
            raise PartialRolloutError("candidate_not_schedulable")
        if execution.execution_id in self._slots:
            raise PartialRolloutError("duplicate_rollout_execution_id")
        if len(self._slots) >= self.max_trajectories:
            raise PartialRolloutError("rollout_trajectory_budget_exceeded")
        projected_bytes = self._total_state_bytes() + len(
            execution.state_json.encode("utf-8")
        )
        if projected_bytes > self.max_total_state_bytes:
            raise PartialRolloutError("rollout_state_budget_exceeded")
        self._slots[execution.execution_id] = _RolloutSlot(
            execution=execution,
            registration_order=len(self._slots),
        )

    def execution(self, execution_id: str) -> CandidateExecution:
        slot = self._slots.get(str(execution_id))
        if slot is None:
            raise PartialRolloutError("unknown_rollout_execution")
        return slot.execution

    def dispatch_next(
        self,
        resume_contexts: Mapping[str, RolloutResumeContext],
    ) -> PartialRolloutDispatch:
        if self._in_flight is not None:
            raise PartialRolloutError("rollout_dispatch_already_in_flight")
        candidates = [
            slot
            for slot in self._slots.values()
            if slot.execution.status != "completed"
        ]
        if not candidates:
            raise PartialRolloutError("no_schedulable_rollout")
        slot = min(
            candidates,
            key=lambda item: (item.slices_run, item.registration_order),
        )
        next_tick = self._tick + 1
        staleness_ticks = 0
        execution = slot.execution
        if execution.status == "paused":
            if slot.paused_at_tick is None:
                raise PartialRolloutError("missing_rollout_pause_tick")
            staleness_ticks = next_tick - slot.paused_at_tick
            if staleness_ticks > self.max_staleness_ticks:
                raise PartialRolloutError("rollout_staleness_budget_exceeded")
            context = resume_contexts.get(execution.execution_id)
            if context is None:
                raise PartialRolloutError("missing_rollout_resume_context")
            try:
                execution = execution.resume(
                    goal=context.goal,
                    plan=context.plan,
                    source_revision=context.source_revision,
                    state_digest=context.state_digest,
                    event_budget_remaining=context.event_budget_remaining,
                    sandbox_checkpoint_identity=(
                        context.sandbox_checkpoint_identity
                    ),
                )
            except CandidateExecutionError as exc:
                raise PartialRolloutError(str(exc)) from exc
        elif execution.status != "active":
            raise PartialRolloutError("candidate_not_schedulable")

        dispatch_token = evidence_digest(
            {
                "execution_id": execution.execution_id,
                "scheduler_tick": next_tick,
                "transition_index": execution.transition_index,
                "state_digest": execution.state_digest,
                "event_budget_remaining": execution.event_budget_remaining,
                "sandbox_checkpoint_identity": (
                    execution.sandbox_checkpoint_identity
                ),
            }
        )
        dispatch = PartialRolloutDispatch(
            dispatch_token=dispatch_token,
            scheduler_tick=next_tick,
            staleness_ticks=staleness_ticks,
            execution=execution,
        )
        self._tick = next_tick
        self._in_flight = dispatch
        return dispatch

    def complete_slice(
        self,
        *,
        dispatch_token: str,
        state: Mapping[str, Any],
        event_cost: int,
    ) -> PartialRolloutSliceResult:
        dispatch = self._in_flight
        if dispatch is None or dispatch.dispatch_token != str(dispatch_token):
            raise PartialRolloutError("invalid_rollout_dispatch_token")
        if type(event_cost) is not int:
            raise PartialRolloutError("rollout_slice_event_budget_exceeded")
        cost = event_cost
        if cost < 1 or cost > self.max_slice_events:
            raise PartialRolloutError("rollout_slice_event_budget_exceeded")
        execution = dispatch.execution
        try:
            updated = execution.apply_state(
                state,
                event_cost=cost,
                expected_state_digest=execution.state_digest,
            )
        except CandidateExecutionError as exc:
            raise PartialRolloutError(str(exc)) from exc

        slot = self._slots[execution.execution_id]
        projected_bytes = (
            self._total_state_bytes()
            - len(slot.execution.state_json.encode("utf-8"))
            + len(updated.state_json.encode("utf-8"))
        )
        if projected_bytes > self.max_total_state_bytes:
            raise PartialRolloutError("rollout_state_budget_exceeded")

        if updated.event_budget_remaining == 0:
            updated = updated.complete()
            paused_at_tick = None
        else:
            updated = updated.pause()
            paused_at_tick = self._tick
        slot.execution = updated
        slot.slices_run += 1
        slot.paused_at_tick = paused_at_tick
        slot.last_staleness_ticks = dispatch.staleness_ticks
        self._in_flight = None
        return PartialRolloutSliceResult(
            execution_id=updated.execution_id,
            scheduler_tick=self._tick,
            event_cost=cost,
            status=updated.status,
            event_budget_remaining=updated.event_budget_remaining,
            state_digest=updated.state_digest,
            staleness_ticks=dispatch.staleness_ticks,
        )

    def snapshot(self) -> Dict[str, Any]:
        payload = {
            "schema": "sara-partial-rollout-scheduler-v1",
            "scheduler_tick": self._tick,
            "max_trajectories": self.max_trajectories,
            "max_slice_events": self.max_slice_events,
            "max_staleness_ticks": self.max_staleness_ticks,
            "max_total_state_bytes": self.max_total_state_bytes,
            "total_state_bytes": self._total_state_bytes(),
            "in_flight": (
                self._in_flight.to_dict()
                if self._in_flight is not None
                else None
            ),
            "trajectories": [
                {
                    "registration_order": slot.registration_order,
                    "slices_run": slot.slices_run,
                    "paused_at_tick": slot.paused_at_tick,
                    "last_staleness_ticks": slot.last_staleness_ticks,
                    "execution": slot.execution.snapshot(),
                }
                for slot in sorted(
                    self._slots.values(),
                    key=lambda item: item.registration_order,
                )
            ],
        }
        payload["snapshot_digest"] = evidence_digest(payload)
        return payload

    def _total_state_bytes(self) -> int:
        return sum(
            len(slot.execution.state_json.encode("utf-8"))
            for slot in self._slots.values()
        )


__all__ = [
    "BoundedPartialRolloutScheduler",
    "PartialRolloutDispatch",
    "PartialRolloutError",
    "PartialRolloutSliceResult",
    "RolloutResumeContext",
]
