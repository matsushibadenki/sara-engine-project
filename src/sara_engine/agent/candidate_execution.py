"""Immutable resumable execution state for bounded agent candidates."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any, Dict, Mapping

from sara_engine.memory.verification_receipt import evidence_digest


class CandidateExecutionError(ValueError):
    """Raised when a candidate execution violates its transition contract."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CandidateExecutionError("candidate_state_not_json_compatible") from exc


def _decode(value: str) -> Dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise CandidateExecutionError("candidate_state_not_mapping")
    return decoded


@dataclass(frozen=True)
class CandidateExecution:
    """A functional state machine whose transitions return isolated copies."""

    execution_id: str
    parent_execution_id: str
    goal: str
    plan_json: str
    source_revision: str
    state_json: str
    state_digest: str
    event_budget_total: int
    event_budget_remaining: int
    sandbox_checkpoint_identity: str
    status: str = "active"
    transition_index: int = 0
    read_only: bool = False
    max_state_bytes: int = 4096

    @classmethod
    def create(
        cls,
        *,
        execution_id: str,
        goal: str,
        plan: Mapping[str, Any],
        source_revision: str,
        state: Mapping[str, Any],
        event_budget: int,
        sandbox_checkpoint_identity: str,
        max_state_bytes: int = 4096,
    ) -> "CandidateExecution":
        if not str(execution_id).strip():
            raise CandidateExecutionError("missing_execution_id")
        if not str(goal).strip():
            raise CandidateExecutionError("missing_execution_goal")
        if not str(source_revision).strip():
            raise CandidateExecutionError("missing_source_revision")
        if not str(sandbox_checkpoint_identity).strip():
            raise CandidateExecutionError("missing_sandbox_checkpoint_identity")
        if type(event_budget) is not int or event_budget < 1:
            raise CandidateExecutionError("invalid_event_budget")
        if type(max_state_bytes) is not int or max_state_bytes < 128:
            raise CandidateExecutionError("invalid_state_budget")

        plan_json = _canonical_json(plan)
        state_json = _canonical_json(state)
        if len(state_json.encode("utf-8")) > int(max_state_bytes):
            raise CandidateExecutionError("candidate_state_budget_exceeded")
        return cls(
            execution_id=str(execution_id),
            parent_execution_id="",
            goal=str(goal),
            plan_json=plan_json,
            source_revision=str(source_revision),
            state_json=state_json,
            state_digest=evidence_digest(_decode(state_json)),
            event_budget_total=event_budget,
            event_budget_remaining=event_budget,
            sandbox_checkpoint_identity=str(sandbox_checkpoint_identity),
            max_state_bytes=max_state_bytes,
        )

    @property
    def plan(self) -> Dict[str, Any]:
        return _decode(self.plan_json)

    @property
    def state(self) -> Dict[str, Any]:
        return _decode(self.state_json)

    @property
    def plan_digest(self) -> str:
        return evidence_digest(self.plan)

    def pause(self) -> "CandidateExecution":
        self._require_status("active")
        return replace(
            self,
            status="paused",
            transition_index=self.transition_index + 1,
        )

    def resume(
        self,
        *,
        goal: str,
        plan: Mapping[str, Any],
        source_revision: str,
        state_digest: str,
        event_budget_remaining: int,
        sandbox_checkpoint_identity: str,
    ) -> "CandidateExecution":
        self._require_status("paused")
        checks = {
            "stale_execution_goal": str(goal) != self.goal,
            "stale_execution_plan": evidence_digest(dict(plan)) != self.plan_digest,
            "stale_source_revision": str(source_revision) != self.source_revision,
            "stale_execution_state": str(state_digest) != self.state_digest,
            "stale_event_budget": (
                type(event_budget_remaining) is not int
                or event_budget_remaining != self.event_budget_remaining
            ),
            "stale_sandbox_checkpoint": str(sandbox_checkpoint_identity)
            != self.sandbox_checkpoint_identity,
        }
        errors = [name for name, failed in checks.items() if failed]
        if errors:
            raise CandidateExecutionError(",".join(errors))
        return replace(
            self,
            status="active",
            transition_index=self.transition_index + 1,
        )

    def fork_for_judging(self, *, execution_id: str) -> "CandidateExecution":
        self._require_status("paused")
        if not str(execution_id).strip():
            raise CandidateExecutionError("missing_execution_id")
        if str(execution_id) == self.execution_id:
            raise CandidateExecutionError("duplicate_execution_id")
        return replace(
            self,
            execution_id=str(execution_id),
            parent_execution_id=self.execution_id,
            status="judging",
            transition_index=self.transition_index + 1,
            read_only=True,
        )

    def apply_state(
        self,
        state: Mapping[str, Any],
        *,
        event_cost: int,
        expected_state_digest: str,
    ) -> "CandidateExecution":
        self._require_status("active")
        if self.read_only:
            raise CandidateExecutionError("read_only_judging_fork")
        if str(expected_state_digest) != self.state_digest:
            raise CandidateExecutionError("stale_execution_state")
        if type(event_cost) is not int:
            raise CandidateExecutionError("candidate_event_budget_exceeded")
        cost = event_cost
        if cost < 1 or cost > self.event_budget_remaining:
            raise CandidateExecutionError("candidate_event_budget_exceeded")
        state_json = _canonical_json(state)
        if len(state_json.encode("utf-8")) > self.max_state_bytes:
            raise CandidateExecutionError("candidate_state_budget_exceeded")
        return replace(
            self,
            state_json=state_json,
            state_digest=evidence_digest(_decode(state_json)),
            event_budget_remaining=self.event_budget_remaining - cost,
            transition_index=self.transition_index + 1,
        )

    def complete(self) -> "CandidateExecution":
        self._require_status("active")
        if self.event_budget_remaining != 0:
            raise CandidateExecutionError("candidate_event_budget_remaining")
        return replace(
            self,
            status="completed",
            transition_index=self.transition_index + 1,
        )

    def snapshot(self) -> Dict[str, Any]:
        payload = {
            "schema": "sara-candidate-execution-snapshot-v1",
            "execution_id": self.execution_id,
            "parent_execution_id": self.parent_execution_id,
            "goal": self.goal,
            "plan": self.plan,
            "plan_digest": self.plan_digest,
            "source_revision": self.source_revision,
            "state": self.state,
            "state_digest": self.state_digest,
            "event_budget_total": self.event_budget_total,
            "event_budget_remaining": self.event_budget_remaining,
            "sandbox_checkpoint_identity": self.sandbox_checkpoint_identity,
            "status": self.status,
            "transition_index": self.transition_index,
            "read_only": self.read_only,
            "max_state_bytes": self.max_state_bytes,
        }
        payload["snapshot_digest"] = evidence_digest(payload)
        return payload

    def _require_status(self, expected: str) -> None:
        if self.status != expected:
            if self.read_only:
                raise CandidateExecutionError("read_only_judging_fork")
            raise CandidateExecutionError(
                f"invalid_candidate_transition:{self.status}->{expected}"
            )


__all__ = ["CandidateExecution", "CandidateExecutionError"]
