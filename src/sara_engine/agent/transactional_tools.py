"""Bounded transactional tool-state edits for verified agent plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

from sara_engine.memory.verification_receipt import evidence_digest

from .bounded_agent_loop import AgentPlanDecision


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_value(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _json_value(item) for key, item in value.items())
    return False


@dataclass(frozen=True)
class ToolStateEdit:
    operation: str
    key: str
    value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransactionalToolRequest:
    request_id: str
    tool_name: str
    goal: str
    expected_outcome: str
    rollback_action: str
    source_ref: str
    edits: Tuple[ToolStateEdit, ...]
    observed: bool = True
    verified: bool = True
    event_cost: int = 1

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["edits"] = [item.to_dict() for item in self.edits]
        return payload


@dataclass(frozen=True)
class TransactionalToolResult:
    request_id: str
    tool_name: str
    executed: bool
    committed: bool
    rolled_back: bool
    decision: str
    before_digest: str
    after_digest: str
    restored_digest: str
    edit_count: int
    event_cost: int
    state_bytes: int
    durable_mutation_allowed: bool
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "sara-transactional-tool-result-v1",
            **asdict(self),
            "trace": dict(self.trace),
        }


class BoundedTransactionalToolAdapter:
    """Apply verified JSON state edits atomically within declared hard budgets."""

    def __init__(
        self,
        *,
        allowed_tools: Sequence[str],
        max_edits: int = 8,
        max_event_cost: int = 16,
        max_state_bytes: int = 4096,
    ) -> None:
        self.allowed_tools = frozenset(
            str(item).strip() for item in allowed_tools if str(item).strip()
        )
        self.max_edits = max(1, int(max_edits))
        self.max_event_cost = max(1, int(max_event_cost))
        self.max_state_bytes = max(128, int(max_state_bytes))

    def execute(
        self,
        state: MutableMapping[str, Any],
        *,
        plan: AgentPlanDecision,
        request: TransactionalToolRequest,
        observed_outcome: str,
    ) -> TransactionalToolResult:
        before = dict(state)
        before_digest = evidence_digest(before)
        errors = self._validate(plan=plan, request=request, state=before)
        if type(state) is not dict:
            errors.append("unsupported_tool_state_mapping")
            errors = sorted(set(errors))
        if errors:
            return self._result(
                request=request,
                executed=False,
                committed=False,
                rolled_back=False,
                decision="reject_tool_request",
                before_digest=before_digest,
                after_digest=before_digest,
                restored_digest=before_digest,
                state_bytes=len(_json_bytes(before)) if _json_value(before) else 0,
                trace={"errors": errors, "side_effects_executed": False},
            )

        staged = dict(before)
        staged_error = ""
        for edit in request.edits:
            if edit.operation == "set":
                staged[edit.key] = edit.value
            elif edit.operation == "delete":
                if edit.key not in staged:
                    staged_error = "delete_key_missing"
                    break
                del staged[edit.key]
            if len(_json_bytes(staged)) > self.max_state_bytes:
                staged_error = "tool_state_budget_exceeded"
                break

        staged_digest = evidence_digest(staged)
        outcome_matches = str(observed_outcome) == request.expected_outcome
        if staged_error or not outcome_matches:
            restored_digest = evidence_digest(state)
            return self._result(
                request=request,
                executed=True,
                committed=False,
                rolled_back=True,
                decision=(
                    "rollback_staged_tool_error"
                    if staged_error
                    else "rollback_unexpected_tool_outcome"
                ),
                before_digest=before_digest,
                after_digest=staged_digest,
                restored_digest=restored_digest,
                state_bytes=len(_json_bytes(before)),
                trace={
                    "errors": [staged_error] if staged_error else [],
                    "outcome_matches": outcome_matches,
                    "rollback_action": request.rollback_action,
                    "source_ref": request.source_ref,
                    "side_effects_executed": False,
                    "byte_equivalent_restoration": restored_digest == before_digest,
                },
            )

        state.clear()
        state.update(staged)
        committed_digest = evidence_digest(state)
        return self._result(
            request=request,
            executed=True,
            committed=True,
            rolled_back=False,
            decision="commit_verified_tool_outcome",
            before_digest=before_digest,
            after_digest=committed_digest,
            restored_digest="",
            state_bytes=len(_json_bytes(dict(state))),
            trace={
                "errors": [],
                "outcome_matches": True,
                "rollback_action": request.rollback_action,
                "source_ref": request.source_ref,
                "side_effects_executed": False,
                "operational_state_committed": True,
            },
        )

    def _validate(
        self,
        *,
        plan: AgentPlanDecision,
        request: TransactionalToolRequest,
        state: Mapping[str, Any],
    ) -> list[str]:
        errors = []
        if not plan.accepted:
            errors.append("plan_not_accepted")
        if request.tool_name not in self.allowed_tools:
            errors.append("tool_not_allowed")
        if request.goal != plan.goal:
            errors.append("tool_goal_mismatch")
        if request.expected_outcome != plan.expected_outcome:
            errors.append("tool_outcome_mismatch")
        if request.rollback_action != plan.rollback_action:
            errors.append("tool_rollback_mismatch")
        if not request.source_ref or not request.observed or not request.verified:
            errors.append("tool_evidence_not_verified")
        if not request.request_id:
            errors.append("missing_tool_request_id")
        if not request.edits or len(request.edits) > self.max_edits:
            errors.append("tool_edit_budget_exceeded")
        if request.event_cost < 1 or request.event_cost > self.max_event_cost:
            errors.append("tool_event_budget_exceeded")
        keys = [item.key for item in request.edits]
        if len(keys) != len(set(keys)):
            errors.append("duplicate_tool_edit_key")
        if any(
            item.operation not in {"set", "delete"}
            or not item.key
            or (item.operation == "set" and not _json_value(item.value))
            for item in request.edits
        ):
            errors.append("malformed_tool_edit")
        if not _json_value(dict(state)):
            errors.append("tool_state_not_json_compatible")
        elif len(_json_bytes(state)) > self.max_state_bytes:
            errors.append("tool_state_budget_exceeded")
        return sorted(set(errors))

    @staticmethod
    def _result(
        *,
        request: TransactionalToolRequest,
        executed: bool,
        committed: bool,
        rolled_back: bool,
        decision: str,
        before_digest: str,
        after_digest: str,
        restored_digest: str,
        state_bytes: int,
        trace: Dict[str, Any],
    ) -> TransactionalToolResult:
        return TransactionalToolResult(
            request_id=request.request_id,
            tool_name=request.tool_name,
            executed=executed,
            committed=committed,
            rolled_back=rolled_back,
            decision=decision,
            before_digest=before_digest,
            after_digest=after_digest,
            restored_digest=restored_digest,
            edit_count=len(request.edits),
            event_cost=request.event_cost,
            state_bytes=state_bytes,
            durable_mutation_allowed=False,
            trace=trace,
        )


__all__ = [
    "BoundedTransactionalToolAdapter",
    "ToolStateEdit",
    "TransactionalToolRequest",
    "TransactionalToolResult",
]
