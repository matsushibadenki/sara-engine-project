"""In-memory isolated reversible sandbox for bounded agent tool edits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Dict, Mapping, Optional, Sequence

from sara_engine.memory.verification_receipt import evidence_digest

from .bounded_agent_loop import AgentPlanDecision
from .tool_result_pairing import IndexedToolCall, IndexedToolResult
from .transactional_tools import (
    BoundedTransactionalToolAdapter,
    TransactionalToolRequest,
    TransactionalToolResult,
)


class ReversibleToolSandboxError(ValueError):
    """Raised when sandbox isolation or checkpoint integrity is violated."""


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
        raise ReversibleToolSandboxError("sandbox_state_not_json_compatible") from exc


def _decode(value: str) -> Dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ReversibleToolSandboxError("sandbox_state_not_mapping")
    return decoded


def _checkpoint_identity(
    *,
    sandbox_id: str,
    revision: int,
    state_digest: str,
) -> str:
    return evidence_digest(
        {
            "sandbox_id": sandbox_id,
            "revision": revision,
            "state_digest": state_digest,
        }
    )


@dataclass(frozen=True)
class SandboxCheckpoint:
    sandbox_id: str
    revision: int
    state_json: str
    state_digest: str
    checkpoint_identity: str

    @property
    def state(self) -> Dict[str, Any]:
        return _decode(self.state_json)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sandbox_id": self.sandbox_id,
            "revision": self.revision,
            "state": self.state,
            "state_digest": self.state_digest,
            "checkpoint_identity": self.checkpoint_identity,
        }


@dataclass(frozen=True)
class SandboxExecutionResult:
    tool_result: TransactionalToolResult
    before_checkpoint: SandboxCheckpoint
    after_checkpoint: SandboxCheckpoint
    sandbox_revision_advanced: bool
    external_side_effects_executed: bool = False
    durable_mutation_allowed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "sara-reversible-tool-sandbox-execution-v1",
            "tool_result": self.tool_result.to_dict(),
            "before_checkpoint": self.before_checkpoint.to_dict(),
            "after_checkpoint": self.after_checkpoint.to_dict(),
            "sandbox_revision_advanced": self.sandbox_revision_advanced,
            "external_side_effects_executed": self.external_side_effects_executed,
            "durable_mutation_allowed": self.durable_mutation_allowed,
        }


@dataclass(frozen=True)
class SandboxRollbackResult:
    restored: bool
    requested_checkpoint_identity: str
    before_digest: str
    restored_digest: str
    byte_equivalent_restoration: bool
    revision: int
    external_side_effects_executed: bool = False
    durable_mutation_allowed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "sara-reversible-tool-sandbox-rollback-v1",
            **asdict(self),
        }


class IsolatedReversibleToolSandbox:
    """Own a private JSON state with bounded execution and exact rollback."""

    def __init__(
        self,
        initial_state: Mapping[str, Any],
        *,
        sandbox_id: str,
        allowed_tools: Sequence[str],
        max_operations: int = 16,
        max_edits: int = 8,
        max_event_cost: int = 16,
        max_state_bytes: int = 4096,
    ) -> None:
        if not str(sandbox_id).strip():
            raise ReversibleToolSandboxError("missing_sandbox_id")
        if type(max_operations) is not int or max_operations < 1:
            raise ReversibleToolSandboxError("invalid_sandbox_operation_budget")
        self.sandbox_id = str(sandbox_id)
        self.max_operations = max_operations
        if type(max_state_bytes) is not int or max_state_bytes < 128:
            raise ReversibleToolSandboxError("invalid_sandbox_state_budget")
        self.max_state_bytes = max_state_bytes
        state_json = _canonical_json(initial_state)
        if len(state_json.encode("utf-8")) > self.max_state_bytes:
            raise ReversibleToolSandboxError("sandbox_state_budget_exceeded")
        self._state = _decode(state_json)
        self._revision = 0
        self._operation_count = 0
        self._adapter = BoundedTransactionalToolAdapter(
            allowed_tools=allowed_tools,
            max_edits=max_edits,
            max_event_cost=max_event_cost,
            max_state_bytes=self.max_state_bytes,
        )

    @property
    def state(self) -> Dict[str, Any]:
        return _decode(_canonical_json(self._state))

    @property
    def operation_count(self) -> int:
        return self._operation_count

    def checkpoint(self) -> SandboxCheckpoint:
        state_json = _canonical_json(self._state)
        state_digest = evidence_digest(_decode(state_json))
        return SandboxCheckpoint(
            sandbox_id=self.sandbox_id,
            revision=self._revision,
            state_json=state_json,
            state_digest=state_digest,
            checkpoint_identity=_checkpoint_identity(
                sandbox_id=self.sandbox_id,
                revision=self._revision,
                state_digest=state_digest,
            ),
        )

    def execute(
        self,
        *,
        plan: AgentPlanDecision,
        request: TransactionalToolRequest,
        observed_outcome: str,
        calls: Optional[Sequence[IndexedToolCall]] = None,
        results: Optional[Sequence[IndexedToolResult]] = None,
    ) -> SandboxExecutionResult:
        self._consume_operation()
        if (calls is None) != (results is None):
            raise ReversibleToolSandboxError("incomplete_sandbox_tool_pairing")
        before = self.checkpoint()
        if calls is not None and results is not None:
            tool_result = self._adapter.execute_paired(
                self._state,
                plan=plan,
                request=request,
                observed_outcome=observed_outcome,
                calls=calls,
                results=results,
            )
        else:
            tool_result = self._adapter.execute(
                self._state,
                plan=plan,
                request=request,
                observed_outcome=observed_outcome,
            )
        revision_advanced = bool(tool_result.committed)
        if revision_advanced:
            self._revision += 1
        after = self.checkpoint()
        return SandboxExecutionResult(
            tool_result=tool_result,
            before_checkpoint=before,
            after_checkpoint=after,
            sandbox_revision_advanced=revision_advanced,
        )

    def rollback(self, checkpoint: SandboxCheckpoint) -> SandboxRollbackResult:
        self._consume_operation()
        self._validate_checkpoint(checkpoint)
        before_digest = evidence_digest(self._state)
        restored = checkpoint.state
        if len(checkpoint.state_json.encode("utf-8")) > self.max_state_bytes:
            raise ReversibleToolSandboxError("sandbox_state_budget_exceeded")
        self._state = restored
        self._revision += 1
        restored_digest = evidence_digest(self._state)
        return SandboxRollbackResult(
            restored=True,
            requested_checkpoint_identity=checkpoint.checkpoint_identity,
            before_digest=before_digest,
            restored_digest=restored_digest,
            byte_equivalent_restoration=(
                restored_digest == checkpoint.state_digest
                and _canonical_json(self._state) == checkpoint.state_json
            ),
            revision=self._revision,
        )

    def snapshot(self) -> Dict[str, Any]:
        checkpoint = self.checkpoint()
        payload = {
            "schema": "sara-reversible-tool-sandbox-v1",
            "sandbox_id": self.sandbox_id,
            "revision": self._revision,
            "operation_count": self._operation_count,
            "max_operations": self.max_operations,
            "max_state_bytes": self.max_state_bytes,
            "state": checkpoint.state,
            "state_digest": checkpoint.state_digest,
            "checkpoint_identity": checkpoint.checkpoint_identity,
            "external_side_effects_enabled": False,
            "durable_mutation_allowed": False,
        }
        payload["snapshot_digest"] = evidence_digest(payload)
        return payload

    def _consume_operation(self) -> None:
        if self._operation_count >= self.max_operations:
            raise ReversibleToolSandboxError("sandbox_operation_budget_exceeded")
        self._operation_count += 1

    def _validate_checkpoint(self, checkpoint: SandboxCheckpoint) -> None:
        if checkpoint.sandbox_id != self.sandbox_id:
            raise ReversibleToolSandboxError("foreign_sandbox_checkpoint")
        if len(checkpoint.state_json.encode("utf-8")) > self.max_state_bytes:
            raise ReversibleToolSandboxError("sandbox_state_budget_exceeded")
        decoded_state = checkpoint.state
        state_digest = evidence_digest(decoded_state)
        expected_identity = _checkpoint_identity(
            sandbox_id=checkpoint.sandbox_id,
            revision=checkpoint.revision,
            state_digest=checkpoint.state_digest,
        )
        if (
            checkpoint.revision > self._revision
            or checkpoint.state_digest != state_digest
            or checkpoint.checkpoint_identity != expected_identity
            or _canonical_json(decoded_state) != checkpoint.state_json
        ):
            raise ReversibleToolSandboxError("invalid_sandbox_checkpoint")


__all__ = [
    "IsolatedReversibleToolSandbox",
    "ReversibleToolSandboxError",
    "SandboxCheckpoint",
    "SandboxExecutionResult",
    "SandboxRollbackResult",
]
