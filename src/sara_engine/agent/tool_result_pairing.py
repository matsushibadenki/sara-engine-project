"""Bounded typed pairing for parallel agent tool calls and results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from sara_engine.memory.verification_receipt import evidence_digest


_RESULT_TYPES = {
    "null": type(None),
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "object": dict,
}


def _json_value(value: Any) -> bool:
    if value is None or type(value) in {bool, int, str}:
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_json_value(item) for item in value)
    if type(value) is dict:
        return all(
            isinstance(key, str) and _json_value(item)
            for key, item in value.items()
        )
    return False


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class IndexedToolCall:
    index: int
    call_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    expected_result_type: str
    event_cost: int = 1

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["arguments"] = dict(self.arguments)
        return payload


@dataclass(frozen=True)
class IndexedToolResult:
    index: int
    call_id: str
    tool_name: str
    value: Any
    success: bool = True
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolPairingValidation:
    commit_allowed: bool
    errors: Tuple[str, ...]
    call_count: int
    result_count: int
    event_cost: int
    state_bytes: int
    pairing_digest: str
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "sara-indexed-tool-pairing-v1",
            **asdict(self),
            "errors": list(self.errors),
            "trace": dict(self.trace),
        }


class IndexedToolResultPairingGate:
    """Validate exact typed pairing before a parallel tool batch may commit."""

    def __init__(
        self,
        *,
        max_calls: int = 8,
        max_event_cost: int = 16,
        max_state_bytes: int = 4096,
    ) -> None:
        self.max_calls = max(1, int(max_calls))
        self.max_event_cost = max(1, int(max_event_cost))
        self.max_state_bytes = max(128, int(max_state_bytes))

    def validate(
        self,
        calls: Sequence[IndexedToolCall],
        results: Sequence[IndexedToolResult],
    ) -> ToolPairingValidation:
        call_items = tuple(calls)
        result_items = tuple(results)
        errors: list[str] = []

        if not call_items or len(call_items) > self.max_calls:
            errors.append("tool_call_count_exceeded")
        expected_indexes = list(range(len(call_items)))
        call_indexes = [item.index for item in call_items]
        call_ids = [item.call_id for item in call_items]
        if call_indexes != expected_indexes:
            errors.append("noncanonical_tool_call_indexes")
        if len(call_indexes) != len(set(call_indexes)):
            errors.append("duplicate_tool_call_index")
        if len(call_ids) != len(set(call_ids)):
            errors.append("duplicate_tool_call_id")

        event_cost = sum(
            item.event_cost for item in call_items if type(item.event_cost) is int
        )
        if any(type(item.event_cost) is not int or item.event_cost < 1 for item in call_items):
            errors.append("invalid_tool_call_event_cost")
        if event_cost > self.max_event_cost:
            errors.append("tool_call_event_budget_exceeded")
        if any(
            not item.call_id
            or not item.tool_name
            or item.expected_result_type not in _RESULT_TYPES
            or not isinstance(item.arguments, Mapping)
            or not _json_value(dict(item.arguments))
            for item in call_items
        ):
            errors.append("malformed_typed_tool_call")

        result_indexes = [item.index for item in result_items]
        result_ids = [item.call_id for item in result_items]
        if len(result_indexes) != len(set(result_indexes)):
            errors.append("duplicate_tool_result_index")
        if len(result_ids) != len(set(result_ids)):
            errors.append("duplicate_tool_result_id")
        if len(result_items) < len(call_items):
            errors.append("missing_tool_result")
        elif len(result_items) > len(call_items):
            errors.append("unexpected_tool_result")

        call_id_set = set(call_ids)
        result_id_set = set(result_ids)
        if call_id_set - result_id_set:
            errors.append("missing_tool_result")
        if result_id_set - call_id_set:
            errors.append("unexpected_tool_result")

        for position, (call, result) in enumerate(zip(call_items, result_items)):
            if (
                result.index != call.index
                or result.call_id != call.call_id
                or result.tool_name != call.tool_name
            ):
                errors.append("reordered_tool_results")
                continue
            if result.index != position:
                errors.append("noncanonical_tool_result_indexes")
            if not result.success:
                errors.append("tool_result_failed")
            expected_type = _RESULT_TYPES.get(call.expected_result_type)
            if (
                expected_type is None
                or type(result.value) is not expected_type
                or not _json_value(result.value)
            ):
                errors.append("tool_result_type_mismatch")

        payload = {
            "calls": [item.to_dict() for item in call_items],
            "results": [item.to_dict() for item in result_items],
        }
        state_bytes = 0
        try:
            state_bytes = len(_json_bytes(payload))
        except (TypeError, ValueError):
            errors.append("tool_pairing_not_json_compatible")
        if state_bytes > self.max_state_bytes:
            errors.append("tool_pairing_state_budget_exceeded")

        unique_errors = tuple(sorted(set(errors)))
        return ToolPairingValidation(
            commit_allowed=not unique_errors,
            errors=unique_errors,
            call_count=len(call_items),
            result_count=len(result_items),
            event_cost=event_cost,
            state_bytes=state_bytes,
            pairing_digest=evidence_digest(payload)
            if state_bytes and not unique_errors
            else "",
            trace={
                "call_indexes": call_indexes,
                "result_indexes": result_indexes,
                "call_ids": call_ids,
                "result_ids": result_ids,
                "side_effects_executed": False,
                "durable_mutation_allowed": False,
            },
        )


__all__ = [
    "IndexedToolCall",
    "IndexedToolResult",
    "IndexedToolResultPairingGate",
    "ToolPairingValidation",
]
