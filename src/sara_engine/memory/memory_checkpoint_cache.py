"""Default-off bounded sparse memory checkpoint cache for Phase 34."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from sara_engine.memory.verification_receipt import evidence_digest


SCHEMA = "sara-bounded-memory-checkpoint-cache-v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _bounded_strings(values: Iterable[str], limit: int) -> Tuple[str, ...]:
    normalized = sorted({str(value).strip() for value in values if str(value).strip()})
    return tuple(normalized[:limit])


@dataclass(frozen=True)
class MemoryCheckpointCandidate:
    """Observed sparse state proposed at one verified semantic boundary."""

    event_start: int
    event_end: int
    summary_ids: Tuple[str, ...]
    source_refs: Tuple[str, ...]
    source_revision: str
    state_group_id: str
    parent_digest: str
    runtime_fingerprint: str
    schema_fingerprint: str
    semantic_boundary: bool = True
    observed: bool = True
    verified: bool = True
    contradicted: bool = False
    expires_at: Optional[int] = None


@dataclass(frozen=True)
class MemoryCheckpoint:
    """Immutable evidence reference; it never contains mutable historical state."""

    checkpoint_id: str
    event_start: int
    event_end: int
    summary_ids: Tuple[str, ...]
    source_refs: Tuple[str, ...]
    source_revision: str
    state_group_id: str
    parent_digests: Tuple[str, ...]
    runtime_fingerprint: str
    schema_fingerprint: str
    expires_at: Optional[int]
    verified: bool = True
    contradicted: bool = False

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "event_start": self.event_start,
            "event_end": self.event_end,
            "summary_ids": list(self.summary_ids),
            "source_refs": list(self.source_refs),
            "source_revision": self.source_revision,
            "state_group_id": self.state_group_id,
            "parent_digests": list(self.parent_digests),
            "runtime_fingerprint": self.runtime_fingerprint,
            "schema_fingerprint": self.schema_fingerprint,
            "expires_at": self.expires_at,
            "verified": self.verified,
            "contradicted": self.contradicted,
        }

    def is_valid(self) -> bool:
        return self.checkpoint_id == evidence_digest(self.identity_payload())

    def to_dict(self) -> Dict[str, Any]:
        return {"checkpoint_id": self.checkpoint_id, **self.identity_payload()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemoryCheckpoint":
        checkpoint = cls(
            checkpoint_id=str(value.get("checkpoint_id", "")),
            event_start=int(value.get("event_start", -1)),
            event_end=int(value.get("event_end", -1)),
            summary_ids=tuple(str(item) for item in value.get("summary_ids", ())),
            source_refs=tuple(str(item) for item in value.get("source_refs", ())),
            source_revision=str(value.get("source_revision", "")),
            state_group_id=str(value.get("state_group_id", "")),
            parent_digests=tuple(str(item) for item in value.get("parent_digests", ())),
            runtime_fingerprint=str(value.get("runtime_fingerprint", "")),
            schema_fingerprint=str(value.get("schema_fingerprint", "")),
            expires_at=(
                int(value["expires_at"])
                if value.get("expires_at") is not None
                else None
            ),
            verified=bool(value.get("verified", False)),
            contradicted=bool(value.get("contradicted", False)),
        )
        if not checkpoint.is_valid():
            raise ValueError("memory_checkpoint_identity_mismatch")
        return checkpoint


@dataclass(frozen=True)
class CheckpointOperationResult:
    accepted: bool
    decision: str
    checkpoint_id: str = ""
    evicted_ids: Tuple[str, ...] = ()
    merged_ids: Tuple[str, ...] = ()
    durable_mutation_allowed: bool = False


@dataclass(frozen=True)
class CheckpointRetrievalResult:
    abstained: bool
    decision: str
    evidence: Tuple[Dict[str, Any], ...]
    selected_checkpoint_ids: Tuple[str, ...]
    scanned_checkpoints: int
    event_cost: int
    durable_mutation_allowed: bool = False


class BoundedSparseMemoryCheckpointCache:
    """Keep sparse evidence checkpoints under hard deterministic ceilings."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        retention_profile: str = "equal",
        max_checkpoints: int = 8,
        selected_k: int = 2,
        max_summary_ids: int = 8,
        max_state_bytes: int = 8192,
        max_event_cost: int = 256,
        max_merges_per_event: int = 2,
    ) -> None:
        if retention_profile not in {"equal", "logarithmic"}:
            raise ValueError("unsupported_checkpoint_retention_profile")
        limits = (
            max_checkpoints,
            selected_k,
            max_summary_ids,
            max_state_bytes,
            max_event_cost,
            max_merges_per_event,
        )
        if any(type(value) is not int or value < 1 for value in limits):
            raise ValueError("invalid_memory_checkpoint_cache_limit")
        if selected_k > max_checkpoints:
            raise ValueError("selected_k_exceeds_checkpoint_budget")
        self.enabled = bool(enabled)
        self.retention_profile = retention_profile
        self.max_checkpoints = max_checkpoints
        self.selected_k = selected_k
        self.max_summary_ids = max_summary_ids
        self.max_state_bytes = max_state_bytes
        self.max_event_cost = max_event_cost
        self.max_merges_per_event = max_merges_per_event
        self._checkpoints: Dict[str, MemoryCheckpoint] = {}
        self.admission_count = 0
        self.eviction_count = 0
        self.merge_count = 0
        self.expiry_count = 0
        self.invalidation_count = 0

    @property
    def checkpoints(self) -> Tuple[MemoryCheckpoint, ...]:
        return tuple(sorted(self._checkpoints.values(), key=self._chronological_key))

    def admit(
        self,
        candidate: MemoryCheckpointCandidate,
        *,
        current_event: Optional[int] = None,
    ) -> CheckpointOperationResult:
        decision = self._admission_decision(candidate, current_event=current_event)
        if decision != "admit":
            return CheckpointOperationResult(False, decision)
        checkpoint = self._checkpoint_from_candidate(candidate)
        if checkpoint.checkpoint_id in self._checkpoints:
            return CheckpointOperationResult(
                True,
                "duplicate_checkpoint_preserved",
                checkpoint.checkpoint_id,
            )
        previous_checkpoints = dict(self._checkpoints)
        previous_evictions = self.eviction_count
        previous_merges = self.merge_count
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        merged_ids: Tuple[str, ...] = ()
        evicted_ids: Tuple[str, ...] = ()
        if len(self._checkpoints) > self.max_checkpoints:
            if self.retention_profile == "logarithmic":
                merged_ids = self._merge_oldest_compatible()
            if len(self._checkpoints) > self.max_checkpoints:
                oldest = min(self._checkpoints.values(), key=self._chronological_key)
                del self._checkpoints[oldest.checkpoint_id]
                self.eviction_count += 1
                evicted_ids = (oldest.checkpoint_id,)
        if self._serialized_size() > self.max_state_bytes:
            self._checkpoints = previous_checkpoints
            self.eviction_count = previous_evictions
            self.merge_count = previous_merges
            return CheckpointOperationResult(False, "state_byte_budget_exceeded")
        self.admission_count += 1
        return CheckpointOperationResult(
            True,
            "admitted",
            checkpoint.checkpoint_id,
            evicted_ids,
            merged_ids,
        )

    def retrieve(
        self,
        query_summary_ids: Sequence[str],
        *,
        source_revision: str,
        runtime_fingerprint: str,
        schema_fingerprint: str,
        current_event: Optional[int] = None,
    ) -> CheckpointRetrievalResult:
        if not self.enabled:
            return self._abstain("checkpoint_cache_disabled", 0, 0)
        query = _bounded_strings(query_summary_ids, self.max_summary_ids)
        if not query:
            return self._abstain("empty_checkpoint_query", 0, 0)
        ordered = self.checkpoints
        scanned = len(ordered)
        event_cost = scanned * len(query)
        if event_cost > self.max_event_cost:
            return self._abstain("retrieval_event_budget_exceeded", scanned, event_cost)
        query_set = set(query)
        stale_match = False
        contradiction_match = False
        eligible = []
        most_recent = max((item.event_end for item in ordered), default=-1)
        for checkpoint in ordered:
            if not query_set.intersection(checkpoint.summary_ids):
                continue
            if checkpoint.contradicted:
                contradiction_match = True
                continue
            if (
                checkpoint.source_revision != str(source_revision)
                or checkpoint.runtime_fingerprint != str(runtime_fingerprint)
                or checkpoint.schema_fingerprint != str(schema_fingerprint)
                or (
                    current_event is not None
                    and checkpoint.expires_at is not None
                    and current_event >= checkpoint.expires_at
                )
            ):
                stale_match = True
                continue
            overlap = len(query_set.intersection(checkpoint.summary_ids))
            recency = 1 if checkpoint.event_end == most_recent else 0
            score = 4 * overlap + 2 * int(checkpoint.verified) + recency
            eligible.append((score, checkpoint))
        eligible.sort(
            key=lambda item: (
                -item[0],
                item[1].event_start,
                item[1].event_end,
                item[1].checkpoint_id,
            )
        )
        selected = tuple(item[1] for item in eligible[: self.selected_k])
        if not selected:
            if contradiction_match:
                return self._abstain("reject_contradiction", scanned, event_cost)
            if stale_match:
                return self._abstain("reject_stale_checkpoint", scanned, event_cost)
            return self._abstain("abstain_no_supported_checkpoint", scanned, event_cost)
        evidence = tuple(
            {
                "checkpoint_id": item.checkpoint_id,
                "event_start": item.event_start,
                "event_end": item.event_end,
                "summary_ids": list(item.summary_ids),
                "source_refs": list(item.source_refs),
                "source_revision": item.source_revision,
                "state_group_id": item.state_group_id,
                "parent_digests": list(item.parent_digests),
                "restores_mutable_state": False,
            }
            for item in selected
        )
        return CheckpointRetrievalResult(
            False,
            "retrieve_evidence_references",
            evidence,
            tuple(item.checkpoint_id for item in selected),
            scanned,
            event_cost,
        )

    def expire(self, current_event: int) -> Tuple[str, ...]:
        expired = tuple(
            item.checkpoint_id
            for item in self.checkpoints
            if item.expires_at is not None and current_event >= item.expires_at
        )
        for checkpoint_id in expired:
            del self._checkpoints[checkpoint_id]
        self.expiry_count += len(expired)
        return expired

    def invalidate(
        self,
        *,
        source_revision: Optional[str] = None,
        state_group_id: Optional[str] = None,
    ) -> Tuple[str, ...]:
        if source_revision is None and state_group_id is None:
            raise ValueError("invalidation_selector_required")
        invalidated = tuple(
            item.checkpoint_id
            for item in self.checkpoints
            if (
                source_revision is not None
                and item.source_revision == str(source_revision)
            )
            or (
                state_group_id is not None
                and item.state_group_id == str(state_group_id)
            )
        )
        for checkpoint_id in invalidated:
            del self._checkpoints[checkpoint_id]
        self.invalidation_count += len(invalidated)
        return invalidated

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "enabled": self.enabled,
            "retention_profile": self.retention_profile,
            "limits": {
                "max_checkpoints": self.max_checkpoints,
                "selected_k": self.selected_k,
                "max_summary_ids": self.max_summary_ids,
                "max_state_bytes": self.max_state_bytes,
                "max_event_cost": self.max_event_cost,
                "max_merges_per_event": self.max_merges_per_event,
            },
            "counters": {
                "admission_count": self.admission_count,
                "eviction_count": self.eviction_count,
                "merge_count": self.merge_count,
                "expiry_count": self.expiry_count,
                "invalidation_count": self.invalidation_count,
            },
            "checkpoints": [item.to_dict() for item in self.checkpoints],
            "durable_mutation_allowed": False,
        }

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> "BoundedSparseMemoryCheckpointCache":
        if value.get("schema") != SCHEMA:
            raise ValueError("unsupported_memory_checkpoint_cache_schema")
        limits = value.get("limits")
        if not isinstance(limits, Mapping):
            raise ValueError("missing_memory_checkpoint_cache_limits")
        cache = cls(
            enabled=bool(value.get("enabled", False)),
            retention_profile=str(value.get("retention_profile", "")),
            max_checkpoints=int(limits.get("max_checkpoints", 0)),
            selected_k=int(limits.get("selected_k", 0)),
            max_summary_ids=int(limits.get("max_summary_ids", 0)),
            max_state_bytes=int(limits.get("max_state_bytes", 0)),
            max_event_cost=int(limits.get("max_event_cost", 0)),
            max_merges_per_event=int(limits.get("max_merges_per_event", 0)),
        )
        rows = value.get("checkpoints")
        if not isinstance(rows, list) or len(rows) > cache.max_checkpoints:
            raise ValueError("invalid_memory_checkpoint_count")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("invalid_memory_checkpoint_row")
            checkpoint = MemoryCheckpoint.from_dict(row)
            if not checkpoint.verified or checkpoint.contradicted:
                raise ValueError("unverified_or_contradicted_checkpoint_state")
            cache._checkpoints[checkpoint.checkpoint_id] = checkpoint
        counters = value.get("counters", {})
        if isinstance(counters, Mapping):
            for name in (
                "admission_count",
                "eviction_count",
                "merge_count",
                "expiry_count",
                "invalidation_count",
            ):
                setattr(cache, name, max(0, int(counters.get(name, 0))))
        if cache._serialized_size() > cache.max_state_bytes:
            raise ValueError("memory_checkpoint_state_byte_budget_exceeded")
        return cache

    def _admission_decision(
        self,
        candidate: MemoryCheckpointCandidate,
        *,
        current_event: Optional[int],
    ) -> str:
        if not self.enabled:
            return "checkpoint_cache_disabled"
        if not candidate.semantic_boundary:
            return "not_semantic_boundary"
        if not candidate.observed or not candidate.verified:
            return "unverified_checkpoint"
        if candidate.contradicted:
            return "contradicted_checkpoint"
        if candidate.event_start < 0 or candidate.event_end <= candidate.event_start:
            return "invalid_event_interval"
        if current_event is not None and candidate.event_end > current_event:
            return "future_event_interval"
        summary_ids = _bounded_strings(candidate.summary_ids, self.max_summary_ids + 1)
        if not summary_ids or len(summary_ids) > self.max_summary_ids:
            return "summary_width_exceeded"
        required = (
            candidate.source_refs,
            candidate.source_revision,
            candidate.state_group_id,
            candidate.parent_digest,
            candidate.runtime_fingerprint,
            candidate.schema_fingerprint,
        )
        if any(not value for value in required):
            return "missing_checkpoint_provenance"
        if candidate.expires_at is not None and candidate.expires_at <= candidate.event_end:
            return "invalid_checkpoint_expiry"
        return "admit"

    def _checkpoint_from_candidate(
        self, candidate: MemoryCheckpointCandidate
    ) -> MemoryCheckpoint:
        values = {
            "event_start": int(candidate.event_start),
            "event_end": int(candidate.event_end),
            "summary_ids": _bounded_strings(candidate.summary_ids, self.max_summary_ids),
            "source_refs": _bounded_strings(candidate.source_refs, self.max_summary_ids),
            "source_revision": str(candidate.source_revision),
            "state_group_id": str(candidate.state_group_id),
            "parent_digests": (str(candidate.parent_digest),),
            "runtime_fingerprint": str(candidate.runtime_fingerprint),
            "schema_fingerprint": str(candidate.schema_fingerprint),
            "expires_at": candidate.expires_at,
            "verified": True,
            "contradicted": False,
        }
        checkpoint_id = evidence_digest(
            {
                **values,
                "summary_ids": list(values["summary_ids"]),
                "source_refs": list(values["source_refs"]),
                "parent_digests": list(values["parent_digests"]),
            }
        )
        return MemoryCheckpoint(checkpoint_id=checkpoint_id, **values)

    def _merge_oldest_compatible(self) -> Tuple[str, ...]:
        ordered = self.checkpoints
        merge_count = 0
        for index, left in enumerate(ordered[:-1]):
            if merge_count >= self.max_merges_per_event:
                break
            right = ordered[index + 1]
            compatible = (
                left.source_revision == right.source_revision
                and left.state_group_id == right.state_group_id
                and left.runtime_fingerprint == right.runtime_fingerprint
                and left.schema_fingerprint == right.schema_fingerprint
            )
            summary_ids = _bounded_strings(
                left.summary_ids + right.summary_ids,
                self.max_summary_ids + 1,
            )
            if not compatible or len(summary_ids) > self.max_summary_ids:
                continue
            parent_digests = _bounded_strings(
                left.parent_digests + right.parent_digests,
                self.max_summary_ids,
            )
            source_refs = _bounded_strings(
                left.source_refs + right.source_refs,
                self.max_summary_ids,
            )
            payload = {
                "event_start": min(left.event_start, right.event_start),
                "event_end": max(left.event_end, right.event_end),
                "summary_ids": list(summary_ids),
                "source_refs": list(source_refs),
                "source_revision": left.source_revision,
                "state_group_id": left.state_group_id,
                "parent_digests": list(parent_digests),
                "runtime_fingerprint": left.runtime_fingerprint,
                "schema_fingerprint": left.schema_fingerprint,
                "expires_at": min(
                    value
                    for value in (left.expires_at, right.expires_at)
                    if value is not None
                ) if left.expires_at is not None or right.expires_at is not None else None,
                "verified": True,
                "contradicted": False,
            }
            merged = MemoryCheckpoint(
                checkpoint_id=evidence_digest(payload),
                event_start=payload["event_start"],
                event_end=payload["event_end"],
                summary_ids=summary_ids,
                source_refs=source_refs,
                source_revision=left.source_revision,
                state_group_id=left.state_group_id,
                parent_digests=parent_digests,
                runtime_fingerprint=left.runtime_fingerprint,
                schema_fingerprint=left.schema_fingerprint,
                expires_at=payload["expires_at"],
            )
            del self._checkpoints[left.checkpoint_id]
            del self._checkpoints[right.checkpoint_id]
            self._checkpoints[merged.checkpoint_id] = merged
            self.merge_count += 1
            return (left.checkpoint_id, right.checkpoint_id)
        return ()

    def _serialized_size(self) -> int:
        return len(_canonical_bytes([item.to_dict() for item in self.checkpoints]))

    @staticmethod
    def _chronological_key(item: MemoryCheckpoint) -> Tuple[int, int, str]:
        return item.event_start, item.event_end, item.checkpoint_id

    @staticmethod
    def _abstain(
        decision: str, scanned: int, event_cost: int
    ) -> CheckpointRetrievalResult:
        return CheckpointRetrievalResult(
            True,
            decision,
            (),
            (),
            scanned,
            event_cost,
        )


__all__ = [
    "BoundedSparseMemoryCheckpointCache",
    "CheckpointOperationResult",
    "CheckpointRetrievalResult",
    "MemoryCheckpoint",
    "MemoryCheckpointCandidate",
    "SCHEMA",
]
