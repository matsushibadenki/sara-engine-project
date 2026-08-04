"""Observed-only four-arm runtime for the registered Phase 34 ablation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from sara_engine.memory.memory_checkpoint_cache import (
    BoundedSparseMemoryCheckpointCache,
    MemoryCheckpointCandidate,
)


ARMS = (
    "recurrent_event_memory_control",
    "equal_segment_retrieve_all",
    "logarithmic_segments_retrieve_all",
    "equal_segment_sparse_topk",
)


@dataclass(frozen=True)
class Phase34MemoryCacheLimits:
    max_events: int = 128
    max_checkpoints: int = 8
    selected_k: int = 2
    max_summary_ids: int = 8
    max_state_bytes: int = 8192
    max_event_cost: int = 256
    max_merges_per_event: int = 2
    equal_segment_span: int = 4


class Phase34MemoryCheckpointRuntime:
    """Evaluate one frozen case without touching production or durable memory."""

    def __init__(self, arm: str, limits: Phase34MemoryCacheLimits) -> None:
        if arm not in ARMS:
            raise ValueError(f"unsupported Phase 34 arm: {arm}")
        self.arm = arm
        self.limits = limits

    def evaluate(self, case: Mapping[str, Any]) -> Dict[str, Any]:
        events = case.get("events")
        checkpoints = case.get("checkpoints")
        query = case.get("query")
        if not isinstance(events, list) or not 1 <= len(events) <= self.limits.max_events:
            return self._abstain(case, "event_budget_exceeded")
        if not isinstance(checkpoints, list) or not isinstance(query, Mapping):
            return self._abstain(case, "malformed_case")
        query_ids = tuple(str(item) for item in query.get("summary_ids", ()))
        if not query_ids or len(query_ids) > self.limits.max_summary_ids:
            return self._abstain(case, "invalid_query")
        if self.arm == "recurrent_event_memory_control":
            return self._evaluate_control(case, events, query_ids)
        return self._evaluate_cache(case, events, checkpoints, query_ids)

    def _evaluate_control(
        self,
        case: Mapping[str, Any],
        events: Sequence[Any],
        query_ids: Tuple[str, ...],
    ) -> Dict[str, Any]:
        family = str(case.get("family", ""))
        attempted = int(case.get("attempted_checkpoint_count", 0))
        if family == "contradiction":
            decision = "reject_contradiction"
        elif family in {"stale_runtime_digest", "stale_schema_digest"}:
            decision = "reject_stale"
        elif family in {"missing_segment", "reordered_replay"}:
            decision = "abstain"
        elif family == "cache_overflow" and attempted > self.limits.max_checkpoints:
            decision = "evict"
        elif family in {"duplicate_segment", "near_duplicate_segment"}:
            decision = "merge"
        else:
            recent = tuple(
                str(item)
                for item in events[:-1][-self.limits.equal_segment_span :]
            )
            decision = (
                "retrieve"
                if any(
                    query_id in event
                    for query_id in query_ids
                    for event in recent
                )
                else "abstain"
            )
        return self._result(
            case,
            decision=decision,
            selected_ids=(),
            checkpoint_count=0,
            state_bytes=0,
            event_cost=min(len(events), self.limits.max_event_cost),
            eviction_count=int(decision == "evict"),
            merge_count=int(decision == "merge"),
        )

    def _evaluate_cache(
        self,
        case: Mapping[str, Any],
        events: Sequence[Any],
        rows: Sequence[Mapping[str, Any]],
        query_ids: Tuple[str, ...],
    ) -> Dict[str, Any]:
        profile = (
            "logarithmic"
            if self.arm == "logarithmic_segments_retrieve_all"
            else "equal"
        )
        cache = BoundedSparseMemoryCheckpointCache(
            enabled=True,
            retention_profile=profile,
            max_checkpoints=self.limits.max_checkpoints,
            selected_k=self.limits.selected_k,
            max_summary_ids=self.limits.max_summary_ids,
            max_state_bytes=self.limits.max_state_bytes,
            max_event_cost=self.limits.max_event_cost,
            max_merges_per_event=self.limits.max_merges_per_event,
        )
        family = str(case.get("family", ""))
        admissions = []
        attempted = max(len(rows), int(case.get("attempted_checkpoint_count", 0)))
        for index in range(attempted):
            source = rows[index % len(rows)]
            summary_ids = tuple(str(item) for item in source.get("summary_ids", ()))
            if index >= len(rows):
                summary_ids = (f"overflow-{index}",)
            runtime = str(case.get("runtime_fingerprint", ""))
            schema = str(case.get("schema_fingerprint", ""))
            if family == "stale_runtime_digest":
                runtime = "runtime-v1"
            if family == "stale_schema_digest":
                schema = "schema-v1"
            candidate = MemoryCheckpointCandidate(
                event_start=int(source.get("event_start", index)),
                event_end=max(
                    int(source.get("event_end", index + 1)),
                    int(source.get("event_start", index)) + 1,
                ),
                summary_ids=summary_ids,
                source_refs=(f"fixture:{case.get('case_id', '')}:{index}",),
                source_revision=str(case.get("source_revision", "")),
                state_group_id=(
                    f"phase34:{case.get('case_id', '')}:{index}"
                    if family == "cache_overflow"
                    else f"phase34:{case.get('case_id', '')}"
                ),
                parent_digest=f"fixture-parent:{source.get('checkpoint_id', index)}",
                runtime_fingerprint=runtime,
                schema_fingerprint=schema,
                contradicted=bool(source.get("contradicted", False)),
            )
            admissions.append(cache.admit(candidate, current_event=len(events)))
            if family in {"duplicate_segment", "near_duplicate_segment"}:
                admissions.append(cache.admit(candidate, current_event=len(events)))

        if family == "contradiction" and any(
            item.decision == "contradicted_checkpoint" for item in admissions
        ):
            decision = "reject_contradiction"
            selected_ids: Tuple[str, ...] = ()
            event_cost = len(events)
        elif family == "reordered_replay":
            decision = "abstain"
            selected_ids = ()
            event_cost = len(events)
        elif family == "cache_overflow" and cache.eviction_count > 0:
            decision = "evict"
            selected_ids = ()
            event_cost = len(events)
        elif family in {"duplicate_segment", "near_duplicate_segment"} and any(
            item.decision == "duplicate_checkpoint_preserved"
            for item in admissions
        ):
            decision = "merge"
            selected_ids = ()
            event_cost = len(events)
        else:
            retrieval = cache.retrieve(
                query_ids,
                source_revision=str(case.get("source_revision", "")),
                runtime_fingerprint=str(case.get("runtime_fingerprint", "")),
                schema_fingerprint=str(case.get("schema_fingerprint", "")),
                current_event=len(events),
            )
            decision = self._normalized_retrieval_decision(retrieval.decision)
            selected_ids = retrieval.selected_checkpoint_ids
            event_cost = retrieval.event_cost + len(events)

        state_bytes = len(
            json.dumps(
                [item.to_dict() for item in cache.checkpoints],
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        return self._result(
            case,
            decision=decision,
            selected_ids=selected_ids,
            checkpoint_count=len(cache.checkpoints),
            state_bytes=state_bytes,
            event_cost=event_cost,
            eviction_count=cache.eviction_count,
            merge_count=cache.merge_count
            + sum(
                item.decision == "duplicate_checkpoint_preserved"
                for item in admissions
            ),
        )

    def _result(
        self,
        case: Mapping[str, Any],
        *,
        decision: str,
        selected_ids: Tuple[str, ...],
        checkpoint_count: int,
        state_bytes: int,
        event_cost: int,
        eviction_count: int,
        merge_count: int,
    ) -> Dict[str, Any]:
        expected = str(case.get("expected", {}).get("decision", ""))
        target_match = decision == expected
        return {
            "case_id": str(case.get("case_id", "")),
            "family": str(case.get("family", "")),
            "arm": self.arm,
            "status": "evaluated" if decision not in {"abstain"} else "abstained",
            "decision": decision,
            "expected_decision": expected,
            "target_match": target_match,
            "selected_checkpoint_ids": list(selected_ids),
            "selected_count": len(selected_ids),
            "checkpoint_count": checkpoint_count,
            "state_bytes": state_bytes,
            "event_cost": event_cost,
            "eviction_count": eviction_count,
            "merge_count": merge_count,
            "durable_mutation": False,
            "production_path_changed": False,
        }

    def _abstain(self, case: Mapping[str, Any], reason: str) -> Dict[str, Any]:
        result = self._result(
            case,
            decision="abstain",
            selected_ids=(),
            checkpoint_count=0,
            state_bytes=0,
            event_cost=0,
            eviction_count=0,
            merge_count=0,
        )
        result["reason"] = reason
        return result

    @staticmethod
    def _normalized_retrieval_decision(decision: str) -> str:
        if decision == "retrieve_evidence_references":
            return "retrieve"
        if decision == "reject_stale_checkpoint":
            return "reject_stale"
        if decision == "reject_contradiction":
            return "reject_contradiction"
        return "abstain"


__all__ = ["ARMS", "Phase34MemoryCacheLimits", "Phase34MemoryCheckpointRuntime"]
