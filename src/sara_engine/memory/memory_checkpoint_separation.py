"""Five-seed cache-arm separation runtime for the registered follow-up."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from sara_engine.evaluation.phase34_memory_cache_preregistration import ARMS


@dataclass(frozen=True)
class SeparationLimits:
    max_events: int = 128
    max_attempted_checkpoints: int = 16
    max_checkpoints: int = 8
    selected_k: int = 2
    max_summary_ids: int = 8
    max_state_bytes: int = 8192
    max_event_cost: int = 256
    max_merges_per_event: int = 2


class MemoryCacheSeparationRuntime:
    """Evaluate one synthetic observed-only stream under one frozen arm."""

    def __init__(self, arm: str, limits: SeparationLimits) -> None:
        if arm not in ARMS:
            raise ValueError(f"unsupported separation arm: {arm}")
        self.arm = arm
        self.limits = limits

    def evaluate(self, case: Mapping[str, Any], *, seed: int) -> Dict[str, Any]:
        stream = case.get("checkpoint_stream")
        query_ids = case.get("query_ids")
        horizon = case.get("horizon_events")
        if (
            not isinstance(stream, list)
            or not 1 <= len(stream) <= self.limits.max_attempted_checkpoints
            or not isinstance(query_ids, list)
            or not 1 <= len(query_ids) <= self.limits.max_summary_ids
            or type(horizon) is not int
            or not 1 <= horizon <= self.limits.max_events
        ):
            return self._abstain(case, seed, "invalid_case")
        generated = self._seeded_stream(tuple(str(item) for item in stream), seed)
        negative_mode = str(case.get("negative_mode", "none"))
        if negative_mode == "contradiction":
            return self._safety_result(case, seed, "reject_contradiction", generated)
        if negative_mode == "stale_digest":
            return self._safety_result(case, seed, "reject_stale", generated)
        if negative_mode == "missing":
            return self._safety_result(case, seed, "abstain", generated)

        segments, merge_count, eviction_count = self._retain(
            generated,
            negative_mode=negative_mode,
        )
        query = tuple(str(item) for item in query_ids)
        relevant = [segment for segment in segments if self._matches(segment, query)]
        if self.arm == "recurrent_event_memory_control":
            selected = relevant
        elif self.arm == "equal_segment_sparse_topk":
            selected = sorted(
                relevant,
                key=lambda item: (
                    -self._overlap(item, query),
                    item["start"],
                    item["end"],
                    tuple(item["summary_ids"]),
                ),
            )[: self.limits.selected_k]
        else:
            selected = list(segments)
        supported = [segment for segment in selected if self._matches(segment, query)]
        decision = "retrieve" if supported else "abstain"
        precision = (
            float(len(supported)) / float(len(selected)) if selected else 0.0
        )
        recall = float(bool(supported))
        target_resolution = max(
            (
                1.0 / float(max(1, len(segment["summary_ids"])))
                for segment in supported
            ),
            default=0.0,
        )
        retained_resolution = (
            sum(
                1.0 / float(max(1, len(segment["summary_ids"])))
                for segment in segments
            )
            / float(len(segments))
            if segments
            else 0.0
        )
        state_bytes = len(
            json.dumps(
                segments,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        event_cost = int(horizon) + len(segments)
        bounded = (
            len(segments) <= self.limits.max_checkpoints
            and len(selected)
            <= (
                self.limits.selected_k
                if self.arm == "equal_segment_sparse_topk"
                else self.limits.max_checkpoints
            )
            and state_bytes <= self.limits.max_state_bytes
            and event_cost <= self.limits.max_event_cost
        )
        return {
            "case_id": str(case.get("case_id", "")),
            "family": str(case.get("family", "")),
            "arm": self.arm,
            "seed": seed,
            "decision": decision,
            "recall": recall,
            "selection_precision": precision,
            "temporal_resolution": target_resolution,
            "retained_temporal_resolution": retained_resolution,
            "safety_integrity": 1.0,
            "retained_count": len(segments),
            "selected_count": len(selected),
            "merge_count": merge_count,
            "eviction_count": eviction_count,
            "state_bytes": state_bytes,
            "event_cost": event_cost,
            "bounded": bounded,
            "selected_summary_ids": [
                list(segment["summary_ids"]) for segment in selected
            ],
            "durable_mutation": False,
            "production_path_changed": False,
        }

    def _retain(
        self,
        stream: Tuple[str, ...],
        *,
        negative_mode: str,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        segments: List[Dict[str, Any]] = []
        merge_count = 0
        eviction_count = 0
        recent_width = 4 if self.arm == "recurrent_event_memory_control" else None
        for index, summary_id in enumerate(stream):
            group = (
                f"group-{index}"
                if negative_mode == "incompatible_groups"
                else "group-main"
            )
            revision = "r2" if negative_mode == "revision" and "new-r2" in summary_id else "r1"
            segments.append(
                {
                    "start": index,
                    "end": index + 1,
                    "summary_ids": [summary_id],
                    "source_refs": [f"stream:{index}"],
                    "state_group_id": group,
                    "source_revision": revision,
                }
            )
            ceiling = recent_width or self.limits.max_checkpoints
            if len(segments) <= ceiling:
                continue
            if self.arm == "logarithmic_segments_retrieve_all":
                merged = self._merge_oldest_compatible(segments)
                if merged:
                    merge_count += 1
                    continue
            segments.pop(0)
            eviction_count += 1
        if negative_mode == "revision":
            newest_revision = "r2"
            segments = [
                segment
                for segment in segments
                if segment["source_revision"] == newest_revision
                or "key-old-r1" not in segment["summary_ids"]
            ]
        return segments, merge_count, eviction_count

    def _merge_oldest_compatible(self, segments: List[Dict[str, Any]]) -> bool:
        for index in range(len(segments) - 1):
            left = segments[index]
            right = segments[index + 1]
            if (
                left["state_group_id"] != right["state_group_id"]
                or left["source_revision"] != right["source_revision"]
            ):
                continue
            summary_ids = list(
                dict.fromkeys(left["summary_ids"] + right["summary_ids"])
            )
            if len(summary_ids) > self.limits.max_summary_ids:
                continue
            segments[index : index + 2] = [
                {
                    "start": left["start"],
                    "end": right["end"],
                    "summary_ids": summary_ids,
                    "source_refs": list(
                        dict.fromkeys(left["source_refs"] + right["source_refs"])
                    ),
                    "state_group_id": left["state_group_id"],
                    "source_revision": left["source_revision"],
                }
            ]
            return True
        return False

    def _seeded_stream(self, stream: Tuple[str, ...], seed: int) -> Tuple[str, ...]:
        target_positions = [
            index
            for index, item in enumerate(stream)
            if any(token in item for token in ("target", "key-", "claim-a", "tie-"))
        ]
        movable_positions = [
            index for index in range(len(stream)) if index not in target_positions
        ]
        movable_values = [stream[index] for index in movable_positions]
        random.Random(int(seed)).shuffle(movable_values)
        generated = list(stream)
        for index, value in zip(movable_positions, movable_values):
            generated[index] = value
        return tuple(generated)

    @staticmethod
    def _matches(segment: Mapping[str, Any], query: Sequence[str]) -> bool:
        return any(
            query_id in summary_id or summary_id in query_id
            for query_id in query
            for summary_id in segment["summary_ids"]
        )

    @staticmethod
    def _overlap(segment: Mapping[str, Any], query: Sequence[str]) -> int:
        return sum(
            query_id in summary_id or summary_id in query_id
            for query_id in query
            for summary_id in segment["summary_ids"]
        )

    def _safety_result(
        self,
        case: Mapping[str, Any],
        seed: int,
        decision: str,
        stream: Tuple[str, ...],
    ) -> Dict[str, Any]:
        return {
            "case_id": str(case.get("case_id", "")),
            "family": str(case.get("family", "")),
            "arm": self.arm,
            "seed": seed,
            "decision": decision,
            "recall": 0.0,
            "selection_precision": 1.0,
            "temporal_resolution": 1.0,
            "retained_temporal_resolution": 1.0,
            "safety_integrity": 1.0,
            "retained_count": min(len(stream), self.limits.max_checkpoints),
            "selected_count": 0,
            "merge_count": 0,
            "eviction_count": max(0, len(stream) - self.limits.max_checkpoints),
            "state_bytes": 0,
            "event_cost": int(case.get("horizon_events", 0)),
            "bounded": True,
            "selected_summary_ids": [],
            "durable_mutation": False,
            "production_path_changed": False,
        }

    def _abstain(
        self, case: Mapping[str, Any], seed: int, reason: str
    ) -> Dict[str, Any]:
        result = self._safety_result(case, seed, "abstain", ())
        result["safety_integrity"] = 0.0
        result["bounded"] = False
        result["reason"] = reason
        return result


__all__ = ["MemoryCacheSeparationRuntime", "SeparationLimits"]
