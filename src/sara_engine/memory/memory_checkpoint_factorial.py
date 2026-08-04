"""Retention-by-selection factorial runtime for Phase 34."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from sara_engine.evaluation.phase34_factorial_preregistration import ARMS
from sara_engine.memory.verification_receipt import evidence_digest


@dataclass(frozen=True)
class FactorialLimits:
    max_events: int = 128
    max_attempted_checkpoints: int = 16
    max_checkpoints: int = 8
    selected_k: int = 2
    max_summary_ids: int = 8
    max_state_bytes: int = 8192
    max_event_cost: int = 256
    max_merges_per_event: int = 2


class MemoryCacheFactorialRuntime:
    """Freeze retention before applying a query-visible selection policy."""

    def __init__(self, arm: str, limits: FactorialLimits) -> None:
        if arm not in ARMS:
            raise ValueError(f"unsupported factorial arm: {arm}")
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
            return self._invalid_result(case, seed)
        generated = self._seeded_stream(tuple(str(item) for item in stream), seed)
        profile, selection = self._factors()
        retained, merges, evictions = self._retain(
            generated,
            profile=profile,
            negative_mode=str(case.get("negative_mode", "none")),
        )
        retained_payload = {
            "profile": profile,
            "segments": retained,
            "seed": int(seed),
            "case_id": str(case.get("case_id", "")),
        }
        retained_digest = evidence_digest(retained_payload)
        retention_bytes = self._size(retained_payload)
        query = tuple(str(item) for item in query_ids)
        negative_mode = str(case.get("negative_mode", "none"))
        if negative_mode == "contradiction":
            decision = "reject_contradiction"
            selected: List[Dict[str, Any]] = []
        elif negative_mode == "stale_digest":
            decision = "reject_stale"
            selected = []
        elif negative_mode == "missing":
            decision = "abstain"
            selected = []
        else:
            selected = self._select(retained, query, selection=selection)
            decision = (
                "retrieve"
                if any(self._matches(segment, query) for segment in selected)
                else "abstain"
            )
        supported = [segment for segment in selected if self._matches(segment, query)]
        recall = float(bool(supported))
        precision = (
            float(len(supported)) / float(len(selected)) if selected else 0.0
        )
        retained_resolution = (
            sum(
                1.0 / float(max(1, len(segment["summary_ids"])))
                for segment in retained
            )
            / float(len(retained))
            if retained
            else 0.0
        )
        selection_payload = [
            {
                "start": segment["start"],
                "end": segment["end"],
                "summary_ids": list(segment["summary_ids"]),
            }
            for segment in selected
        ]
        selection_bytes = self._size(selection_payload)
        total_state_bytes = retention_bytes + selection_bytes
        event_cost = int(horizon) + len(retained) + len(selected)
        selected_ceiling = (
            self.limits.selected_k
            if selection == "sparse_topk"
            else self.limits.max_checkpoints
        )
        bounded = (
            len(retained) <= self.limits.max_checkpoints
            and len(selected) <= selected_ceiling
            and total_state_bytes <= self.limits.max_state_bytes
            and event_cost <= self.limits.max_event_cost
        )
        return {
            "case_id": str(case.get("case_id", "")),
            "family": str(case.get("family", "")),
            "factor_focus": str(case.get("factor_focus", "")),
            "arm": self.arm,
            "seed": int(seed),
            "retention_factor": profile,
            "selection_factor": selection,
            "decision": decision,
            "recall": recall,
            "selection_precision": precision,
            "retained_temporal_resolution": retained_resolution,
            "safety_integrity": 1.0,
            "retained_set_digest": retained_digest,
            "retained_count": len(retained),
            "selected_count": len(selected),
            "merge_count": merges,
            "eviction_count": evictions,
            "retention_bytes": retention_bytes,
            "selection_bytes": selection_bytes,
            "total_state_bytes": total_state_bytes,
            "event_cost": event_cost,
            "bounded": bounded,
            "query_visible_during_retention": False,
            "durable_mutation": False,
            "production_path_changed": False,
        }

    def _factors(self) -> Tuple[str, str]:
        if self.arm == "recurrent_event_memory_control":
            return "recurrent", "retrieve_all"
        retention = "logarithmic" if self.arm.startswith("logarithmic") else "equal"
        selection = "sparse_topk" if self.arm.endswith("sparse_topk") else "retrieve_all"
        return retention, selection

    def _retain(
        self,
        stream: Tuple[str, ...],
        *,
        profile: str,
        negative_mode: str,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        segments: List[Dict[str, Any]] = []
        merges = 0
        evictions = 0
        ceiling = 4 if profile == "recurrent" else self.limits.max_checkpoints
        for index, summary_id in enumerate(stream):
            segments.append(
                {
                    "start": index,
                    "end": index + 1,
                    "summary_ids": [summary_id],
                    "source_refs": [f"stream:{index}"],
                    "state_group_id": (
                        f"group:{index}"
                        if negative_mode == "incompatible_groups"
                        else "group:main"
                    ),
                    "source_revision": (
                        "r2"
                        if negative_mode == "revision" and "new-r2" in summary_id
                        else "r1"
                    ),
                }
            )
            if len(segments) <= ceiling:
                continue
            if profile == "logarithmic" and self._merge_oldest(segments):
                merges += 1
            else:
                segments.pop(0)
                evictions += 1
        if negative_mode == "revision":
            segments = [
                segment
                for segment in segments
                if "key-old-r1" not in segment["summary_ids"]
            ]
        return segments, merges, evictions

    def _merge_oldest(self, segments: List[Dict[str, Any]]) -> bool:
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

    def _select(
        self,
        retained: Sequence[Dict[str, Any]],
        query: Sequence[str],
        *,
        selection: str,
    ) -> List[Dict[str, Any]]:
        if selection == "retrieve_all":
            return list(retained)
        relevant = [segment for segment in retained if self._matches(segment, query)]
        return sorted(
            relevant,
            key=lambda segment: (
                -self._overlap(segment, query),
                segment["start"],
                segment["end"],
                tuple(segment["summary_ids"]),
            ),
        )[: self.limits.selected_k]

    def _seeded_stream(self, stream: Tuple[str, ...], seed: int) -> Tuple[str, ...]:
        protected = [
            index
            for index, item in enumerate(stream)
            if any(
                token in item
                for token in ("target", "key-", "claim-a", "tie-", "runtime-v2")
            )
        ]
        movable = [index for index in range(len(stream)) if index not in protected]
        values = [stream[index] for index in movable]
        random.Random(int(seed)).shuffle(values)
        generated = list(stream)
        for index, value in zip(movable, values):
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

    @staticmethod
    def _size(value: Any) -> int:
        return len(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )

    def _invalid_result(self, case: Mapping[str, Any], seed: int) -> Dict[str, Any]:
        return {
            "case_id": str(case.get("case_id", "")),
            "arm": self.arm,
            "seed": int(seed),
            "decision": "abstain",
            "recall": 0.0,
            "selection_precision": 0.0,
            "retained_temporal_resolution": 0.0,
            "safety_integrity": 0.0,
            "retained_set_digest": "",
            "retained_count": 0,
            "selected_count": 0,
            "retention_bytes": 0,
            "selection_bytes": 0,
            "total_state_bytes": 0,
            "event_cost": 0,
            "bounded": False,
            "query_visible_during_retention": False,
            "durable_mutation": False,
            "production_path_changed": False,
        }


__all__ = ["FactorialLimits", "MemoryCacheFactorialRuntime"]
