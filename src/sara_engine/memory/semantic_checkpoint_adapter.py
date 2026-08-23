"""Bounded sparse semantic checkpoint adapter for Phase 34."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from sara_engine.evaluation.phase34_factorial_preregistration import ARMS


@dataclass(frozen=True)
class SemanticCheckpointLimits:
    """Frozen resource limits shared by all semantic checkpoint arms."""

    max_events: int = 128
    max_attempted_checkpoints: int = 16
    max_checkpoints: int = 8
    selected_k: int = 2
    max_claims_per_checkpoint: int = 8
    max_state_bytes: int = 8192
    max_event_cost: int = 256


@dataclass(frozen=True)
class SparseSemanticClaim:
    """One evidence-linked claim represented by sparse typed features."""

    claim_key: str
    subjects: Tuple[str, ...]
    axes: Tuple[str, ...]
    source_ref: str
    source_revision: str
    revision_rank: int = 0
    polarity: int = 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "claim_key": self.claim_key,
            "subjects": list(self.subjects),
            "axes": list(self.axes),
            "source_ref": self.source_ref,
            "source_revision": self.source_revision,
            "revision_rank": self.revision_rank,
            "polarity": self.polarity,
        }


@dataclass(frozen=True)
class SparseSemanticQuery:
    """A multilingual query reduced to typed subjects and requested axes."""

    subjects: Tuple[str, ...]
    axes: Tuple[str, ...]


class SparseMultilingualSemanticAdapter:
    """Map bounded English, Japanese, and Simplified-Chinese text to typed axes."""

    _SUBJECT_ALIASES: Mapping[str, Tuple[str, ...]] = {
        "http": ("http", "hypertext transfer protocol", "超文本传输协议"),
        "bcp14": ("bcp 14", "bcp14", "must", "should"),
        "argparse": ("argparse",),
        "parse_args": ("parse_args", "add_argument"),
        "pathlib": ("pathlib", "pure path", "pure paths", "纯路径"),
    }
    _AXIS_ALIASES: Mapping[str, Tuple[str, ...]] = {
        "protocol_layer": (
            "application-level",
            "protocol layer",
            "プロトコル層",
            "协议层",
        ),
        "protocol_state": (
            "stateless",
            "stateful",
            "状態を保持",
            "有状態",
            "有状态",
            "无状态",
        ),
        "message_role": (
            "request or a response",
            "request",
            "response",
            "message role",
            "メッセージ役割",
            "消息角色",
        ),
        "keyword_case": (
            "all capitals",
            "bcp 14 meaning",
            "bcp 14の意味",
            "bcp 14规定的含义",
        ),
        "argument_parse": (
            "parse those out",
            "parses declared",
            "解析",
        ),
        "argv_source": ("sys.argv",),
        "help_usage": (
            "help and usage",
            "help and usage messages",
            "ヘルプと使用法",
            "帮助和用法",
        ),
        "output_container": (
            "argparse.namespace",
            "extracted data",
            "extracted values",
            "抽出した値",
            "提取出的值",
        ),
        "path_io_category": (
            "without i/o",
            "provide i/o operations",
            "adds i/o operations",
            "i/oを行わず",
            "i/o操作を追加",
            "不执行i/o",
            "增加了i/o",
        ),
        "transport_encryption": (
            "transport-encryption",
            "transport encryption",
            "トランスポート暗号化",
            "传输加密",
        ),
        "database_schema": (
            "database schema",
            "データベーススキーマ",
            "数据库模式",
        ),
        "numeric_status": (
            "numeric http status",
            "数値ステータスコード",
            "数字http状态码",
        ),
        "worker_threads": (
            "worker threads",
            "ワーカースレッド",
            "工作线程",
        ),
        "database_storage": (
            "database table",
            "データベース表",
            "数据库表",
        ),
        "path_sync_protocol": (
            "network protocol",
            "ネットワークプロトコル",
            "网络协议",
        ),
    }

    def encode_source(
        self,
        text: str,
        *,
        source_ref: str,
        source_revision: str,
        revision_rank: int = 0,
        polarity: int = 1,
    ) -> SparseSemanticClaim:
        subjects, axes = self._features(text)
        payload = {
            "subjects": subjects,
            "axes": axes,
            "source_ref": source_ref,
            "source_revision": source_revision,
            "revision_rank": int(revision_rank),
            "polarity": int(polarity),
        }
        return SparseSemanticClaim(
            claim_key=self._digest(payload),
            subjects=subjects,
            axes=axes,
            source_ref=str(source_ref),
            source_revision=str(source_revision),
            revision_rank=int(revision_rank),
            polarity=1 if int(polarity) >= 0 else -1,
        )

    def encode_query(self, text: str) -> SparseSemanticQuery:
        subjects, axes = self._features(text)
        return SparseSemanticQuery(subjects=subjects, axes=axes)

    @classmethod
    def _features(cls, text: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        normalized = cls._normalize(text)
        subjects = tuple(
            sorted(
                name
                for name, aliases in cls._SUBJECT_ALIASES.items()
                if any(cls._normalize(alias) in normalized for alias in aliases)
            )
        )
        axes = tuple(
            sorted(
                name
                for name, aliases in cls._AXIS_ALIASES.items()
                if any(cls._normalize(alias) in normalized for alias in aliases)
            )
        )
        return subjects, axes

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(str(text).casefold().replace("/", " / ").split())

    @staticmethod
    def _digest(value: Any) -> str:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class SemanticCheckpointRuntime:
    """Retain query-blind claim checkpoints, then perform bounded semantic selection."""

    def __init__(self, arm: str, limits: SemanticCheckpointLimits) -> None:
        if arm not in ARMS:
            raise ValueError(f"unsupported semantic checkpoint arm: {arm}")
        self.arm = str(arm)
        self.limits = limits

    def evaluate(
        self,
        events: Sequence[SparseSemanticClaim],
        query: SparseSemanticQuery,
        *,
        horizon: int,
        omission_receipt: bool = False,
    ) -> Dict[str, Any]:
        if (
            not 1 <= len(events) <= self.limits.max_attempted_checkpoints
            or not 1 <= int(horizon) <= self.limits.max_events
        ):
            return self._invalid_result()
        profile, selection_policy = self._factors()
        retained, merges, evictions = self._retain(events, profile=profile)
        retained_payload = {
            "profile": profile,
            "checkpoints": retained,
        }
        retained_digest = self._digest(retained_payload)
        selected = self._select(retained, query, policy=selection_policy)
        retained_bytes = self._size(retained_payload)
        selection_payload = [
            {
                "start": checkpoint["start"],
                "end": checkpoint["end"],
                "claim_keys": [
                    claim["claim_key"] for claim in checkpoint["claims"]
                ],
            }
            for checkpoint in selected
        ]
        selected_bytes = self._size(selection_payload)
        state_bytes = retained_bytes + selected_bytes
        event_cost = int(horizon) + len(retained) + len(selected)
        decision = self._decide(selected, query, omission_receipt=omission_receipt)
        selected_ceiling = (
            self.limits.selected_k
            if selection_policy == "sparse_topk"
            else self.limits.max_checkpoints
        )
        bounded = (
            len(retained) <= self.limits.max_checkpoints
            and len(selected) <= selected_ceiling
            and state_bytes <= self.limits.max_state_bytes
            and event_cost <= self.limits.max_event_cost
        )
        return {
            "arm": self.arm,
            "retention_factor": profile,
            "selection_factor": selection_policy,
            "decision": decision["decision"],
            "claim_key": decision.get("claim_key"),
            "source_ref": decision.get("source_ref"),
            "source_revision": decision.get("source_revision"),
            "matched_subjects": list(decision.get("matched_subjects", ())),
            "matched_axes": list(decision.get("matched_axes", ())),
            "retained_set_digest": retained_digest,
            "retained_count": len(retained),
            "selected_count": len(selected),
            "merge_count": merges,
            "eviction_count": evictions,
            "retention_bytes": retained_bytes,
            "selection_bytes": selected_bytes,
            "total_state_bytes": state_bytes,
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
        events: Sequence[SparseSemanticClaim],
        *,
        profile: str,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        checkpoints: List[Dict[str, Any]] = []
        merges = 0
        evictions = 0
        ceiling = 4 if profile == "recurrent" else self.limits.max_checkpoints
        for index, claim in enumerate(events):
            checkpoints.append(
                {"start": index, "end": index + 1, "claims": [claim.as_dict()]}
            )
            if len(checkpoints) <= ceiling:
                continue
            if profile == "logarithmic" and self._merge_oldest(checkpoints):
                merges += 1
            else:
                checkpoints.pop(0)
                evictions += 1
        return checkpoints, merges, evictions

    def _merge_oldest(self, checkpoints: List[Dict[str, Any]]) -> bool:
        for index in range(len(checkpoints) - 1):
            left = checkpoints[index]
            right = checkpoints[index + 1]
            claims = list(left["claims"])
            known = {str(claim["claim_key"]) for claim in claims}
            claims.extend(
                claim
                for claim in right["claims"]
                if str(claim["claim_key"]) not in known
            )
            if len(claims) > self.limits.max_claims_per_checkpoint:
                continue
            checkpoints[index : index + 2] = [
                {
                    "start": left["start"],
                    "end": right["end"],
                    "claims": claims,
                }
            ]
            return True
        return False

    def _select(
        self,
        retained: Sequence[Mapping[str, Any]],
        query: SparseSemanticQuery,
        *,
        policy: str,
    ) -> List[Dict[str, Any]]:
        if policy == "retrieve_all":
            return [dict(checkpoint) for checkpoint in retained]
        ranked = sorted(
            (
                (self._checkpoint_score(checkpoint, query), checkpoint)
                for checkpoint in retained
            ),
            key=lambda item: (
                -item[0],
                -int(item[1]["start"]),
                -int(item[1]["end"]),
            ),
        )
        return [
            dict(checkpoint)
            for score, checkpoint in ranked[: self.limits.selected_k]
            if score > 0
        ]

    def _decide(
        self,
        selected: Sequence[Mapping[str, Any]],
        query: SparseSemanticQuery,
        *,
        omission_receipt: bool,
    ) -> Dict[str, Any]:
        if omission_receipt:
            return {"decision": "abstain_missing"}
        candidates = [
            claim
            for checkpoint in selected
            for claim in checkpoint["claims"]
            if self._claim_matches(claim, query)
        ]
        if not candidates:
            return {"decision": "abstain_unsupported"}
        best_rank = max(int(claim["revision_rank"]) for claim in candidates)
        latest = [
            claim for claim in candidates if int(claim["revision_rank"]) == best_rank
        ]
        polarities = {int(claim["polarity"]) for claim in latest}
        if len(polarities) > 1:
            return {"decision": "abstain_contradiction"}
        unique = {str(claim["claim_key"]): claim for claim in latest}
        if len(unique) != 1:
            return {"decision": "abstain_ambiguous"}
        claim = next(iter(unique.values()))
        return {
            "decision": "retrieve_revision" if best_rank > 0 else "retrieve_original",
            "claim_key": claim["claim_key"],
            "source_ref": claim["source_ref"],
            "source_revision": claim["source_revision"],
            "matched_subjects": tuple(query.subjects),
            "matched_axes": tuple(query.axes),
        }

    @staticmethod
    def _claim_matches(
        claim: Mapping[str, Any], query: SparseSemanticQuery
    ) -> bool:
        subjects = set(str(item) for item in claim["subjects"])
        axes = set(str(item) for item in claim["axes"])
        return (
            bool(query.axes)
            and (not query.subjects or bool(subjects.intersection(query.subjects)))
            and set(query.axes).issubset(axes)
        )

    def _checkpoint_score(
        self, checkpoint: Mapping[str, Any], query: SparseSemanticQuery
    ) -> int:
        best = 0
        for claim in checkpoint["claims"]:
            subject_score = int(
                bool(set(claim["subjects"]).intersection(query.subjects))
            )
            axis_score = len(set(claim["axes"]).intersection(query.axes))
            best = max(best, subject_score + axis_score)
        return best

    @staticmethod
    def _digest(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

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

    def _invalid_result(self) -> Dict[str, Any]:
        return {
            "arm": self.arm,
            "decision": "abstain_invalid",
            "claim_key": None,
            "source_ref": None,
            "source_revision": None,
            "matched_subjects": [],
            "matched_axes": [],
            "retained_set_digest": "",
            "retained_count": 0,
            "selected_count": 0,
            "merge_count": 0,
            "eviction_count": 0,
            "retention_bytes": 0,
            "selection_bytes": 0,
            "total_state_bytes": 0,
            "event_cost": 0,
            "bounded": False,
            "query_visible_during_retention": False,
            "durable_mutation": False,
            "production_path_changed": False,
        }


def noise_claim(index: int) -> SparseSemanticClaim:
    """Create a deterministic non-semantic checkpoint filler."""

    value = int(index)
    key = hashlib.sha256(f"phase34-noise:{value}".encode("utf-8")).hexdigest()
    return SparseSemanticClaim(
        claim_key=key,
        subjects=(f"noise_subject_{value}",),
        axes=(f"noise_axis_{value}",),
        source_ref=f"synthetic:phase34-noise:{value}",
        source_revision="synthetic-v1",
    )


def claim_stream(
    claims: Sequence[SparseSemanticClaim],
    *,
    target_source_ref: str,
    horizon: int,
    control_mode: str,
) -> Tuple[Tuple[SparseSemanticClaim, ...], bool]:
    """Build a frozen query-blind stream and return its omission receipt."""

    target = next(
        (claim for claim in claims if claim.source_ref == target_source_ref), None
    )
    if target is None:
        raise ValueError("target source is absent from reviewed semantic claims")
    others = [claim for claim in claims if claim.source_ref != target_source_ref]
    event_count = min(16, max(1, int(horizon)))
    omission = control_mode == "target_evidence_omitted"
    stream: List[SparseSemanticClaim] = [] if omission else [target]
    stream.extend(others)
    while len(stream) < event_count:
        stream.append(noise_claim(len(stream)))
    stream = stream[:event_count]
    if control_mode == "verified_later_revision":
        revised = SparseSemanticClaim(
            claim_key=hashlib.sha256(
                f"revision:{target.claim_key}".encode("utf-8")
            ).hexdigest(),
            subjects=target.subjects,
            axes=target.axes,
            source_ref=target.source_ref,
            source_revision=f"{target.source_revision}:verified-later",
            revision_rank=1,
            polarity=target.polarity,
        )
        stream[-1] = revised
    elif control_mode == "unresolved_equal_source_contradiction":
        positive = target
        negative = SparseSemanticClaim(
            claim_key=hashlib.sha256(
                f"contradiction:{target.claim_key}".encode("utf-8")
            ).hexdigest(),
            subjects=target.subjects,
            axes=target.axes,
            source_ref=target.source_ref,
            source_revision=target.source_revision,
            revision_rank=target.revision_rank,
            polarity=-target.polarity,
        )
        if len(stream) == 1:
            stream[0] = positive
        else:
            stream[-2:] = [positive, negative]
    return tuple(stream), omission


__all__ = [
    "SemanticCheckpointLimits",
    "SemanticCheckpointRuntime",
    "SparseMultilingualSemanticAdapter",
    "SparseSemanticClaim",
    "SparseSemanticQuery",
    "claim_stream",
    "noise_claim",
]
