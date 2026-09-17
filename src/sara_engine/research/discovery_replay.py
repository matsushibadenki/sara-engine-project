"""Bounded replay over recorded research-discovery branches only."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from typing import Iterable, Mapping


SCHEMA = "sara-discovery-tree-v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_STATUS = frozenset({"valid", "negative", "failed", "missing"})


@dataclass(frozen=True)
class ReplayRecord:
    sequence: int
    node_id: str
    parent_id: str | None
    hypothesis_id: str
    policy_id: str
    candidate_sha256: str
    source_sha256: str
    preregistration_sha256: str
    evaluator_id: str
    split: str
    status: str
    score: float | None
    cpu_ms: int
    event_count: int
    state_bytes: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ReplayRecord":
        expected = frozenset(cls.__dataclass_fields__)
        if frozenset(value) != expected:
            raise ValueError("Discovery record fields must match the v1 schema exactly")
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ReplayView:
    revealed: tuple[ReplayRecord, ...]
    rounds: int
    reveals: int
    stopped: bool


def _identifier(value: object, label: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"Invalid {label}")


def _validate_record(record: ReplayRecord, *, root: bool) -> None:
    if type(record.sequence) is not int or record.sequence < 0:
        raise ValueError("Invalid record sequence")
    for label in ("node_id", "hypothesis_id", "policy_id", "evaluator_id"):
        _identifier(getattr(record, label), label)
    for label in ("candidate_sha256", "source_sha256", "preregistration_sha256"):
        if not isinstance(getattr(record, label), str) or _DIGEST.fullmatch(getattr(record, label)) is None:
            raise ValueError(f"Invalid {label}")
    if record.split != "development":
        raise ValueError("Only development trees are allowed in observed-only replay")
    if root:
        if record.parent_id is not None or record.status != "root" or record.score is not None:
            raise ValueError("Invalid root record")
    else:
        _identifier(record.parent_id, "parent_id")
        if record.status not in _STATUS:
            raise ValueError("Invalid discovery outcome status")
        if record.status in ("valid", "negative"):
            if type(record.score) not in (int, float) or not math.isfinite(record.score):
                raise ValueError("Scored outcomes require a finite score")
        elif record.score is not None:
            raise ValueError("Non-valid outcomes cannot carry a score")
    for label in ("cpu_ms", "event_count", "state_bytes"):
        value = getattr(record, label)
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid {label}")
    if root and (record.cpu_ms or record.event_count or record.state_bytes):
        raise ValueError("Root cannot carry execution cost")


class DiscoveryReplayWorld:
    """Read-only world; policies can see only nodes revealed by prior actions."""

    def __init__(self, records: Iterable[ReplayRecord], *, max_nodes: int = 1024,
                 max_batch: int = 8, max_reveals: int = 128) -> None:
        if any(type(value) is not int or value < 1
               for value in (max_nodes, max_batch, max_reveals)):
            raise ValueError("Replay bounds must be positive integers")
        materialized = tuple(records)
        if not materialized or len(materialized) > max_nodes:
            raise ValueError("Discovery tree is empty or exceeds the node budget")
        seen: dict[str, ReplayRecord] = {}
        children: dict[str, list[str]] = {}
        for index, record in enumerate(materialized):
            if not isinstance(record, ReplayRecord):
                raise ValueError("Invalid discovery record type")
            _validate_record(record, root=index == 0)
            if record.sequence != index or record.node_id in seen:
                raise ValueError("Discovery sequence or node identity is not unique and contiguous")
            if index:
                if record.parent_id not in seen:
                    raise ValueError("Parent must precede child")
                if record.evaluator_id != materialized[0].evaluator_id:
                    raise ValueError("Evaluator identity changed inside one replay tree")
                if record.parent_id != materialized[0].node_id and children.get(record.parent_id):
                    raise ValueError("Non-root branches must have one recorded continuation")
                children.setdefault(record.parent_id, []).append(record.node_id)
            seen[record.node_id] = record
        self._records = materialized
        self._by_id = seen
        self._children = {key: tuple(value) for key, value in children.items()}
        self._root_id = materialized[0].node_id
        self._revealed = {self._root_id}
        self._rounds = 0
        self._reveals = 0
        self._stopped = False
        self._max_batch = max_batch
        self._max_reveals = max_reveals

    @property
    def view(self) -> ReplayView:
        return ReplayView(
            revealed=tuple(record for record in self._records if record.node_id in self._revealed),
            rounds=self._rounds,
            reveals=self._reveals,
            stopped=self._stopped,
        )

    @property
    def tree_sha256(self) -> str:
        payload = {"schema": SCHEMA, "records": [record.__dict__ for record in self._records]}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256(canonical.encode()).hexdigest()

    def reveal(self, actions: Iterable[str]) -> ReplayView:
        """Atomically reveal recorded children of visible leaves or one root branch."""
        if self._stopped:
            raise ValueError("Replay has stopped")
        selected = tuple(actions)
        if not selected:
            self._stopped = True
            self._rounds += 1
            return self.view
        if len(selected) > self._max_batch or len(set(selected)) != len(selected):
            raise ValueError("Replay batch is too large or contains duplicate actions")
        additions: list[str] = []
        for action in selected:
            if action not in self._revealed:
                raise ValueError("Replay action is not visible")
            candidates = [node_id for node_id in self._children.get(action, ())
                          if node_id not in self._revealed]
            if not candidates:
                raise ValueError("Replay action has no recorded continuation")
            additions.append(candidates[0])
        if len(set(additions)) != len(additions):
            raise ValueError("Replay actions resolve to the same recorded node")
        if self._reveals + len(additions) > self._max_reveals:
            raise ValueError("Replay reveal budget exceeded")
        self._revealed.update(additions)
        self._reveals += len(additions)
        self._rounds += 1
        return self.view
