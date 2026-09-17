"""Append-only intent-before-outcome capture for development discoveries."""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path

from sara_engine.utils.project_paths import (
    ensure_allowed_output_path, ensure_parent_directory, workspace_path,
)

from .discovery_replay import DiscoveryReplayWorld, ReplayRecord
from .discovery_export import SOURCE_SCHEMA


CAPTURE_SCHEMA = "sara-discovery-capture-v1"
GENESIS_HASH = "0" * 64
MAX_CAPTURE_BYTES = 4 * 1024 * 1024
MAX_CAPTURE_EVENTS = 2047
MAX_CAPTURE_LINE_BYTES = 2048


@dataclass(frozen=True)
class CaptureIntent:
    sequence: int
    node_id: str
    parent_id: str | None
    hypothesis_id: str
    policy_id: str
    candidate_sha256: str
    preregistration_sha256: str
    evaluator_id: str
    split: str = "development"


@dataclass(frozen=True)
class CaptureOutcome:
    sequence: int
    node_id: str
    status: str
    score: float | None
    cpu_ms: int
    event_count: int
    state_bytes: int


CaptureEvent = CaptureIntent | CaptureOutcome


@dataclass(frozen=True)
class CaptureView:
    events: tuple[CaptureEvent, ...]
    pending_node_ids: tuple[str, ...]
    head_sha256: str


@dataclass(frozen=True)
class CaptureProjection:
    capture_head_sha256: str
    snapshot_sha256: str
    snapshot_json: bytes
    record_count: int


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate capture key")
        result[key] = value
    return result


def _record(intent: CaptureIntent, outcome: CaptureOutcome | None) -> ReplayRecord:
    root = intent.sequence == 0
    return ReplayRecord(
        sequence=intent.sequence, node_id=intent.node_id,
        parent_id=intent.parent_id, hypothesis_id=intent.hypothesis_id,
        policy_id=intent.policy_id, candidate_sha256=intent.candidate_sha256,
        source_sha256=GENESIS_HASH,
        preregistration_sha256=intent.preregistration_sha256,
        evaluator_id=intent.evaluator_id, split=intent.split,
        status="root" if root else (outcome.status if outcome else "missing"),
        score=None if root or outcome is None else outcome.score,
        cpu_ms=0 if root or outcome is None else outcome.cpu_ms,
        event_count=0 if root or outcome is None else outcome.event_count,
        state_bytes=0 if root or outcome is None else outcome.state_bytes,
    )


def _validate_events(events: tuple[CaptureEvent, ...], head: str) -> CaptureView:
    intents: list[CaptureIntent] = []
    completed: dict[str, CaptureOutcome] = {}
    for event in events:
        if isinstance(event, CaptureIntent):
            if event.sequence != len(intents) or any(item.node_id == event.node_id for item in intents):
                raise ValueError("Capture intent sequence or identity is invalid")
            if event.sequence and event.parent_id != intents[0].node_id and event.parent_id not in completed:
                raise ValueError("Capture parent outcome must precede child intent")
            intents.append(event)
        elif isinstance(event, CaptureOutcome):
            if type(event.sequence) is not int or event.sequence <= 0 or event.sequence >= len(intents):
                raise ValueError("Capture outcome has no prior intent")
            intent = intents[event.sequence]
            if event.node_id != intent.node_id or event.node_id in completed:
                raise ValueError("Capture outcome identity is duplicate or mismatched")
            completed[event.node_id] = event
        else:
            raise ValueError("Invalid capture event type")
    if intents:
        DiscoveryReplayWorld(tuple(_record(item, completed.get(item.node_id))
                                   for item in intents))
    return CaptureView(events, tuple(item.node_id for item in intents[1:]
                                     if item.node_id not in completed), head)


class DiscoveryCaptureLog:
    """Single-host source log; capture alone never grants export approval."""

    def __init__(self, path: str) -> None:
        if not isinstance(path, str) or not path:
            raise ValueError("Capture path is required")
        workspace = Path(workspace_path()).resolve()
        resolved = Path(ensure_allowed_output_path(path))
        if not resolved.is_relative_to(workspace) or resolved.suffix != ".jsonl" or resolved.is_symlink():
            raise ValueError("Capture log must be a non-symlink JSONL file under workspace")
        self.path = Path(ensure_parent_directory(str(resolved)))
        self.lock_path = Path(ensure_allowed_output_path(str(self.path) + ".lock"))

    def _read_locked(self) -> CaptureView:
        if self.path.is_symlink():
            raise ValueError("Capture log symlinks are not allowed")
        try:
            with self.path.open("rb") as handle:
                raw = handle.read(MAX_CAPTURE_BYTES + 1)
        except FileNotFoundError:
            return CaptureView((), (), GENESIS_HASH)
        if len(raw) > MAX_CAPTURE_BYTES or (raw and not raw.endswith(b"\n")):
            raise ValueError("Capture log is oversized or incomplete")
        lines = raw.splitlines()
        if len(lines) > MAX_CAPTURE_EVENTS:
            raise ValueError("Capture event budget exceeded")
        events: list[CaptureEvent] = []
        previous = GENESIS_HASH
        for line in lines:
            if not line or len(line) > MAX_CAPTURE_LINE_BYTES:
                raise ValueError("Invalid capture line size")
            try:
                value = json.loads(line, object_pairs_hook=_unique_object)
                if not isinstance(value, dict) or set(value) != {
                    "schema", "kind", "event", "previous_sha256", "entry_sha256",
                } or value["schema"] != CAPTURE_SCHEMA or value["previous_sha256"] != previous:
                    raise ValueError("Invalid capture chain")
                shape = {"intent": CaptureIntent, "outcome": CaptureOutcome}.get(value["kind"])
                if shape is None or not isinstance(value["event"], dict) or set(value["event"]) != set(shape.__dataclass_fields__):
                    raise ValueError("Invalid capture event fields")
                event = shape(**value["event"])
                payload = {key: value[key] for key in ("schema", "kind", "event", "previous_sha256")}
                expected = sha256(_canonical(payload)).hexdigest()
                if value["entry_sha256"] != expected or line != _canonical(value):
                    raise ValueError("Capture entry hash or canonical encoding failed")
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
                raise ValueError("Invalid capture encoding") from exc
            events.append(event)
            previous = expected
        return _validate_events(tuple(events), previous)

    def load(self) -> CaptureView:
        return self._locked(None)

    def project_source_snapshot(self, *, expected_head_sha256: str) -> CaptureProjection:
        """Prepare an unapproved development snapshot from a complete pinned log."""
        view = self._locked(None, expected_head_sha256)
        if view.pending_node_ids:
            raise ValueError("Capture has pending outcomes")
        intents = [event for event in view.events if isinstance(event, CaptureIntent)]
        outcomes = {event.node_id: event for event in view.events
                    if isinstance(event, CaptureOutcome)}
        if len(intents) < 2:
            raise ValueError("Capture has no completed discovery action")
        records = tuple(_record(intent, outcomes.get(intent.node_id))
                        for intent in intents)
        DiscoveryReplayWorld(records)
        payload = {
            "schema": SOURCE_SCHEMA,
            "split": "development",
            "records": [{key: value for key, value in record.__dict__.items()
                         if key != "source_sha256"} for record in records],
        }
        snapshot = _canonical(payload)
        return CaptureProjection(view.head_sha256, sha256(snapshot).hexdigest(),
                                 snapshot, len(records))

    def begin(self, intent: CaptureIntent, *, expected_head_sha256: str | None = None) -> str:
        if type(intent) is not CaptureIntent:
            raise ValueError("Capture begin requires an intent")
        return self._locked(intent, expected_head_sha256).head_sha256

    def complete(self, outcome: CaptureOutcome, *, expected_head_sha256: str | None = None) -> str:
        if type(outcome) is not CaptureOutcome:
            raise ValueError("Capture completion requires an outcome")
        return self._locked(outcome, expected_head_sha256).head_sha256

    def _locked(self, event: CaptureEvent | None, expected_head_sha256: str | None = None) -> CaptureView:
        if expected_head_sha256 is not None and (not isinstance(expected_head_sha256, str)
                                                 or len(expected_head_sha256) != 64
                                                 or any(ch not in "0123456789abcdef" for ch in expected_head_sha256)):
            raise ValueError("Invalid expected capture head")
        if self.lock_path.is_symlink():
            raise ValueError("Capture lock symlinks are not allowed")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.lock_path, flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            view = self._read_locked()
            if expected_head_sha256 is not None and view.head_sha256 != expected_head_sha256:
                raise ValueError("Capture head changed")
            if event is None:
                return view
            if type(event) not in (CaptureIntent, CaptureOutcome):
                raise ValueError("Invalid capture event type")
            next_view = _validate_events((*view.events, event), view.head_sha256)
            if len(next_view.events) > MAX_CAPTURE_EVENTS:
                raise ValueError("Capture event budget exceeded")
            kind = "intent" if isinstance(event, CaptureIntent) else "outcome"
            payload = {"schema": CAPTURE_SCHEMA, "kind": kind, "event": event.__dict__,
                       "previous_sha256": view.head_sha256}
            entry = {**payload, "entry_sha256": sha256(_canonical(payload)).hexdigest()}
            line = _canonical(entry) + b"\n"
            size = self.path.stat().st_size if self.path.exists() else 0
            if len(line) - 1 > MAX_CAPTURE_LINE_BYTES or size + len(line) > MAX_CAPTURE_BYTES:
                raise ValueError("Capture byte budget exceeded")
            write_flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                write_flags |= os.O_NOFOLLOW
            output = os.open(self.path, write_flags, 0o600)
            try:
                if os.write(output, line) != len(line):
                    raise OSError("Short capture append")
                os.fsync(output)
            finally:
                os.close(output)
            parent = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
            return CaptureView(next_view.events, next_view.pending_node_ids,
                               entry["entry_sha256"])
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
