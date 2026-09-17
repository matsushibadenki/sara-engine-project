"""Bounded append-only, hash-chained development discovery journal."""
from __future__ import annotations

import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import RLock
from typing import Callable, Iterable

from sara_engine.utils.project_paths import (
    ensure_allowed_output_path, ensure_parent_directory, workspace_path,
)

from .discovery_replay import DiscoveryReplayWorld, ReplayRecord, ReplayView


JOURNAL_SCHEMA = "sara-discovery-journal-v1"
GENESIS_HASH = "0" * 64
MAX_JOURNAL_BYTES = 4 * 1024 * 1024
MAX_LINE_BYTES = 2048
MAX_RECORDS = 1024


def _expected_head(value: str | None) -> None:
    if value is not None and (not isinstance(value, str) or len(value) != 64
                              or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError("Invalid expected journal head")


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _entry(record: ReplayRecord, previous_sha256: str) -> dict:
    payload = {"schema": JOURNAL_SCHEMA, "previous_sha256": previous_sha256,
               "record": record.__dict__}
    return {**payload, "entry_sha256": sha256(_canonical(payload)).hexdigest()}


class DiscoveryJournal:
    """Single-host journal with verified prefix and serialized append operations."""

    def __init__(self, path: str) -> None:
        if not isinstance(path, str) or not path:
            raise ValueError("Journal path is required")
        workspace = Path(workspace_path()).resolve()
        try:
            resolved = Path(ensure_allowed_output_path(path))
        except ValueError:
            raise ValueError("Discovery journal must be a JSONL file under workspace") from None
        if not resolved.is_relative_to(workspace) or resolved.suffix != ".jsonl":
            raise ValueError("Discovery journal must be a JSONL file under workspace")
        if resolved.is_symlink():
            raise ValueError("Journal symlinks are not allowed")
        self.path = Path(ensure_parent_directory(str(resolved)))
        self.lock_path = Path(ensure_allowed_output_path(str(self.path) + ".lock"))
        self._thread_lock = RLock()

    def _read_locked(self) -> tuple[tuple[ReplayRecord, ...], str]:
        if self.path.is_symlink():
            raise ValueError("Journal symlinks are not allowed")
        try:
            with self.path.open("rb") as handle:
                raw = handle.read(MAX_JOURNAL_BYTES + 1)
        except FileNotFoundError:
            return (), GENESIS_HASH
        if len(raw) > MAX_JOURNAL_BYTES or (raw and not raw.endswith(b"\n")):
            raise ValueError("Journal is oversized or has an incomplete final line")
        lines = raw.splitlines()
        if len(lines) > MAX_RECORDS:
            raise ValueError("Journal record budget exceeded")
        records = []
        previous = GENESIS_HASH
        for line in lines:
            if not line or len(line) > MAX_LINE_BYTES:
                raise ValueError("Invalid journal line size")
            try:
                value = json.loads(line, object_pairs_hook=self._unique_object)
                if not isinstance(value, dict) or set(value) != {
                    "schema", "previous_sha256", "record", "entry_sha256",
                }:
                    raise ValueError("Invalid journal entry fields")
                if value["schema"] != JOURNAL_SCHEMA or value["previous_sha256"] != previous:
                    raise ValueError("Invalid journal chain")
                record = ReplayRecord.from_mapping(value["record"])
                expected = _entry(record, previous)
                if value != expected or line != _canonical(expected):
                    raise ValueError("Journal entry is not canonical or fails its hash")
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
                raise ValueError("Invalid journal encoding") from exc
            records.append(record)
            previous = value["entry_sha256"]
        if records:
            DiscoveryReplayWorld(records, max_nodes=MAX_RECORDS)
        return tuple(records), previous

    @staticmethod
    def _unique_object(pairs: list[tuple[str, object]]) -> dict:
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate journal key")
            value[key] = item
        return value

    def _with_lock(self, operation: Callable[[], object]) -> object:
        with self._thread_lock:
            if self.lock_path.is_symlink():
                raise ValueError("Journal lock symlinks are not allowed")
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(self.lock_path, flags, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                return operation()
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def load(self, *, expected_head_sha256: str | None = None) -> tuple[ReplayRecord, ...]:
        _expected_head(expected_head_sha256)

        def read() -> tuple[ReplayRecord, ...]:
            records, head = self._read_locked()
            if expected_head_sha256 is not None and head != expected_head_sha256:
                raise ValueError("Journal head changed")
            return records

        return self._with_lock(read)  # type: ignore[return-value]

    def append(self, record: ReplayRecord, *, expected_head_sha256: str | None = None) -> str:
        return self.append_batch((record,), expected_head_sha256=expected_head_sha256)

    def append_batch(self, new_records: Iterable[ReplayRecord], *,
                     expected_head_sha256: str | None = None) -> str:
        materialized = tuple(new_records)
        if not materialized or any(not isinstance(record, ReplayRecord)
                                   for record in materialized):
            raise ValueError("Invalid discovery record batch")
        _expected_head(expected_head_sha256)

        def write() -> str:
            records, previous = self._read_locked()
            if expected_head_sha256 is not None and previous != expected_head_sha256:
                raise ValueError("Journal head changed")
            if len(records) + len(materialized) > MAX_RECORDS:
                raise ValueError("Journal record budget exceeded")
            DiscoveryReplayWorld((*records, *materialized), max_nodes=MAX_RECORDS)
            lines = []
            for record in materialized:
                entry = _entry(record, previous)
                line = _canonical(entry) + b"\n"
                if len(line) - 1 > MAX_LINE_BYTES:
                    raise ValueError("Journal entry exceeds line budget")
                lines.append(line)
                previous = entry["entry_sha256"]
            body = b"".join(lines)
            current_size = self.path.stat().st_size if self.path.exists() else 0
            if current_size + len(body) > MAX_JOURNAL_BYTES:
                raise ValueError("Journal byte budget exceeded")
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(self.path, flags, 0o600)
            try:
                if os.write(descriptor, body) != len(body):
                    raise OSError("Short journal append")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            parent = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
            return previous

        return self._with_lock(write)  # type: ignore[return-value]


def run_replay_policy(world: DiscoveryReplayWorld,
                      policy: Callable[[ReplayView], Iterable[str]],
                      *, max_rounds: int = 32) -> ReplayView:
    """Pass only the revealed view to a policy; never expose the full world."""
    if not isinstance(world, DiscoveryReplayWorld) or not callable(policy):
        raise ValueError("Invalid replay controller arguments")
    if type(max_rounds) is not int or not 1 <= max_rounds <= MAX_RECORDS:
        raise ValueError("Replay round budget must be positive")
    for _ in range(max_rounds):
        actions = tuple(policy(world.view))
        view = world.reveal(actions)
        if view.stopped:
            return view
    return world.view
