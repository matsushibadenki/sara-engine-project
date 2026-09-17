"""Explicit review gate for sanitized development discovery trees."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from typing import Iterable

from sara_engine.utils.project_paths import (
    INTERIM_DATA_DIR, PROCESSED_DATA_DIR, WORKSPACE_DIR,
    resolve_project_relative,
)

from .discovery_journal import DiscoveryJournal, GENESIS_HASH
from .discovery_replay import DiscoveryReplayWorld, ReplayRecord


REVIEW_SCHEMA = "sara-discovery-export-review-v1"
SOURCE_SCHEMA = "sara-discovery-source-snapshot-v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
MAX_SOURCE_BYTES = 4 * 1024 * 1024
_FORBIDDEN_SPLIT_KEYS = frozenset({"heldout", "held_out", "frozen_test", "sealed_test"})
_FORBIDDEN_SPLIT_VALUES = frozenset({"heldout", "held_out", "frozen_test", "sealed_test", "test"})


@dataclass(frozen=True)
class ExportReviewReceipt:
    schema: str
    reviewer_id: str
    source_sha256: str
    sanitized_tree_sha256: str
    approved: bool
    raw_text_removed: bool
    heldout_excluded: bool


@dataclass(frozen=True)
class ReviewedExport:
    record_count: int
    source_sha256: str
    sanitized_tree_sha256: str


@dataclass(frozen=True)
class SourceMaterialAudit:
    source_sha256: str
    byte_count: int
    split: str
    record_count: int
    tree_sha256: str


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate source material key")
        value[key] = item
    return value


def _has_sealed_marker(value: object) -> bool:
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, item in current.items():
                if key.casefold() in _FORBIDDEN_SPLIT_KEYS:
                    return True
                if key.casefold() == "split" and isinstance(item, str) and item.casefold() in _FORBIDDEN_SPLIT_VALUES:
                    return True
                pending.append(item)
        elif isinstance(current, list):
            pending.extend(current)
    return False


def audit_source_material(source_path: str) -> SourceMaterialAudit:
    """Read one managed development JSON snapshot; this is not human review."""
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("Source material path is required")
    requested = Path(resolve_project_relative(source_path))
    if requested.suffix != ".json" or requested.is_symlink():
        raise ValueError("Source material must be a non-symlink JSON file")
    resolved = requested.resolve()
    allowed = (Path(WORKSPACE_DIR).resolve(), Path(PROCESSED_DATA_DIR).resolve(),
               Path(INTERIM_DATA_DIR).resolve())
    if not any(resolved.is_relative_to(root) for root in allowed):
        raise ValueError("Source material must be under managed development directories")
    flags = os.O_RDONLY | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(requested, flags)
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Source material must be a regular file")
        if metadata.st_size > MAX_SOURCE_BYTES:
            raise ValueError("Source material exceeds the byte budget")
        raw = handle.read(MAX_SOURCE_BYTES + 1)
    if not raw or len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("Source material is empty or exceeds the byte budget")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Source material is not valid JSON") from exc
    if not isinstance(value, dict) or value.get("split") != "development":
        raise ValueError("Source material must declare the development split")
    if _has_sealed_marker(value):
        raise ValueError("Source material contains a held-out or sealed-test marker")
    if set(value) != {"schema", "split", "records"} or value["schema"] != SOURCE_SCHEMA:
        raise ValueError("Source material must match the discovery snapshot schema")
    source_rows = value["records"]
    if not isinstance(source_rows, list) or not 1 <= len(source_rows) <= 1024:
        raise ValueError("Source material record count is invalid")
    expected_fields = set(ReplayRecord.__dataclass_fields__) - {"source_sha256"}
    if any(not isinstance(row, dict) or set(row) != expected_fields for row in source_rows):
        raise ValueError("Source material record fields are invalid")
    digest = sha256(raw).hexdigest()
    records = tuple(ReplayRecord.from_mapping({**row, "source_sha256": digest})
                    for row in source_rows)
    world = DiscoveryReplayWorld(records)
    return SourceMaterialAudit(digest, len(raw), "development", len(records),
                               world.tree_sha256)


def validate_reviewed_export(records: Iterable[ReplayRecord],
                             receipt: ExportReviewReceipt) -> ReviewedExport:
    """Bind a declared human review to one strict, sanitized tree."""
    if not isinstance(receipt, ExportReviewReceipt) or receipt.schema != REVIEW_SCHEMA:
        raise ValueError("Invalid export review receipt")
    if not isinstance(receipt.reviewer_id, str) or _IDENTIFIER.fullmatch(receipt.reviewer_id) is None:
        raise ValueError("Invalid reviewer identity")
    for value in (receipt.source_sha256, receipt.sanitized_tree_sha256):
        if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
            raise ValueError("Invalid review digest")
    if receipt.approved is not True or receipt.raw_text_removed is not True or receipt.heldout_excluded is not True:
        raise ValueError("Export review has not approved this sanitized development tree")
    materialized = tuple(records)
    world = DiscoveryReplayWorld(materialized)
    if any(record.source_sha256 != receipt.source_sha256 for record in materialized):
        raise ValueError("Source lineage does not match the review receipt")
    if world.tree_sha256 != receipt.sanitized_tree_sha256:
        raise ValueError("Sanitized tree does not match the review receipt")
    return ReviewedExport(len(materialized), receipt.source_sha256,
                          receipt.sanitized_tree_sha256)


def export_reviewed_tree(journal: DiscoveryJournal,
                         records: Iterable[ReplayRecord],
                         receipt: ExportReviewReceipt, *,
                         source_path: str) -> str:
    """Export one reviewed tree into a new managed journal, never append to an old tree."""
    if not isinstance(journal, DiscoveryJournal):
        raise ValueError("A managed discovery journal is required")
    materialized = tuple(records)
    validate_reviewed_export(materialized, receipt)
    source = audit_source_material(source_path)
    if source.source_sha256 != receipt.source_sha256:
        raise ValueError("Source file digest does not match the review receipt")
    if source.tree_sha256 != receipt.sanitized_tree_sha256:
        raise ValueError("Source file records do not match the reviewed tree")
    return journal.append_batch(materialized, expected_head_sha256=GENESIS_HASH)
