"""Explicit review gate for sanitized development discovery trees."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .discovery_journal import DiscoveryJournal, GENESIS_HASH
from .discovery_replay import DiscoveryReplayWorld, ReplayRecord


REVIEW_SCHEMA = "sara-discovery-export-review-v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


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
                         receipt: ExportReviewReceipt) -> str:
    """Export one reviewed tree into a new managed journal, never append to an old tree."""
    if not isinstance(journal, DiscoveryJournal):
        raise ValueError("A managed discovery journal is required")
    materialized = tuple(records)
    validate_reviewed_export(materialized, receipt)
    return journal.append_batch(materialized, expected_head_sha256=GENESIS_HASH)
