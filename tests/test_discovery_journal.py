"""Managed discovery journals remain append-only and fail closed on corruption."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
import tempfile
from pathlib import Path

import pytest

from sara_engine.research import (
    DiscoveryJournal, DiscoveryReplayWorld, ExportReviewReceipt,
    ReplayRecord, ReplayView, export_reviewed_tree, run_replay_policy,
    validate_reviewed_export,
)
from sara_engine.utils.project_paths import ensure_output_directory, workspace_path


HASH = "a" * 64


def record(sequence, node_id, parent_id, *, status="valid", score=1.0):
    return ReplayRecord(
        sequence=sequence,
        node_id=node_id,
        parent_id=parent_id,
        hypothesis_id="hypothesis-1",
        policy_id="fixed-policy-v1",
        candidate_sha256=HASH,
        source_sha256=HASH,
        preregistration_sha256=HASH,
        evaluator_id="frozen-evaluator-v1",
        split="development",
        status=status,
        score=score,
        cpu_ms=0 if sequence == 0 else 10,
        event_count=0 if sequence == 0 else 2,
        state_bytes=0 if sequence == 0 else 32,
    )


@pytest.fixture
def journal_path():
    directory = ensure_output_directory(workspace_path("research_replay_tests"))
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        yield str(Path(temporary) / "tree.jsonl")


def test_journal_append_reload_and_chain(journal_path):
    journal = DiscoveryJournal(journal_path)
    assert journal.load() == ()
    root = record(0, "root", None, status="root", score=None)
    child = record(1, "branch-a", "root", status="negative", score=0.25)
    first = journal.append(root)
    second = journal.append(child, expected_head_sha256=first)
    assert first != second
    assert DiscoveryJournal(journal_path).load(expected_head_sha256=second) == (root, child)
    with pytest.raises(ValueError, match="head changed"):
        journal.load(expected_head_sha256=first)
    lines = [json.loads(line) for line in Path(journal_path).read_text().splitlines()]
    assert lines[0]["previous_sha256"] == "0" * 64
    assert lines[0]["entry_sha256"] == first
    assert lines[1]["previous_sha256"] == first
    assert lines[1]["entry_sha256"] == second


def test_journal_rejects_invalid_append_without_mutation(journal_path):
    journal = DiscoveryJournal(journal_path)
    root = record(0, "root", None, status="root", score=None)
    journal.append(root)
    before = Path(journal_path).read_bytes()
    with pytest.raises(ValueError, match="Parent"):
        journal.append(record(1, "bad", "missing"))
    with pytest.raises(ValueError, match="development"):
        journal.append(replace(record(1, "bad", "root"), split="heldout"))
    with pytest.raises(ValueError, match="head changed"):
        journal.append(record(1, "bad", "root"), expected_head_sha256="0" * 64)
    assert Path(journal_path).read_bytes() == before


def test_journal_tamper_and_truncation_fail_closed(journal_path):
    journal = DiscoveryJournal(journal_path)
    journal.append(record(0, "root", None, status="root", score=None))
    journal.append(record(1, "branch-a", "root"))
    original = Path(journal_path).read_bytes()
    Path(journal_path).write_bytes(original.replace(b"branch-a", b"branch-z"))
    with pytest.raises(ValueError, match="hash"):
        journal.load()
    with pytest.raises(ValueError):
        journal.append(record(2, "child", "branch-a"))
    Path(journal_path).write_bytes(original[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        journal.load()


def test_controller_passes_only_revealed_view(journal_path):
    journal = DiscoveryJournal(journal_path)
    rows = (record(0, "root", None, status="root", score=None),
            record(1, "branch-a", "root"),
            record(2, "child-a", "branch-a"))
    for row in rows:
        journal.append(row)
    world = DiscoveryReplayWorld(journal.load())
    observed = []

    def policy(view):
        assert isinstance(view, ReplayView)
        assert not hasattr(view, "_records")
        names = [row.node_id for row in view.revealed]
        observed.append(names)
        if names == ["root"]:
            return ["root"]
        if names == ["root", "branch-a"]:
            return ["branch-a"]
        return []

    result = run_replay_policy(world, policy, max_rounds=3)
    assert observed == [["root"], ["root", "branch-a"],
                        ["root", "branch-a", "child-a"]]
    assert result.stopped is True
    assert result.reveals == 2


def test_journal_path_must_be_managed_workspace():
    with pytest.raises(ValueError, match="under workspace"):
        DiscoveryJournal("/private/tmp/discovery-tree.jsonl")


def test_same_sequence_concurrent_appends_admit_one(journal_path):
    DiscoveryJournal(journal_path).append(record(0, "root", None, status="root", score=None))

    def attempt(index):
        try:
            DiscoveryJournal(journal_path).append(record(1, f"branch-{index}", "root"))
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(attempt, range(4)))
    assert results.count(True) == 1
    assert len(DiscoveryJournal(journal_path).load()) == 2


def test_reviewed_export_binds_exact_sanitized_tree(journal_path):
    rows = (record(0, "root", None, status="root", score=None),
            record(1, "branch-a", "root", status="negative", score=0.25))
    receipt = ExportReviewReceipt(
        schema="sara-discovery-export-review-v1",
        reviewer_id="reviewer-1",
        source_sha256=HASH,
        sanitized_tree_sha256=DiscoveryReplayWorld(rows).tree_sha256,
        approved=True,
        raw_text_removed=True,
        heldout_excluded=True,
    )
    assert validate_reviewed_export(rows, receipt).record_count == 2
    journal = DiscoveryJournal(journal_path)
    head = export_reviewed_tree(journal, rows, receipt)
    assert journal.load(expected_head_sha256=head) == rows
    with pytest.raises(ValueError, match="head changed"):
        export_reviewed_tree(journal, rows, receipt)


def test_unapproved_or_mismatched_export_never_writes(journal_path):
    rows = (record(0, "root", None, status="root", score=None),
            record(1, "branch-a", "root"))
    receipt = ExportReviewReceipt(
        schema="sara-discovery-export-review-v1", reviewer_id="reviewer-1",
        source_sha256=HASH,
        sanitized_tree_sha256=DiscoveryReplayWorld(rows).tree_sha256,
        approved=True, raw_text_removed=True, heldout_excluded=True,
    )
    journal = DiscoveryJournal(journal_path)
    for invalid in (
        replace(receipt, approved=False),
        replace(receipt, raw_text_removed=False),
        replace(receipt, heldout_excluded=False),
        replace(receipt, sanitized_tree_sha256="b" * 64),
        replace(receipt, source_sha256="b" * 64),
    ):
        with pytest.raises(ValueError):
            export_reviewed_tree(journal, rows, invalid)
    with pytest.raises(ValueError, match="development"):
        export_reviewed_tree(journal, (rows[0], replace(rows[1], split="heldout")), receipt)
    assert not Path(journal_path).exists()


def test_invalid_batch_is_rejected_before_any_record_is_written(journal_path):
    journal = DiscoveryJournal(journal_path)
    with pytest.raises(ValueError, match="Parent"):
        journal.append_batch((record(0, "root", None, status="root", score=None),
                              record(1, "bad", "missing")))
    assert not Path(journal_path).exists()
