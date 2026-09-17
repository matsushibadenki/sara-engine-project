"""Prospective capture must persist intent before accepting an outcome."""
from dataclasses import replace
import json
import tempfile
from pathlib import Path

import pytest

from sara_engine.research import (
    CaptureIntent, CaptureOutcome, DiscoveryCaptureLog, audit_source_material,
    run_captured_development_action,
)
from sara_engine.utils.project_paths import ensure_output_directory, workspace_path


HASH = "a" * 64


@pytest.fixture
def capture_path():
    parent = ensure_output_directory(workspace_path("research_capture_tests"))
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        yield str(Path(temporary) / "capture.jsonl")


def intent(sequence, node_id, parent_id):
    return CaptureIntent(
        sequence=sequence, node_id=node_id, parent_id=parent_id,
        hypothesis_id="hypothesis-1", policy_id="fixed-policy-v1",
        candidate_sha256=HASH, preregistration_sha256=HASH,
        evaluator_id="frozen-evaluator-v1",
    )


def outcome(sequence, node_id, *, status="valid", score=0.75):
    return CaptureOutcome(sequence, node_id, status, score, 10, 2, 32)


def test_intent_is_durable_before_outcome_and_pending_survives_reload(capture_path):
    log = DiscoveryCaptureLog(capture_path)
    root_head = log.begin(intent(0, "root", None))
    pending_head = log.begin(intent(1, "branch-a", "root"), expected_head_sha256=root_head)
    pending = DiscoveryCaptureLog(capture_path).load()
    assert pending.pending_node_ids == ("branch-a",)
    assert pending.head_sha256 == pending_head
    with pytest.raises(ValueError, match="pending"):
        log.project_source_snapshot(expected_head_sha256=pending_head)
    finished_head = log.complete(outcome(1, "branch-a"), expected_head_sha256=pending_head)
    assert finished_head != pending_head
    assert DiscoveryCaptureLog(capture_path).load().pending_node_ids == ()
    projection = log.project_source_snapshot(expected_head_sha256=finished_head)
    assert projection.capture_head_sha256 == finished_head
    assert projection.record_count == 2
    snapshot = json.loads(projection.snapshot_json)
    assert snapshot["records"][1]["score"] == 0.75
    assert snapshot["records"][1]["cpu_ms"] == 10
    assert "source_sha256" not in snapshot["records"][1]
    source_path = Path(capture_path).with_name("source.json")
    source_path.write_bytes(projection.snapshot_json)
    source = audit_source_material(str(source_path))
    assert source.source_sha256 == projection.snapshot_sha256
    assert source.record_count == 2
    with pytest.raises(ValueError, match="head changed"):
        log.project_source_snapshot(expected_head_sha256=pending_head)


def test_outcome_without_intent_duplicate_and_invalid_order_never_write(capture_path):
    log = DiscoveryCaptureLog(capture_path)
    with pytest.raises(ValueError, match="requires an intent"):
        log.begin(outcome(1, "branch-a"))
    with pytest.raises(ValueError, match="requires an outcome"):
        log.complete(intent(0, "root", None))
    with pytest.raises(ValueError, match="prior intent"):
        log.complete(outcome(1, "branch-a"))
    assert not Path(capture_path).exists()
    log.begin(intent(0, "root", None))
    with pytest.raises(ValueError, match="prior intent"):
        log.complete(outcome(1, "branch-a"))
    log.begin(intent(1, "branch-a", "root"))
    before = Path(capture_path).read_bytes()
    for invalid in (outcome(1, "wrong"), replace(outcome(1, "branch-a"), score=float("nan")),
                    replace(outcome(1, "branch-a"), status="valid", score=None)):
        with pytest.raises(ValueError):
            log.complete(invalid)
    with pytest.raises(ValueError, match="parent outcome"):
        log.begin(intent(2, "child-a", "branch-a"))
    assert Path(capture_path).read_bytes() == before
    log.complete(outcome(1, "branch-a"))
    with pytest.raises(ValueError, match="duplicate"):
        log.complete(outcome(1, "branch-a"))
    log.begin(intent(2, "child-a", "branch-a"))


def test_capture_detects_tampering_and_stale_head(capture_path):
    log = DiscoveryCaptureLog(capture_path)
    first = log.begin(intent(0, "root", None))
    with pytest.raises(ValueError, match="head changed"):
        log.begin(intent(1, "branch-a", "root"), expected_head_sha256="0" * 64)
    assert log.load().head_sha256 == first
    raw = Path(capture_path).read_bytes()
    Path(capture_path).write_bytes(raw.replace(b"root", b"fake"))
    with pytest.raises(ValueError, match="hash"):
        log.load()


def test_capture_path_is_managed(capture_path):
    with pytest.raises(ValueError, match="workspace"):
        DiscoveryCaptureLog("/private/tmp/capture.jsonl")


def test_runner_persists_intent_before_evaluator_and_records_exact_outcome(capture_path):
    log = DiscoveryCaptureLog(capture_path)
    root_head = log.begin(intent(0, "root", None))

    def evaluate():
        observed = DiscoveryCaptureLog(capture_path).load()
        assert observed.pending_node_ids == ("branch-a",)
        assert isinstance(observed.events[-1], CaptureIntent)
        return outcome(1, "branch-a", score=0.875)

    result = run_captured_development_action(
        log, intent(1, "branch-a", "root"), evaluate,
        expected_head_sha256=root_head,
    )
    assert result.outcome.score == 0.875
    assert result.intent_head_sha256 != root_head
    assert result.outcome_head_sha256 == log.load().head_sha256
    assert log.load().pending_node_ids == ()


def test_runner_leaves_pending_on_exception_or_mismatched_outcome(capture_path):
    log = DiscoveryCaptureLog(capture_path)
    root_head = log.begin(intent(0, "root", None))
    called = []

    def evaluate():
        called.append(True)
        raise RuntimeError("Evaluator stopped")

    with pytest.raises(ValueError, match="head changed"):
        run_captured_development_action(log, intent(1, "branch-a", "root"),
                                        evaluate, expected_head_sha256="0" * 64)
    assert called == []
    with pytest.raises(RuntimeError, match="Evaluator stopped"):
        run_captured_development_action(log, intent(1, "branch-a", "root"),
                                        evaluate, expected_head_sha256=root_head)
    assert called == [True]
    assert log.load().pending_node_ids == ("branch-a",)
    first_pending_head = log.load().head_sha256
    with pytest.raises(ValueError, match="does not match"):
        run_captured_development_action(log, intent(2, "branch-b", "root"),
                                        lambda: outcome(2, "wrong"),
                                        expected_head_sha256=first_pending_head)
    assert log.load().pending_node_ids == ("branch-a", "branch-b")
