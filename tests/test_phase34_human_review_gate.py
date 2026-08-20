from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.phase34_human_review import (
    build_empty_ledger,
    canonical_digest,
    evaluate_review_gate,
    record_decision,
    validate_ledger,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUEST_PATH = (
    PROJECT_ROOT
    / "workspace"
    / "evaluation"
    / "phase34_transcribed_excerpt_human_review_request.json"
)


def _request():
    with REQUEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _record_aligned(request, ledger, record_id, *, reviewer="human-reviewer"):
    target = next(
        item for item in request["targets"] if item["record_id"] == record_id
    )
    return record_decision(
        request,
        ledger,
        record_id=record_id,
        authoritative_section_locator=f"section-for:{record_id}",
        authoritative_text_hash=canonical_digest(
            {"record_id": record_id, "source_ref": target["source_ref"]}
        ),
        alignment_decision="aligned",
        semantic_omission_or_distortion_found=False,
        reviewer=reviewer,
        reviewed_at="2026-08-20T10:00:00+09:00",
        notes="Human compared the stored excerpt with the cited section.",
        human_attestation=True,
    )


def test_empty_ledger_keeps_review_gate_closed():
    request = _request()
    ledger = build_empty_ledger(request)
    report = evaluate_review_gate(request, ledger)

    assert report["validation"]["valid"] is True
    assert report["decision_count"] == 0
    assert report["pending_count"] == 6
    assert report["review_complete"] is False
    assert report["review_gate_passed"] is False
    assert report["semantic_delayed_recall_preregistration_ready"] is False
    assert report["promotion_ready"] is False


def test_six_hash_bound_aligned_decisions_open_only_semantic_preregistration_gate():
    request = _request()
    ledger = build_empty_ledger(request)
    for target in request["targets"]:
        ledger = _record_aligned(request, ledger, target["record_id"])

    report = evaluate_review_gate(request, ledger)

    assert report["validation"]["valid"] is True
    assert report["aligned_count"] == 6
    assert report["review_complete"] is True
    assert report["review_gate_passed"] is True
    assert report["semantic_delayed_recall_preregistration_ready"] is True
    assert report["promotion_ready"] is False
    assert report["next_action"] == "preregister_semantic_delayed_recall_workload"


def test_misaligned_decision_completes_review_but_keeps_gate_closed():
    request = _request()
    ledger = build_empty_ledger(request)
    for target in request["targets"][:-1]:
        ledger = _record_aligned(request, ledger, target["record_id"])
    target = request["targets"][-1]
    ledger = record_decision(
        request,
        ledger,
        record_id=target["record_id"],
        authoritative_section_locator="authoritative-section",
        authoritative_text_hash=canonical_digest("authoritative-text"),
        alignment_decision="misaligned",
        semantic_omission_or_distortion_found=True,
        reviewer="human-reviewer",
        reviewed_at="2026-08-20T11:00:00+09:00",
        notes="The stored excerpt changes the authoritative meaning.",
        human_attestation=True,
    )

    report = evaluate_review_gate(request, ledger)

    assert report["review_complete"] is True
    assert report["review_gate_passed"] is False
    assert report["misaligned_count"] == 1
    assert (
        report["next_action"]
        == "resolve_misaligned_sources_under_new_preregistration"
    )


def test_decision_replacement_requires_explicit_operator_intent():
    request = _request()
    target_id = request["targets"][0]["record_id"]
    ledger = _record_aligned(request, build_empty_ledger(request), target_id)

    with pytest.raises(ValueError, match="already exists"):
        record_decision(
            request,
            ledger,
            record_id=target_id,
            authoritative_section_locator="different-section",
            authoritative_text_hash=canonical_digest("different"),
            alignment_decision="unresolved",
            semantic_omission_or_distortion_found=None,
            reviewer="human-reviewer",
            reviewed_at="2026-08-20T12:00:00+09:00",
            notes="Needs another source comparison.",
            human_attestation=True,
        )


def test_tampered_request_or_decision_fails_closed():
    request = _request()
    ledger = _record_aligned(
        request, build_empty_ledger(request), request["targets"][0]["record_id"]
    )
    tampered_ledger = copy.deepcopy(ledger)
    tampered_ledger["decisions"][0]["stored_excerpt_hash"] = "0" * 64

    validation = validate_ledger(request, tampered_ledger)
    report = evaluate_review_gate(request, tampered_ledger)

    assert validation["valid"] is False
    assert any(
        error.startswith("decision_excerpt_hash_mismatch")
        for error in validation["errors"]
    )
    assert report["review_gate_passed"] is False
    assert report["next_action"] == "repair_invalid_decision_ledger"

    tampered_request = copy.deepcopy(request)
    tampered_request["targets"][0]["source_hash"] = "f" * 64
    with pytest.raises(ValueError, match="invalid review ledger"):
        _record_aligned(
            tampered_request,
            ledger,
            tampered_request["targets"][1]["record_id"],
        )
