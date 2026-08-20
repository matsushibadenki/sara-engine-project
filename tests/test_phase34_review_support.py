from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sara_engine.evaluation.phase34_review_support import (
    RFC_URL,
    build_comparison_packet,
    build_preregistration,
    build_source_rows,
    validate_preregistration,
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


def _git_payload(path: str) -> bytes:
    if path.endswith("argparse.rst"):
        text = """
argparse module

The argparse module makes it easy to write user-friendly command-line interfaces.
The program defines what arguments it requires and parses values from sys.argv.

ArgumentParser.add_argument attaches an argument specification to the parser.
ArgumentParser.parse_args runs the parser and returns a Namespace.
"""
    else:
        text = """
pathlib module

This module offers classes representing filesystem paths with semantics
appropriate for different operating systems. Pure paths provide computational
operations without I/O and concrete paths also provide I/O operations.
"""
    return (text.strip() + "\n" + ("supporting documentation " * 20)).encode("utf-8")


def _http_payload(url: str) -> bytes:
    assert url == RFC_URL
    text = """
Abstract

The Hypertext Transfer Protocol is a stateless application-level protocol for
distributed, collaborative, hypertext information systems.

1.3. Core Semantics

HTTP provides a uniform interface for interacting with a resource regardless
of its type, nature, or implementation. Each message is either a request or a
response.

2.2. Requirements Notation

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, MAY, and OPTIONAL are to be
interpreted as described in BCP 14 when they appear in all capitals.
"""
    return (text.strip() + "\n" + ("standards text " * 20)).encode("utf-8")


def test_review_support_preregistration_is_request_bound_and_deterministic():
    request = _request()
    first = build_preregistration(request)
    second = build_preregistration(request)

    assert first == second
    assert first["registered_before_collection"] is True
    assert first["source_count"] == 3
    assert first["claim_boundaries"]["automated_alignment_decision_allowed"] is False
    assert validate_preregistration(first, request)["valid"] is True


def test_review_packet_ranks_source_paragraphs_without_making_decisions():
    request = _request()
    registration = build_preregistration(request)
    rows = build_source_rows(
        registration,
        request,
        git_blob_loader=_git_payload,
        http_loader=_http_payload,
        collection_time="2026-08-20T00:00:00Z",
    )
    packet = build_comparison_packet(request, registration, rows)

    assert len(rows) == 3
    assert packet["target_count"] == 6
    assert packet["automated_alignment_decision_made"] is False
    assert packet["review_complete"] is False
    assert packet["review_gate_passed"] is False
    assert packet["promotion_ready"] is False
    assert all(
        item["candidate_paragraphs"]
        and item["human_review"]["alignment_decision"] == "pending"
        and item["human_review"]["selected_authoritative_text_hash"] == ""
        for item in packet["comparisons"]
    )


def test_review_support_rejects_request_or_registration_drift():
    request = _request()
    registration = build_preregistration(request)
    tampered_registration = copy.deepcopy(registration)
    tampered_registration["sources"][0]["repository_path"] = "Doc/library/other.rst"

    validation = validate_preregistration(tampered_registration, request)

    assert validation["valid"] is False
    assert any(
        error == "review_support_registration_mismatch:sources"
        for error in validation["errors"]
    )
    with pytest.raises(ValueError, match="invalid review-support preregistration"):
        build_source_rows(
            tampered_registration,
            request,
            git_blob_loader=_git_payload,
            http_loader=_http_payload,
            collection_time="2026-08-20T00:00:00Z",
        )
