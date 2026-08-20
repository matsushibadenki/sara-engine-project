"""Immutable review-support snapshot and comparison packet for Phase 34."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Dict, List, Mapping, Sequence

from sara_engine.evaluation.phase34_cpython_snapshot import COMMIT
from sara_engine.evaluation.phase34_human_review import canonical_digest, validate_request


SCHEMA = "sara-phase34-transcribed-excerpt-review-support-preregistration-v1"
EXPERIMENT_ID = "phase34-transcribed-excerpt-review-support-v1"
CPYTHON_REPOSITORY_URL = "https://github.com/python/cpython.git"
RFC_URL = "https://www.rfc-editor.org/rfc/rfc9110.txt"
MAX_SOURCE_BYTES = 2_000_000
SOURCES = (
    {
        "source_id": "cpython-argparse",
        "transport": "git_smart_http_shallow_fetch",
        "repository_path": "Doc/library/argparse.rst",
        "source_ref": (
            "https://github.com/python/cpython/blob/"
            f"{COMMIT}/Doc/library/argparse.rst"
        ),
        "source_revision": COMMIT,
    },
    {
        "source_id": "cpython-pathlib",
        "transport": "git_smart_http_shallow_fetch",
        "repository_path": "Doc/library/pathlib.rst",
        "source_ref": (
            "https://github.com/python/cpython/blob/"
            f"{COMMIT}/Doc/library/pathlib.rst"
        ),
        "source_revision": COMMIT,
    },
    {
        "source_id": "rfc9110",
        "transport": "https_rfc_editor",
        "repository_path": "",
        "source_ref": RFC_URL,
        "source_revision": "RFC 9110, June 2022",
    },
)
TARGET_BINDINGS = {
    "arch-migration-python-001": {
        "source_id": "cpython-argparse",
        "authoritative_locator": "Doc/library/argparse.rst module introduction",
    },
    "arch-migration-python-002": {
        "source_id": "cpython-argparse",
        "authoritative_locator": "Doc/library/argparse.rst ArgumentParser.add_argument and parse_args",
    },
    "arch-migration-python-003": {
        "source_id": "cpython-pathlib",
        "authoritative_locator": "Doc/library/pathlib.rst module introduction",
    },
    "arch-migration-ietf-001": {
        "source_id": "rfc9110",
        "authoritative_locator": "RFC 9110 Abstract",
    },
    "arch-migration-ietf-002": {
        "source_id": "rfc9110",
        "authoritative_locator": "RFC 9110 Section 1.3 Core Semantics",
    },
    "arch-migration-ietf-003": {
        "source_id": "rfc9110",
        "authoritative_locator": "RFC 9110 Section 2.2 Requirements Notation",
    },
}


def build_preregistration(request: Mapping[str, Any]) -> Dict[str, Any]:
    request_validation = validate_request(request)
    if not request_validation["valid"]:
        raise ValueError("invalid review request: " + "; ".join(request_validation["errors"]))
    if set(request_validation["target_ids"]) != set(TARGET_BINDINGS):
        raise ValueError("review request targets do not match the frozen support bindings")
    candidate: Dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "registered_before_collection": True,
        "request_fingerprint": request_validation["request_fingerprint"],
        "cpython_repository_url": CPYTHON_REPOSITORY_URL,
        "cpython_commit": COMMIT,
        "rfc_url": RFC_URL,
        "sources": [dict(source) for source in SOURCES],
        "source_count": len(SOURCES),
        "target_bindings": {
            key: dict(TARGET_BINDINGS[key]) for key in sorted(TARGET_BINDINGS)
        },
        "collection_policy": {
            "exact_sources_only": True,
            "max_source_bytes": MAX_SOURCE_BYTES,
            "utf8_required": True,
            "content_truncation_allowed": False,
            "overwrite_on_mismatch": False,
            "mutate_review_request": False,
            "create_human_decision": False,
        },
        "claim_boundaries": {
            "review_support_only": True,
            "automated_alignment_decision_allowed": False,
            "human_review_complete": False,
            "semantic_preregistration_ready": False,
            "promotion_ready": False,
        },
    }
    candidate["protocol_fingerprint"] = canonical_digest(candidate)
    return candidate


def validate_preregistration(
    registration: Mapping[str, Any], request: Mapping[str, Any]
) -> Dict[str, Any]:
    errors: List[str] = []
    try:
        expected = build_preregistration(request)
    except ValueError as exc:
        return {"valid": False, "errors": [str(exc)], "computed_fingerprint": ""}
    for key, expected_value in expected.items():
        if registration.get(key) != expected_value:
            errors.append(f"review_support_registration_mismatch:{key}")
    return {
        "valid": not errors,
        "errors": errors,
        "computed_fingerprint": canonical_digest(
            {key: value for key, value in registration.items() if key != "protocol_fingerprint"}
        ),
    }


def build_source_rows(
    registration: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    git_blob_loader: Callable[[str], bytes],
    http_loader: Callable[[str], bytes],
    collection_time: str,
) -> List[Dict[str, Any]]:
    validation = validate_preregistration(registration, request)
    if not validation["valid"]:
        raise ValueError("invalid review-support preregistration: " + "; ".join(validation["errors"]))
    rows: List[Dict[str, Any]] = []
    for source in registration["sources"]:
        if source["transport"] == "git_smart_http_shallow_fetch":
            payload = git_blob_loader(str(source["repository_path"]))
        else:
            payload = http_loader(str(source["source_ref"]))
        if len(payload) > MAX_SOURCE_BYTES:
            raise ValueError(f"review support source exceeds byte ceiling: {source['source_id']}")
        try:
            content = payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError as exc:
            raise ValueError(f"review support source is not UTF-8: {source['source_id']}") from exc
        if len(content.strip()) < 200:
            raise ValueError(f"review support source is too short: {source['source_id']}")
        rows.append(
            {
                "schema": "sara-phase34-transcribed-excerpt-review-support-source-v1",
                "source_id": source["source_id"],
                "source_ref": source["source_ref"],
                "source_revision": source["source_revision"],
                "repository_path": source["repository_path"],
                "transport": source["transport"],
                "content": content,
                "source_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "response_body_hash": hashlib.sha256(payload).hexdigest(),
                "collection_time": collection_time,
                "content_truncated": False,
                "observed_only": True,
                "compliance_level": "allow",
                "protocol_fingerprint": registration["protocol_fingerprint"],
            }
        )
    return rows


def _normalized_tokens(value: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _paragraphs(content: str) -> List[str]:
    values = []
    for block in re.split(r"\n\s*\n", content):
        normalized = " ".join(line.strip() for line in block.splitlines()).strip()
        if len(_normalized_tokens(normalized)) >= 5:
            values.append(normalized)
    return values


def _overlap(left: str, right: str) -> float:
    left_tokens = set(_normalized_tokens(left))
    right_tokens = set(_normalized_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def build_comparison_packet(
    request: Mapping[str, Any],
    registration: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    validation = validate_preregistration(registration, request)
    if not validation["valid"]:
        raise ValueError("invalid review-support preregistration: " + "; ".join(validation["errors"]))
    by_source = {str(row.get("source_id", "")): row for row in rows}
    if set(by_source) != {source["source_id"] for source in SOURCES}:
        raise ValueError("review support rows do not match the frozen sources")
    comparisons = []
    for target in request["targets"]:
        record_id = str(target["record_id"])
        binding = TARGET_BINDINGS[record_id]
        row = by_source[binding["source_id"]]
        content = str(row.get("content", ""))
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != row.get("source_hash"):
            raise ValueError(f"review support source hash mismatch: {binding['source_id']}")
        ranked = sorted(
            (
                (_overlap(str(target["stored_excerpt"]), paragraph), index, paragraph)
                for index, paragraph in enumerate(_paragraphs(content))
            ),
            key=lambda item: (-item[0], item[1]),
        )[:3]
        candidates = [
            {
                "rank": rank,
                "paragraph_index": index,
                "token_jaccard": round(score, 6),
                "authoritative_text_hash": hashlib.sha256(
                    paragraph.encode("utf-8")
                ).hexdigest(),
                "authoritative_text": paragraph,
            }
            for rank, (score, index, paragraph) in enumerate(ranked, start=1)
        ]
        comparisons.append(
            {
                "record_id": record_id,
                "stored_excerpt_hash": target["source_hash"],
                "stored_excerpt": target["stored_excerpt"],
                "cited_source_ref": target["source_ref"],
                "authoritative_source_ref": row["source_ref"],
                "authoritative_source_revision": row["source_revision"],
                "authoritative_source_hash": row["source_hash"],
                "authoritative_locator": binding["authoritative_locator"],
                "stored_excerpt_exact_substring": str(target["stored_excerpt"]) in content,
                "candidate_paragraphs": candidates,
                "human_review": {
                    "selected_authoritative_text_hash": "",
                    "alignment_decision": "pending",
                    "semantic_omission_or_distortion_found": None,
                    "notes": "",
                },
            }
        )
    packet = {
        "schema": "sara-phase34-transcribed-excerpt-review-comparison-packet-v1",
        "observed_only": True,
        "request_fingerprint": registration["request_fingerprint"],
        "protocol_fingerprint": registration["protocol_fingerprint"],
        "source_snapshot_fingerprint": canonical_digest([dict(row) for row in rows]),
        "target_count": len(comparisons),
        "comparisons": comparisons,
        "automated_alignment_decision_made": False,
        "review_complete": False,
        "review_gate_passed": False,
        "promotion_ready": False,
    }
    packet["packet_fingerprint"] = canonical_digest(packet)
    return packet


__all__ = [
    "CPYTHON_REPOSITORY_URL",
    "EXPERIMENT_ID",
    "MAX_SOURCE_BYTES",
    "RFC_URL",
    "SCHEMA",
    "SOURCES",
    "TARGET_BINDINGS",
    "build_comparison_packet",
    "build_preregistration",
    "build_source_rows",
    "validate_preregistration",
]
