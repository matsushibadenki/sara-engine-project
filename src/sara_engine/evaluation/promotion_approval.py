"""Evidence-bound human approval for next-level promotion reviews."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any, Dict, Mapping


APPROVAL_SCHEMA = "sara-next-level-human-approval-v1"


def evidence_fingerprint(reports: Mapping[str, Mapping[str, Any]]) -> str:
    payload = {
        key: value
        for key, value in sorted(reports.items())
        if key != "human_approval"
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def build_approval(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    reviewer: str,
    note: str = "",
) -> Dict[str, Any]:
    if not str(reviewer).strip():
        raise ValueError("reviewer is required")
    return {
        "schema": APPROVAL_SCHEMA,
        "approved": True,
        "reviewer": str(reviewer).strip(),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "note": str(note),
        "evidence_fingerprint": evidence_fingerprint(reports),
        "scope": "next-level-phases-21-25-and-independent-gates",
    }


def validate_approval(
    approval: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
) -> bool:
    return bool(
        approval.get("schema") == APPROVAL_SCHEMA
        and approval.get("approved") is True
        and str(approval.get("reviewer", "")).strip()
        and str(approval.get("reviewed_at", "")).strip()
        and approval.get("evidence_fingerprint") == evidence_fingerprint(reports)
    )


__all__ = ["APPROVAL_SCHEMA", "build_approval", "evidence_fingerprint", "validate_approval"]
