"""Deterministic verification receipts for durable Event Memory admission."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Dict, Iterable, Mapping, Tuple


RECEIPT_SCHEMA = "sara-verification-receipt-v1"


def evidence_digest(value: Any) -> str:
    """Return a stable digest for JSON-compatible evidence."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class VerificationReceipt:
    verifier_id: str
    verifier_version: str
    decision: str
    evidence_digest: str
    source_refs: Tuple[str, ...]
    source_revision: str
    observed: bool
    source_backed: bool
    verified: bool
    contradicted: bool = False
    abstained: bool = False
    schema: str = RECEIPT_SCHEMA
    integrity_digest: str = ""

    def payload(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "verifier_id": self.verifier_id,
            "verifier_version": self.verifier_version,
            "decision": self.decision,
            "evidence_digest": self.evidence_digest,
            "source_refs": list(self.source_refs),
            "source_revision": self.source_revision,
            "observed": self.observed,
            "source_backed": self.source_backed,
            "verified": self.verified,
            "contradicted": self.contradicted,
            "abstained": self.abstained,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self.payload(), "integrity_digest": self.integrity_digest}

    def is_valid(self) -> bool:
        if self.schema != RECEIPT_SCHEMA:
            return False
        if not self.verifier_id or not self.verifier_version or not self.decision:
            return False
        if not self.evidence_digest or len(self.evidence_digest) != 64:
            return False
        if self.source_backed and not self.source_refs:
            return False
        return self.integrity_digest == evidence_digest(self.payload())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerificationReceipt":
        payload = dict(value)
        payload["source_refs"] = tuple(str(item) for item in payload.get("source_refs", ()))
        return cls(**payload)


def issue_verification_receipt(
    *,
    verifier_id: str,
    verifier_version: str,
    decision: str,
    evidence: Any,
    source_refs: Iterable[str],
    source_revision: str,
    observed: bool,
    source_backed: bool,
    verified: bool,
    contradicted: bool = False,
    abstained: bool = False,
) -> VerificationReceipt:
    normalized_refs = tuple(sorted({str(item) for item in source_refs if str(item)}))
    receipt = VerificationReceipt(
        verifier_id=str(verifier_id),
        verifier_version=str(verifier_version),
        decision=str(decision),
        evidence_digest=evidence_digest(evidence),
        source_refs=normalized_refs,
        source_revision=str(source_revision),
        observed=bool(observed),
        source_backed=bool(source_backed),
        verified=bool(verified),
        contradicted=bool(contradicted),
        abstained=bool(abstained),
    )
    return VerificationReceipt(
        verifier_id=receipt.verifier_id,
        verifier_version=receipt.verifier_version,
        decision=receipt.decision,
        evidence_digest=receipt.evidence_digest,
        source_refs=receipt.source_refs,
        source_revision=receipt.source_revision,
        observed=receipt.observed,
        source_backed=receipt.source_backed,
        verified=receipt.verified,
        contradicted=receipt.contradicted,
        abstained=receipt.abstained,
        integrity_digest=evidence_digest(receipt.payload()),
    )


__all__ = [
    "RECEIPT_SCHEMA",
    "VerificationReceipt",
    "evidence_digest",
    "issue_verification_receipt",
]
