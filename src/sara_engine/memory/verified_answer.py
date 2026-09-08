"""Read-only answer binding for Event Memory compatibility evaluation.

Receipts establish content integrity inside a trusted verifier boundary; they
are not signatures or proof that an external factual claim is true.
"""

from dataclasses import dataclass
from typing import Optional

from sara_engine.memory.event_state_cache import EventStateEntry
from sara_engine.memory.verification_receipt import VerificationReceipt, evidence_digest


@dataclass(frozen=True)
class VerifiedAnswer:
    entry_id: str
    text: str
    language: str
    source_ref: str
    source_revision: str
    receipt: VerificationReceipt

    def evidence_payload(self) -> dict:
        return {
            "schema": "sara-verified-answer-binding-v1",
            "entry_id": self.entry_id,
            "text": self.text,
            "language": self.language,
            "source_ref": self.source_ref,
            "source_revision": self.source_revision,
        }


@dataclass(frozen=True)
class AnswerResolution:
    decision: str
    text: str = ""
    source_ref: str = ""
    source_revision: str = ""


def resolve_verified_answer(
    entry: Optional[EventStateEntry],
    answer: Optional[VerifiedAnswer],
    *,
    now_segment: int,
    language: str,
) -> AnswerResolution:
    """Resolve an already selected entry without retrieval or factual inference.

    Limit validation and hashing inputs before allocating a canonical payload.
    Never truncate evidence to make an invalid binding appear valid.
    """
    def abstain(reason: str) -> AnswerResolution:
        return AnswerResolution(reason)

    if entry is None or answer is None:
        return abstain("missing_evidence")
    if type(now_segment) is not int:
        return abstain("invalid_time")
    if language not in {"en", "ja", "zh-CN"} or answer.language != language:
        return abstain("language_mismatch")
    fields = (answer.entry_id, answer.source_ref, answer.source_revision)
    if any(not isinstance(value, str) or not value or len(value) > 512 for value in fields):
        return abstain("invalid_identity")
    if not isinstance(answer.text, str) or not answer.text.strip() or len(answer.text) > 4096:
        return abstain("answer_size_invalid")
    if not entry.observed or not entry.verified:
        return abstain("entry_unverified")
    if entry.time_segment > now_segment:
        return abstain("future_evidence")
    if entry.expires_at is not None and now_segment >= entry.expires_at:
        return abstain("expired_evidence")
    if (entry.entry_id, entry.source_ref, entry.source_revision) != fields:
        return abstain("entry_binding_mismatch")
    receipt = answer.receipt
    if not isinstance(receipt, VerificationReceipt):
        return abstain("receipt_invalid")
    receipt_strings = (
        receipt.verifier_id, receipt.verifier_version, receipt.decision,
        receipt.evidence_digest, receipt.source_revision, receipt.schema,
        receipt.integrity_digest,
    )
    if (
        len(receipt.source_refs) != 1
        or any(not isinstance(value, str) or len(value) > 512 for value in receipt_strings)
        or receipt.source_refs != (answer.source_ref,)
        or not receipt.is_valid()
    ):
        return abstain("receipt_invalid")
    if (
        not receipt.observed or not receipt.source_backed or not receipt.verified
        or receipt.contradicted or receipt.abstained
        or receipt.source_revision != answer.source_revision
        or receipt.decision != "verified_answer_binding"
    ):
        return abstain("receipt_not_authorizing_answer")
    if receipt.evidence_digest != evidence_digest(answer.evidence_payload()):
        return abstain("answer_digest_mismatch")
    return AnswerResolution("answer", answer.text, answer.source_ref, answer.source_revision)
