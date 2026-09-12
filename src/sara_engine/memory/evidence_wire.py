"""Strict V1/V2 JSON decoding for evidence publishers; never issue receipts."""

from dataclasses import dataclass, fields
import math

from sara_engine.memory.event_state_cache import EventStateEntry
from sara_engine.memory.verification_receipt import VerificationReceipt
from sara_engine.memory.verified_answer import VerifiedAnswer
from sara_engine.memory.structured_query import TopicEvidence, TopicQuery, resolve_topic_query
from sara_engine.memory.topic_evidence_pages import EvidencePage


@dataclass(frozen=True)
class AuthoritativeEvidencePage:
    page: EvidencePage
    publisher_id: str
    snapshot_sequence: int
    issued_at_epoch: int


class AuthoritativeEvidenceError(ValueError):
    def __init__(self, decision):
        self.decision = decision
        super().__init__(decision)


def _object(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("Unexpected evidence object fields")


def _text(value, limit=512):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError("Invalid evidence string")


def _receipt(value):
    _object(value, (field.name for field in fields(VerificationReceipt)))
    for key, item in value.items():
        if key in ("observed", "source_backed", "verified", "contradicted", "abstained"):
            if type(item) is not bool:
                raise ValueError("Invalid receipt flag")
        elif key == "source_refs":
            if not isinstance(item, list) or len(item) != 1:
                raise ValueError("Invalid receipt sources")
            _text(item[0])
        else:
            _text(item)
    return VerificationReceipt(**{**value, "source_refs": tuple(value["source_refs"])})


def _entry(value):
    _object(value, (field.name for field in fields(EventStateEntry)))
    integers = {"time_segment", "sequence_support_count", "event_cost", "access_count"}
    scalars = {"confidence", "uncertainty", "source_reliability", "resonance_score", "sequence_support_score",
               "credit_score", "credit_responsibility", "credit_confidence", "credit_longevity", "utility"}
    for key, item in value.items():
        if key in integers or key == "expires_at":
            if key == "expires_at" and item is None:
                continue
            if type(item) is not int or not 0 <= item <= 2**63 - 1:
                raise ValueError("Invalid entry integer")
        elif key in scalars:
            if type(item) not in (int, float) or not math.isfinite(item) or not 0 <= item <= 1:
                raise ValueError("Invalid entry scalar")
        elif key in ("observed", "verified"):
            if type(item) is not bool:
                raise ValueError("Invalid entry flag")
        elif key == "signature":
            if not isinstance(item, list) or len(item) > 64 or any(type(bit) is not int or not 0 <= bit < 2**64 for bit in item):
                raise ValueError("Invalid signature")
        elif key == "causal_predecessors":
            if not isinstance(item, list) or len(item) > 12:
                raise ValueError("Invalid predecessors")
            for predecessor in item:
                _text(predecessor)
        else:
            _text(item)
    return EventStateEntry(**{**value, "signature": tuple(value["signature"]), "causal_predecessors": tuple(value["causal_predecessors"])})


def decode_evidence_page(payload, *, now_segment):
    """Decode the bounded wire subset, then verify every topic/answer binding.

    Receipt integrity is not source authentication. The trusted connection must
    establish source authority separately. No coercion of strings to flags/IDs.
    """
    _object(payload, ("schema", "scope", "snapshot", "index", "total_pages", "total_records", "valid_until_segment", "records"))
    if payload["schema"] != "sara-evidence-page-v1":
        raise ValueError("Unsupported evidence schema")
    for key in ("scope", "snapshot"):
        _text(payload[key], 128)
        if not payload[key].strip():
            raise ValueError("Empty page identity")
    for key, low, high in (("index", 0, 11), ("total_pages", 1, 12), ("total_records", 0, 12)):
        if type(payload[key]) is not int or not low <= payload[key] <= high:
            raise ValueError("Invalid page count")
    deadline = payload["valid_until_segment"]
    if type(now_segment) is not int or (deadline is not None and (type(deadline) is not int or deadline <= now_segment)):
        raise ValueError("Invalid or expired page time")
    rows = payload["records"]
    if not isinstance(rows, list) or len(rows) > payload["total_records"]:
        raise ValueError("Invalid record count")
    records = []
    for row in rows:
        _object(row, ("topic_id", "entry", "answer", "receipt"))
        _text(row["topic_id"], 128)
        answer = row["answer"]
        _object(answer, (field.name for field in fields(VerifiedAnswer)))
        for key in ("entry_id", "language", "source_ref", "source_revision", "text"):
            _text(answer[key], 4096 if key == "text" else 512)
        item = TopicEvidence(row["topic_id"], _entry(row["entry"]),
                             VerifiedAnswer(**{**answer, "receipt": _receipt(answer["receipt"])}), _receipt(row["receipt"]))
        checked = resolve_topic_query(TopicQuery((item.topic_id,), language=item.answer.language), (item,), now_segment=now_segment)
        if checked.decision != "answer":
            raise ValueError("Invalid evidence binding: " + checked.decision)
        records.append(item)
    return EvidencePage(payload["scope"], payload["snapshot"], payload["index"], payload["total_pages"],
                        payload["total_records"], tuple(records), deadline)


def decode_authoritative_evidence_page(
    payload, *, expected_publisher, expected_scope, now_epoch,
    max_future_skew_seconds=30, max_snapshot_lifetime_seconds=3600,
):
    """Decode V2 publisher/time metadata and the existing verified records."""
    try:
        _object(payload, (
            "schema", "publisher_id", "scope", "snapshot", "snapshot_sequence",
            "issued_at_epoch", "expires_at_epoch", "index", "total_pages",
            "total_records", "records",
        ))
        if payload["schema"] != "sara-evidence-page-v2":
            raise AuthoritativeEvidenceError("invalid_page")
        for value in (expected_publisher, expected_scope):
            _text(value, 128)
            if not value or value != value.strip():
                raise AuthoritativeEvidenceError("invalid_page")
        if payload["publisher_id"] != expected_publisher:
            raise AuthoritativeEvidenceError("publisher_mismatch")
        if payload["scope"] != expected_scope:
            raise AuthoritativeEvidenceError("scope_mismatch")
        sequence = payload["snapshot_sequence"]
        issued = payload["issued_at_epoch"]
        expires = payload["expires_at_epoch"]
        limits = (max_future_skew_seconds, max_snapshot_lifetime_seconds)
        if (
            type(now_epoch) is not int or now_epoch < 0
            or any(type(value) is not int or not 0 <= value <= 86400 for value in limits)
            or type(sequence) is not int or not 0 <= sequence <= 2**63 - 1
            or type(issued) is not int or type(expires) is not int
            or not 0 <= issued < expires <= 2**63 - 1
            or issued > now_epoch + max_future_skew_seconds
            or expires - issued > max_snapshot_lifetime_seconds
        ):
            raise AuthoritativeEvidenceError("publisher_time_invalid")
        if expires <= now_epoch:
            raise AuthoritativeEvidenceError("snapshot_expired")
        v1 = {
            "schema": "sara-evidence-page-v1", "scope": payload["scope"],
            "snapshot": payload["snapshot"], "index": payload["index"],
            "total_pages": payload["total_pages"], "total_records": payload["total_records"],
            "valid_until_segment": expires, "records": payload["records"],
        }
        page = decode_evidence_page(v1, now_segment=now_epoch)
        return AuthoritativeEvidencePage(page, payload["publisher_id"], sequence, issued)
    except AuthoritativeEvidenceError:
        raise
    except (KeyError, TypeError, ValueError):
        raise AuthoritativeEvidenceError("invalid_page") from None
