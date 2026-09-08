"""Bounded, read-only resolution of explicit topics from a trusted producer.

This boundary does not parse language or establish semantic topic assignments.
The producer must supply a complete bounded evidence snapshot, including
conflicting sources. Integrity receipts do not authenticate that producer.
"""

from dataclasses import dataclass

from sara_engine.memory.event_state_cache import EventStateEntry
from sara_engine.memory.verification_receipt import VerificationReceipt, evidence_digest
from sara_engine.memory.verified_answer import AnswerResolution, VerifiedAnswer, resolve_verified_answer


@dataclass(frozen=True)
class TopicQuery:
    requested: tuple[str, ...]
    excluded: tuple[str, ...] = ()
    language: str = "en"
    unresolved: bool = False


@dataclass(frozen=True)
class TopicEvidence:
    topic_id: str
    entry: EventStateEntry
    answer: VerifiedAnswer
    receipt: VerificationReceipt

    def evidence_payload(self) -> dict:
        return {
            "schema": "sara-topic-answer-binding-v1",
            "topic_id": self.topic_id,
            "answer": self.answer.evidence_payload(),
        }


@dataclass(frozen=True)
class TopicAnswer:
    topic_id: str
    evidence: tuple[AnswerResolution, ...]


@dataclass(frozen=True)
class QueryResolution:
    decision: str
    items: tuple[TopicAnswer, ...] = ()


def _valid_topic(value) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 128 and value == value.strip()


def _valid_topics(values) -> bool:
    return (
        isinstance(values, tuple) and len(values) <= 4
        and all(_valid_topic(value) for value in values)
        and len(set(values)) == len(values)
    )


def _valid_binding(item: TopicEvidence) -> bool:
    receipt = item.receipt
    if not isinstance(receipt, VerificationReceipt):
        return False
    fields = (
        receipt.verifier_id, receipt.verifier_version, receipt.decision,
        receipt.evidence_digest, receipt.source_revision, receipt.schema,
        receipt.integrity_digest,
    )
    if (
        not isinstance(receipt.source_refs, tuple) or len(receipt.source_refs) != 1
        or any(not isinstance(value, str) or len(value) > 512 for value in fields)
        or receipt.source_refs != (item.answer.source_ref,)
    ):
        return False
    return (
        receipt.is_valid() and receipt.observed and receipt.source_backed
        and receipt.verified and not receipt.contradicted and not receipt.abstained
        and receipt.decision == "verified_topic_answer_binding"
        and receipt.source_revision == item.answer.source_revision
        and receipt.evidence_digest == evidence_digest(item.evidence_payload())
    )


def resolve_topic_query(
    query: TopicQuery, evidence: tuple[TopicEvidence, ...], *, now_segment: int,
) -> QueryResolution:
    """Return complete coverage or no items; inspect at most twelve records.

    Exact topic IDs avoid question-wide signature truncation. They require a
    separately validated structured producer; arbitrary chat cannot use this
    function as evidence that its topic interpretation was correct.
    """
    if (
        not isinstance(query, TopicQuery)
        or not _valid_topics(query.requested) or not query.requested
        or not _valid_topics(query.excluded)
        or set(query.requested).intersection(query.excluded)
        or not isinstance(query.language, str) or query.language not in ("en", "ja", "zh-CN")
        or type(query.unresolved) is not bool
    ):
        return QueryResolution("invalid_query")
    if type(now_segment) is not int:
        return QueryResolution("invalid_time")
    if query.unresolved:
        return QueryResolution("unresolved_reference")
    if not isinstance(evidence, tuple) or len(evidence) > 12:
        return QueryResolution("evidence_limit")
    if any(not isinstance(item, TopicEvidence) or not _valid_topic(item.topic_id) for item in evidence):
        return QueryResolution("invalid_evidence")

    grouped = {topic: [] for topic in query.requested}
    for item in evidence:
        if item.topic_id not in grouped:
            continue
        if not isinstance(item.entry, EventStateEntry) or not isinstance(item.answer, VerifiedAnswer):
            return QueryResolution("invalid_evidence")
        if not isinstance(item.answer.receipt, VerificationReceipt) or not isinstance(item.answer.receipt.source_refs, tuple):
            return QueryResolution("receipt_invalid")
        answer = resolve_verified_answer(item.entry, item.answer, now_segment=now_segment, language=query.language)
        if answer.decision != "answer":
            return QueryResolution(answer.decision)
        if not _valid_binding(item):
            return QueryResolution("invalid_topic_binding")
        grouped[item.topic_id].append(answer)

    if any(not values for values in grouped.values()):
        return QueryResolution("incomplete_coverage")
    if any(len({answer.text for answer in values}) != 1 for values in grouped.values()):
        return QueryResolution("conflicting_evidence")
    items = tuple(
        TopicAnswer(topic, tuple(sorted(set(values), key=lambda value: (value.source_ref, value.source_revision, value.text))))
        for topic, values in grouped.items()
    )
    return QueryResolution("answer", items)
