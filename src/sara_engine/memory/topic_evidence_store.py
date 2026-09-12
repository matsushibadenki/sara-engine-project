"""Bounded authoritative local scope for the restricted question path.

Trusted verifiers publish complete snapshots, not search results. Generation
checks prevent lost updates; they do not authenticate external source freshness.
"""

from dataclasses import dataclass
from threading import RLock

from sara_engine.memory.structured_query import (
    QueryResolution, TopicEvidence, TopicQuery, resolve_complete_topic_query,
    resolve_topic_query,
)
from sara_engine.memory.topic_question import parse_topic_question


@dataclass(frozen=True)
class PublishResult:
    decision: str
    generation: int


@dataclass(frozen=True)
class StoreAnswer:
    generation: int
    result: QueryResolution


class TopicEvidenceStore:
    """Own at most twelve verified records; never truncate or silently evict.

    Publication is an explicit complete-scope replacement by a trusted producer.
    Invalid current-generation publications invalidate availability, preventing
    an old snapshot from answering after a known failed refresh. Stale writes
    are rejected without invalidating a newer generation. No disk I/O occurs.
    Snapshot freshness defaults to eight caller-defined logical segments;
    callers must advance a trusted monotonic clock and report fetch failures.
    Republishing stale remote content cannot be detected by this local lease.
    """

    def __init__(self, *, max_age_segments: int = 8):
        if type(max_age_segments) is not int or not 1 <= max_age_segments <= 65536:
            raise ValueError("Snapshot age must be between 1 and 65536 segments")
        self._lock = RLock()
        self._generation = 0
        self._records = ()
        self._available = False
        self._max_age_segments = max_age_segments
        self._expires_at = None
        self._last_segment = None

    @property
    def generation(self):
        with self._lock:
            return self._generation

    def publish(self, records: tuple[TopicEvidence, ...], *, expected_generation: int, now_segment: int,
                valid_until_segment: int | None = None, before_commit=None) -> PublishResult:
        with self._lock:
            if type(expected_generation) is not int or expected_generation != self._generation:
                return PublishResult("stale_generation", self._generation)
            decision = self._check_time(now_segment)
            if decision is None and valid_until_segment is not None:
                if type(valid_until_segment) is not int:
                    decision = "invalid_deadline"
                elif valid_until_segment <= now_segment:
                    decision = "snapshot_expired"
            if decision is None:
                decision = self._validate(records, now_segment)
            if decision == "published" and before_commit is not None:
                if not callable(before_commit):
                    decision = "invalid_commit_callback"
                else:
                    try:
                        commit_decision = before_commit()
                    except Exception:
                        commit_decision = "publication_unavailable"
                    if commit_decision is not None:
                        decision = commit_decision
            self._generation += 1
            if decision != "published":
                self._records = ()
                self._available = False
                self._expires_at = None
                return PublishResult(decision, self._generation)
            self._records = records
            self._available = True
            self._expires_at = now_segment + self._max_age_segments
            if valid_until_segment is not None:
                self._expires_at = min(self._expires_at, valid_until_segment)
            return PublishResult("published", self._generation)

    def _check_time(self, now_segment):
        if type(now_segment) is not int:
            return "invalid_time"
        if self._last_segment is not None and now_segment < self._last_segment:
            return "time_regression"
        self._last_segment = now_segment
        return None

    def refresh_failed(self, *, expected_generation: int) -> PublishResult:
        """Explicitly revoke a failed fetch without manufacturing invalid records."""
        with self._lock:
            if type(expected_generation) is not int or expected_generation != self._generation:
                return PublishResult("stale_generation", self._generation)
            self._generation += 1
            self._records = ()
            self._available = False
            self._expires_at = None
            return PublishResult("refresh_failed", self._generation)

    @staticmethod
    def _validate(records, now_segment):
        if type(now_segment) is not int:
            return "invalid_time"
        if not isinstance(records, tuple) or len(records) > 12:
            return "evidence_limit"
        identities = set()
        for item in records:
            if not isinstance(item, TopicEvidence):
                return "invalid_evidence"
            # Resolve each record independently so cross-source contradictions
            # are retained and visible to query-time conflict checks.
            language = getattr(item.answer, "language", None)
            checked = resolve_topic_query(TopicQuery((item.topic_id,), language=language), (item,), now_segment=now_segment)
            if checked.decision != "answer":
                return checked.decision
            identity = (item.topic_id, item.answer.source_ref, language)
            if identity in identities:
                return "duplicate_source"
            identities.add(identity)
        return "published"

    def answer(self, query: TopicQuery, *, now_segment: int) -> StoreAnswer:
        with self._lock:
            failure = self._check_time(now_segment)
            if failure is not None:
                return StoreAnswer(self._generation, QueryResolution(failure))
            if self._available and now_segment >= self._expires_at:
                self._records = ()
                self._available = False
                self._expires_at = None
                return StoreAnswer(self._generation, QueryResolution("snapshot_expired"))
            if not self._available:
                return StoreAnswer(self._generation, QueryResolution("evidence_unavailable"))
            # Hold the lock through the bounded read: publication cannot mix
            # generations. Filter language, but never filter conflicting sources.
            language = getattr(query, "language", None)
            records = tuple(item for item in self._records if item.answer.language == language)
            result = resolve_complete_topic_query(query, iter(records), now_segment=now_segment)
            return StoreAnswer(self._generation, result)

    def answer_question(self, text: str, *, language: str, aliases: tuple[tuple[str, str], ...], now_segment: int) -> StoreAnswer:
        parsed = parse_topic_question(text, language=language, aliases=aliases)
        if parsed.query is None:
            return StoreAnswer(self.generation, QueryResolution(parsed.decision))
        return self.answer(parsed.query, now_segment=now_segment)
