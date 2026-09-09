"""Bounded transport-neutral pagination for a trusted evidence publisher."""

from dataclasses import dataclass

from sara_engine.memory.structured_query import TopicEvidence
from sara_engine.memory.topic_evidence_store import PublishResult, TopicEvidenceStore


@dataclass(frozen=True)
class EvidencePage:
    scope: str
    snapshot: str
    index: int
    total_pages: int
    total_records: int
    records: tuple[TopicEvidence, ...]
    valid_until_segment: int | None = None


def refresh_from_pages(store: TopicEvidenceStore, fetch_page, *, scope: str,
                       expected_generation: int, now_segment: int,
                       clock=None, max_fetch_segments: int = 8) -> PublishResult:
    """Publish only a complete, consistent snapshot with at most twelve pages.

    The callback receives zero-based page indices and must enforce transport
    timeouts and response-byte limits. Snapshot IDs and counts are assertions by
    a trusted source, not authentication or proof of external freshness.
    Optional clock samples use the same logical units as source deadlines and
    are checked before/after each fetch. They cannot interrupt a blocked callback.
    Without a clock, now_segment remains a fixed caller-supplied observation.
    """
    if type(expected_generation) is not int or expected_generation != store.generation:
        return PublishResult("stale_generation", store.generation)

    def fail(reason):
        revoked = store.refresh_failed(expected_generation=expected_generation)
        if revoked.decision == "stale_generation":
            return revoked
        return PublishResult(reason, revoked.generation)

    if not isinstance(scope, str) or not 0 < len(scope) <= 128 or scope != scope.strip():
        return fail("invalid_scope")
    if type(now_segment) is not int:
        return fail("invalid_time")
    if type(max_fetch_segments) is not int or not 1 <= max_fetch_segments <= 65536:
        return fail("invalid_fetch_budget")
    current_time = now_segment

    def check_clock():
        nonlocal current_time
        if clock is None:
            return None
        try:
            sampled = clock()
        except Exception:
            return "clock_unavailable"
        if type(sampled) is not int:
            return "invalid_time"
        if sampled < current_time:
            return "time_regression"
        current_time = sampled
        if current_time - now_segment >= max_fetch_segments:
            return "fetch_deadline"
        return None

    records = []
    manifest = None
    for index in range(12):
        clock_failure = check_clock()
        if clock_failure:
            return fail(clock_failure)
        try:
            page = fetch_page(index)
        except Exception:
            return fail("page_unavailable")
        clock_failure = check_clock()
        if clock_failure:
            return fail(clock_failure)
        if not isinstance(page, EvidencePage):
            return fail("invalid_page")
        if (
            page.scope != scope or not isinstance(page.snapshot, str)
            or not 0 < len(page.snapshot) <= 128
            or type(page.index) is not int or page.index != index
            or type(page.total_pages) is not int or not 1 <= page.total_pages <= 12
            or type(page.total_records) is not int or not 0 <= page.total_records <= 12
            or not isinstance(page.records, tuple) or len(page.records) > 12
            or (page.valid_until_segment is not None and type(page.valid_until_segment) is not int)
        ):
            return fail("invalid_page")
        current = (page.snapshot, page.total_pages, page.total_records, page.valid_until_segment)
        if manifest is None:
            manifest = current
        elif current != manifest:
            return fail("snapshot_mismatch")
        if page.valid_until_segment is not None and page.valid_until_segment <= current_time:
            return fail("snapshot_expired")
        if len(records) + len(page.records) > page.total_records:
            return fail("record_count_mismatch")
        records.extend(page.records)
        if index + 1 == page.total_pages:
            if len(records) != page.total_records:
                return fail("record_count_mismatch")
            return store.publish(tuple(records), expected_generation=expected_generation, now_segment=current_time,
                                 valid_until_segment=page.valid_until_segment)
    return fail("page_limit")
