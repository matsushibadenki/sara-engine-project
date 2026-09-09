import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from sara_engine.memory.structured_query import TopicQuery
from sara_engine.memory.topic_evidence_store import TopicEvidenceStore
from sara_engine.memory.topic_evidence_pages import EvidencePage, refresh_from_pages

SPEC = importlib.util.spec_from_file_location("page_fixtures", Path(__file__).with_name("test_topic_evidence_store.py"))
FIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIX)


@pytest.fixture
def setup():
    records = FIX.FIXTURES.build_evidence(FIX.PROTOCOL["languages"][0], FIX.PROTOCOL["topics"])
    store = TopicEvidenceStore()
    store.publish(records, expected_generation=0, now_segment=3)
    pages = tuple(EvidencePage("sensors", "snapshot:2", index, 2, 2, (record,)) for index, record in enumerate(records))
    return store, pages, TopicQuery((records[0].topic_id,))


def refresh(store, fetch):
    return refresh_from_pages(store, fetch, scope="sensors", expected_generation=1, now_segment=4)


def test_complete_pages_publish_once(setup):
    store, pages, query = setup
    reads = []
    def fetch(index):
        reads.append(index)
        assert store.generation == 1
        return pages[index]
    assert refresh(store, fetch).decision == "published"
    assert reads == [0, 1]
    assert store.answer(query, now_segment=4).generation == 2
    assert store.answer(query, now_segment=4).result.decision == "answer"


@pytest.mark.parametrize("mutation", [
    {"snapshot": "different"}, {"scope": "other"}, {"index": 0},
    {"total_pages": 3}, {"total_records": 3}, {"records": ()},
])
def test_mixed_or_incomplete_pages_revoke_old_snapshot(setup, mutation):
    store, pages, query = setup
    altered = (pages[0], replace(pages[1], **mutation))
    assert refresh(store, altered.__getitem__).decision != "published"
    assert store.answer(query, now_segment=4).result.decision == "evidence_unavailable"


def test_page_failure_after_valid_prefix(setup):
    store, pages, query = setup
    def fetch(index):
        if index == 0:
            return pages[0]
        raise OSError("Fetch failed")
    assert refresh(store, fetch).decision == "page_unavailable"
    assert not store.answer(query, now_segment=4).result.items


def test_late_fetch_failure_does_not_revoke_new_publication(setup):
    store, pages, query = setup
    def fetch(index):
        store.publish(tuple(page.records[0] for page in pages), expected_generation=1, now_segment=4)
        raise OSError("Old request failed")
    assert refresh(store, fetch).decision == "stale_generation"
    assert store.answer(query, now_segment=4).result.decision == "answer"


def test_page_bound_rejected_before_second_fetch(setup):
    store, pages, _ = setup
    calls = []
    def fetch(index):
        calls.append(index)
        return replace(pages[0], total_pages=13)
    assert refresh(store, fetch).decision == "invalid_page"
    assert calls == [0]


def test_stale_generation_does_not_fetch(setup):
    store, _, _ = setup
    result = refresh_from_pages(store, lambda _: pytest.fail("Unexpected fetch"), scope="sensors", expected_generation=0, now_segment=4)
    assert result.decision == "stale_generation"


def test_fetch_time_does_not_extend_source_deadline(setup):
    store, pages, query = setup
    timed = tuple(replace(page, valid_until_segment=7) for page in pages)
    samples = iter((4, 5, 5, 6))
    result = refresh_from_pages(store, timed.__getitem__, scope="sensors", expected_generation=1,
                                now_segment=4, clock=lambda: next(samples))
    assert result.decision == "published"
    assert store.answer(query, now_segment=6).result.decision == "answer"
    assert store.answer(query, now_segment=7).result.decision == "snapshot_expired"


@pytest.mark.parametrize("samples,deadline,decision", [
    ((4, 5, 5, 6), 6, "snapshot_expired"),
    ((4, 12), None, "fetch_deadline"),
    ((4, 3), None, "time_regression"),
    ((4, True), None, "invalid_time"),
])
def test_failed_timed_fetch_never_publishes(setup, samples, deadline, decision):
    store, pages, query = setup
    timed = tuple(replace(page, valid_until_segment=deadline) for page in pages)
    ticks = iter(samples)
    result = refresh_from_pages(store, timed.__getitem__, scope="sensors", expected_generation=1,
                                now_segment=4, clock=lambda: next(ticks))
    assert result.decision == decision
    assert store.answer(query, now_segment=12).result.decision == "evidence_unavailable"


def test_deadline_mismatch_between_pages(setup):
    store, pages, _ = setup
    timed = (replace(pages[0], valid_until_segment=8), replace(pages[1], valid_until_segment=9))
    assert refresh(store, timed.__getitem__).decision == "snapshot_mismatch"
