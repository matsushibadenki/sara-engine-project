"""Containment must not calculate a discarded Jaccard score."""
from unittest.mock import patch

from sara_engine.memory.event_state_cache import EventStateCandidate, VerifiedHierarchicalEventStateCache
from sara_engine.memory.verification_receipt import issue_verification_receipt


def populated_cache():
    cache = VerifiedHierarchicalEventStateCache()
    for entry_id, signature in [('contained', (1, 3, 5)), ('other', (100, 101))]:
        source = 'fixture:' + entry_id
        receipt = issue_verification_receipt(
            verifier_id='performance-test', verifier_version='v1',
            decision='verified_fixture', evidence={'signature': list(signature)},
            source_refs=(source,), source_revision='', observed=True, source_backed=True, verified=True)
        result = cache.admit(EventStateCandidate(
            entry_id=entry_id, signature=signature, source_ref=source, time_segment=1,
            observed=True, source_backed=True, verified=True, resonance_score=.9,
            verification_receipt=receipt))
        assert result.accepted
    return cache


def test_containment_skips_jaccard_and_keeps_result_accounting():
    cache = populated_cache()
    with patch('sara_engine.memory.event_state_cache._jaccard', side_effect=AssertionError('Discarded calculation')):
        result = cache.retrieve((1, 1, 3, 5, 7), signature_match='contained')
    assert [match['entry_id'] for match in result.matches] == ['contained']
    assert result.matches[0]['components']['sparse_overlap'] == 1.0
    assert result.scanned_entries == 2
    assert result.event_cost == 9
    assert cache.entries['contained'].access_count == 1
    assert cache.entries['other'].access_count == 0
    assert cache.retrieval_count == 1


def test_containment_miss_keeps_abstention_and_trace():
    cache = populated_cache()
    with patch('sara_engine.memory.event_state_cache._jaccard', side_effect=AssertionError('Discarded calculation')):
        result = cache.retrieve((), signature_match='contained')
    assert result.abstained and not result.matches
    assert result.scanned_entries == 2
    assert result.event_cost == 5
    assert cache.lifecycle_trace[-1]['decision'] == 'abstain_insufficient_evidence'


def test_default_jaccard_still_runs():
    from sara_engine.memory.event_state_cache import _jaccard
    cache = populated_cache()
    with patch('sara_engine.memory.event_state_cache._jaccard', wraps=_jaccard) as score:
        result = cache.retrieve((1, 3, 5))
    assert score.call_count == 2
    assert result.matches[0]['entry_id'] == 'contained'
