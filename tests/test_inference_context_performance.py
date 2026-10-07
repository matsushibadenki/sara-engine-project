"""Verify exact alignment metrics without temporary union/difference sets."""
import random
from unittest.mock import patch

from sara_engine.inference import SaraInference


def reference_alignment(query, candidate):
    query = tuple(int(token) for token in query)
    candidate = tuple(int(token) for token in candidate)
    if not query or not candidate:
        return dict.fromkeys(('overlap', 'coverage', 'precision', 'jaccard',
                              'length_ratio', 'suffix_match', 'drift_penalty',
                              'stability_score'), 0.0)
    query_set, candidate_set = set(query), set(candidate)
    overlap = len(query_set.intersection(candidate_set))
    coverage = overlap / max(1, len(query_set))
    precision = overlap / max(1, len(candidate_set))
    jaccard = overlap / max(1, len(query_set.union(candidate_set)))
    length_ratio = min(len(query), len(candidate)) / max(1, max(len(query), len(candidate)))
    suffix = 0
    for left, right in zip(reversed(query), reversed(candidate)):
        if left != right:
            break
        suffix += 1
    suffix_match = suffix / max(1, min(len(query), len(candidate)))
    drift = min(0.4, len(candidate_set.difference(query_set)) / max(1, len(candidate_set)))
    score = coverage * .35 + precision * .15 + jaccard * .15 + length_ratio * .10 + suffix_match * .25 - drift * .15
    return dict(overlap=float(overlap), coverage=coverage, precision=precision,
                jaccard=jaccard, length_ratio=length_ratio, suffix_match=suffix_match,
                drift_penalty=drift, stability_score=max(0.0, min(1.0, score)))


def test_alignment_exactly_matches_legacy_set_arithmetic():
    engine = SaraInference.__new__(SaraInference)
    rng = random.Random(42)
    for _ in range(1000):
        query = [rng.randrange(-5, 16) for _ in range(rng.randrange(17))]
        candidate = [rng.randrange(-5, 16) for _ in range(rng.randrange(17))]
        assert engine._score_direct_context_alignment(query, candidate) == reference_alignment(query, candidate)
    assert engine._score_direct_context_alignment(['1', '2'], ['2', '3']) == reference_alignment(['1', '2'], ['2', '3'])


def test_alignment_does_not_allocate_union_or_difference_sets():
    class NoTemporarySets(set):
        def union(self, other):
            raise AssertionError('Union must be derived from overlap')

        def difference(self, other):
            raise AssertionError('Difference must be derived from overlap')

    engine = SaraInference.__new__(SaraInference)
    with patch('sara_engine.inference.set', NoTemporarySets, create=True):
        assert engine._score_direct_context_alignment([1, 2], [2, 3]) == reference_alignment([1, 2], [2, 3])


def test_retrieval_and_diagnostics_match_reference_scoring():
    from sara_engine.utils.tokenizer import SaraTokenizer
    tokenizer = SaraTokenizer()
    texts = ('The cat eats food and the dog drinks water.',
             '猫はご飯を食べます。犬は水を飲みます。',
             '猫吃食物，狗喝水。')
    for text in texts:
        tokens = tokenizer.encode(text)
        engine = SaraInference.__new__(SaraInference)
        engine.context_encoding = 'stable_v1'
        engine.context_index = {}
        engine.direct_map = {}
        engine.refractory_buffer = []
        engine.retrieval_diagnostics = []
        engine.lif_network = None
        engine.learn_sequence(tokens)
        query = tokens[:4] + [987654]
        actual = engine._find_best_matching_key(query)
        diagnostics = list(engine.retrieval_diagnostics)
        engine.retrieval_diagnostics = []
        with patch.object(engine, '_score_direct_context_alignment', side_effect=reference_alignment):
            expected = engine._find_best_matching_key(query)
        assert actual == expected
        assert engine.retrieval_diagnostics == diagnostics
