"""Preserve ranking semantics and prevent repeated refractory-history scans."""
import random
import math
from unittest.mock import patch

from sara_engine.inference import SaraInference


class SinglePassHistory(list):
    def __iter__(self):
        self.iterations = getattr(self, "iterations", 0) + 1
        assert self.iterations == 1
        return super().__iter__()

    def count(self, value):
        raise AssertionError("History must not be rescanned per candidate")


def test_ranking_reads_history_once():
    engine = SaraInference.__new__(SaraInference)
    engine.direct_map = {(1,): {i: 1.0 for i in range(1000)}}
    engine.refractory_buffer = SinglePassHistory([1, 1, 2])
    result = engine._rank_next_token_candidates((1,), 2.0)
    assert result[-2:] == [(2, 0.5), (1, 0.25)]
    assert engine.refractory_buffer.iterations == 1


def test_ranking_matches_legacy_with_ties_and_duplicate_history():
    rng = random.Random(42)
    engine = SaraInference.__new__(SaraInference)
    for size in (0, 1, 100, 1000):
        weights = {i: rng.choice([0.0, 0.5, 1.0, 2.0]) for i in range(size)}
        engine.direct_map = {(1,): weights}
        for length in (0, 16, 128):
            history = [rng.randrange(max(size, 1)) for _ in range(length)]
            engine.refractory_buffer = history
            for penalty in (0.5, 1.0, 1.2, 2.0):
                expected = [(cid, float(w) / penalty ** history.count(cid)
                             if cid in history else float(w))
                            for cid, w in weights.items()]
                expected.sort(key=lambda item: item[1], reverse=True)
                assert engine._rank_next_token_candidates((1,), penalty) == expected
                assert engine._rank_next_token_candidates((1,), penalty, []) == sorted(
                    weights.items(), key=lambda item: item[1], reverse=True)
            assert engine.refractory_buffer == history


def test_ranking_observes_history_updates_without_stale_cache():
    engine = SaraInference.__new__(SaraInference)
    engine.direct_map = {(1,): {1: 1.0, 2: 1.0}}
    engine.refractory_buffer = [1]
    assert engine._rank_next_token_candidates((1,), 2.0)[0][0] == 2
    engine.refractory_buffer[:] = [2]
    assert engine._rank_next_token_candidates((1,), 2.0)[0][0] == 1


def test_bounded_ranking_matches_full_stable_order():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = [1, 1, 2]
    rng = random.Random(17)
    for size in (0, 1, 10, 1000, 3000):
        engine.direct_map = {(1,): {i: rng.choice([0.0, 0.5, 1.0, float('inf')])
                                  for i in range(size)}}
        expected = engine._rank_next_token_candidates((1,), 1.2)
        for limit in (1, 3, 5, size + 1):
            assert engine._rank_next_token_candidates((1,), 1.2, limit=limit) == expected[:limit]


def test_nan_ranking_preserves_legacy_sort_order():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    for weights in ([1.0, float('nan'), 2.0], [float('nan'), 2.0, 1.0]):
        engine.direct_map = {(1,): dict(enumerate(weights + [0.0] * 3000))}
        expected = engine._rank_next_token_candidates((1,), 1.2)
        for limit in (1, 2, 3):
            actual = engine._rank_next_token_candidates((1,), 1.2, limit=limit)
            assert [cid for cid, _ in actual] == [cid for cid, _ in expected[:limit]]


def test_sampling_uses_bounded_selection_without_full_sort():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = []
    engine.direct_map = {(1,): {i: float(i) for i in range(3000)}}
    original = engine._rank_next_token_candidates
    with patch.object(engine, '_rank_next_token_candidates', wraps=original) as rank:
        engine._sample_next_token((1,), 3, 0.5, 1.2)
        assert rank.call_args.kwargs['limit'] == 3
        assert engine._sample_next_token((1,), 3, 0.0, 1.2) == 2999
        assert rank.call_args.kwargs['limit'] == 1

    import heapq
    with patch('sara_engine.inference.heapq.nlargest', wraps=heapq.nlargest) as select:
        engine._sample_next_token((1,), 3, 0.5, 1.2)
        select.assert_called_once()
    engine.direct_map = {(1,): {1: 1.0, 2: 2.0}}
    with patch('sara_engine.inference.heapq.nlargest') as select:
        engine._sample_next_token((1,), 3, 0.5, 1.2)
        select.assert_not_called()


def test_seeded_sampling_matches_full_ranking_reference():
    engine = SaraInference.__new__(SaraInference)
    engine.refractory_buffer = [1, 1, 2]
    engine.direct_map = {(1,): {i: (i % 7) / 7 for i in range(100)}}
    for k in (1, 3, 100, 200, -1):
        for temperature in (0.0, 0.5, 1.0):
            ranked = engine._rank_next_token_candidates((1,), 1.2)
            for seed in range(20):
                random.seed(seed)
                if temperature <= 0.01 or k == 1:
                    expected = ranked[0][0]
                else:
                    top = ranked[:k]
                    probs = [math.pow(w, 1.0 / temperature) for _, w in top]
                    total = sum(probs)
                    target = random.random()
                    cumulative = 0.0
                    expected = top[-1][0]
                    for (cid, _), probability in zip(top, probs):
                        cumulative += probability / total
                        if target <= cumulative:
                            expected = cid
                            break
                random.seed(seed)
                assert engine._sample_next_token((1,), k, temperature, 1.2) == expected
