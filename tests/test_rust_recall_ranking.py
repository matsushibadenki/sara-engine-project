"""Release-compatible Rust recall must preserve stable prefix ranking."""
import random
import pytest


@pytest.mark.parametrize('threshold', [0.0, 0.1, 1.0])
def test_rust_recall_matches_full_stable_reference(threshold):
    core = pytest.importorskip('sara_engine.sara_rust_core')
    memory = core.ScalableSDRMemory(threshold)
    rng = random.Random(42)
    records = [(1000-i, rng.sample(range(32), 8)) for i in range(500)]
    for identifier, signature in records:
        memory.add_memory(identifier, signature)
    query = [0, 1, 2, 3, 3]
    query_set = set(query)
    reference = [(identifier, len(query_set.intersection(signature))/len(query_set))
                 for identifier, signature in records]
    reference = [(identifier, score) for identifier, score in reference if score >= threshold]
    reference.sort(key=lambda pair: pair[1], reverse=True)
    for limit in (0, 1, 5, 100, 500, 1000):
        assert memory.search(query, limit) == reference[:limit]
    assert memory.search([], 5) == []
    assert memory.memory_count() == 500


def test_rust_recall_ties_follow_insertion_order_and_clear():
    core = pytest.importorskip('sara_engine.sara_rust_core')
    memory = core.ScalableSDRMemory(0.0)
    for identifier in (90, 10, 80, 20, 70):
        memory.add_memory(identifier, [1, 1, 2])
    assert memory.search([1, 2], 3) == [(90, 1.0), (10, 1.0), (80, 1.0)]
    memory.clear()
    assert memory.memory_count() == 0
    assert memory.search([1], 1) == []
