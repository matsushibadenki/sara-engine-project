import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.inference import SaraInference
from sara_engine.utils.project_paths import model_path


def test_inference_missing_memory_file_starts_empty():
    missing_path = model_path("tests", "missing_inference_memory.msgpack")
    if os.path.exists(missing_path):
        os.remove(missing_path)

    engine = SaraInference.__new__(SaraInference)
    engine.model_path = missing_path
    engine.direct_map = {"stale": {"bad": 1.0}}
    engine.refractory_buffer = []
    engine.lif_network = None
    engine._load_memory()

    assert engine.direct_map == {}


def test_inference_learn_sequence_uses_tuple_keys_and_integer_tokens():
    engine = SaraInference.__new__(SaraInference)
    engine.model_path = ""
    engine.direct_map = {}
    engine.context_index = {}
    engine.refractory_buffer = []
    engine.lif_network = None

    engine.learn_sequence([10, 11, 12])

    assert engine.direct_map
    assert engine.context_index
    for key, values in engine.direct_map.items():
        assert isinstance(key, tuple)
        assert all(isinstance(item, int) for item in key)
        assert all(isinstance(token_id, int) for token_id in values.keys())


def test_inference_fuzzy_context_match_recovers_nearby_sequence():
    engine = SaraInference.__new__(SaraInference)
    engine.model_path = ""
    engine.direct_map = {}
    engine.context_index = {}
    engine.refractory_buffer = []
    engine.lif_network = None

    engine.learn_sequence([101, 102, 103, 999])

    matched_key = engine._find_best_matching_key([102, 103])

    assert matched_key is not None
    assert matched_key in engine.direct_map
    assert 999 in engine.direct_map[matched_key]
