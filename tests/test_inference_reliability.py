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
    engine.refractory_buffer = []
    engine.lif_network = None

    engine.learn_sequence([10, 11, 12])

    assert engine.direct_map
    for key, values in engine.direct_map.items():
        assert isinstance(key, tuple)
        assert all(isinstance(item, int) for item in key)
        assert all(isinstance(token_id, int) for token_id in values.keys())
