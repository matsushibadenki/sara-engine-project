import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.inference import SaraInference
from sara_engine.utils.project_paths import model_path


def test_inference_save_and_load_memory_roundtrip():
    memory_path = model_path("tests", "sara_inference_memory.msgpack")
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)

    writer = SaraInference.__new__(SaraInference)
    writer.model_path = memory_path
    writer.direct_map = {
        (12345,): {7: 1.0, 8: 2.0},
        (67890,): {9: 3.5},
    }
    writer.context_index = {
        (12345,): (1, 2, 3),
        (67890,): (4, 5),
    }
    writer.refractory_buffer = []
    writer.lif_network = None
    writer.save_pretrained(memory_path)

    reader = SaraInference.__new__(SaraInference)
    reader.model_path = memory_path
    reader.direct_map = {}
    reader.context_index = {}
    reader.refractory_buffer = []
    reader.lif_network = None
    reader._load_memory()

    assert reader.direct_map == writer.direct_map
    assert reader.context_index == writer.context_index
