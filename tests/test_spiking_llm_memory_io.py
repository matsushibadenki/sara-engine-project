import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.models.spiking_llm import SpikingLLM
from sara_engine.utils.project_paths import model_path


def test_spiking_llm_save_and_load_memory_roundtrip():
    model_file = model_path("tests", "spiking_llm_memory.msgpack")
    os.makedirs(os.path.dirname(model_file), exist_ok=True)

    original = SpikingLLM(num_layers=1, sdr_size=64, vocab_size=512)
    original._direct_map = {
        (1, 3, 5): {10: 1.5, 11: 2.0},
        (2, 4): {12: 0.75},
    }
    original.save_memory(model_file)

    restored = SpikingLLM(num_layers=1, sdr_size=64, vocab_size=512)
    loaded_count = restored.load_memory(model_file)

    assert loaded_count == 2
    assert restored._direct_map == original._direct_map
