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


def test_spiking_llm_save_and_load_memory_roundtrip_with_turboquant():
    model_file = model_path("tests", "spiking_llm_memory_turboquant.msgpack")
    os.makedirs(os.path.dirname(model_file), exist_ok=True)

    original = SpikingLLM(
        num_layers=1,
        sdr_size=64,
        vocab_size=512,
        enable_turboquant=True,
        turboquant_main_bits=3,
    )
    original._direct_map = {
        (1, 3, 5): {10: 1.5, 11: 2.0, 12: 3.0},
        (2, 4): {12: 0.75, 13: 1.75},
    }
    original.save_memory(model_file)

    restored = SpikingLLM(num_layers=1, sdr_size=64, vocab_size=512)
    loaded_count = restored.load_memory(model_file)

    assert loaded_count == 2
    assert set(restored._direct_map.keys()) == set(original._direct_map.keys())
    for key, values in original._direct_map.items():
        restored_values = restored._direct_map[key]
        assert set(restored_values.keys()) == set(values.keys())
        for token_id, weight in values.items():
            assert abs(restored_values[token_id] - weight) < 1.5


def test_spiking_llm_save_and_load_pretrained_with_turboquant():
    save_dir = model_path("tests", "spiking_llm_turboquant")
    os.makedirs(save_dir, exist_ok=True)

    model = SpikingLLM(
        num_layers=1,
        sdr_size=8,
        vocab_size=64,
        context_window=4,
        enable_turboquant=True,
        turboquant_main_bits=3,
    )
    model.tokenizer.model_path = os.path.join(save_dir, "sara_vocab.json")
    model.tokenizer._add_token("猫")
    model.tokenizer._add_token("走る")
    model.pretrained_synapses = {
        1: {1: {2: 1.0, 3: 2.0}},
        2: {2: {4: 0.75}},
    }
    model._direct_map = {
        (1, 2): {3: 1.5, 4: 2.5},
    }
    model.lm_head_w[0] = {5: 1.2, 6: 2.4}
    model.lm_head_w[1] = {7: 0.8}

    model.save_pretrained(save_dir)
    restored = SpikingLLM.from_pretrained(save_dir)

    assert restored.quantization_enabled is True
    assert set(restored.pretrained_synapses.keys()) == set(model.pretrained_synapses.keys())
    assert set(restored._direct_map.keys()) == set(model._direct_map.keys())
    assert set(restored.lm_head_w[0].keys()) == set(model.lm_head_w[0].keys())
    for token_id, weight in model.lm_head_w[0].items():
        assert abs(restored.lm_head_w[0][token_id] - weight) < 1.5


def test_spiking_llm_direct_map_index_and_lru_limit_are_rebuilt():
    model = SpikingLLM(
        num_layers=1,
        sdr_size=64,
        vocab_size=256,
        max_direct_map_patterns=2,
    )
    model._direct_map = {
        (1, 2): {10: 1.0},
        (2, 3): {11: 1.0},
        (50, 51): {12: 1.0},
    }

    model._rebuild_direct_map_index()

    assert len(model._direct_map) == 2
    assert (1, 2) not in model._direct_map
    assert (2, 3) in model._direct_map
    assert (50, 51) in model._direct_map
    recalled, score = model.recall((2, 3), threshold=0.5)
    assert recalled == {11: 1.0}
    assert score == 1.0
