import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sara_engine.models.snn_transformer import SNNTransformerConfig, SpikingTransformerModel
from sara_engine.utils.project_paths import model_path
from sara_engine.utils.turboquant import HybridTurboQuantEngine, TurboQuantConfig


def test_turboquant_weight_row_reconstructs_structure():
    engine = HybridTurboQuantEngine(TurboQuantConfig(main_bits=3))
    row = {10: 0.4, 11: 1.2, 12: 2.8}

    payload = engine.quantize_weight_row(row)
    restored = engine.reconstruct_weight_row(payload)

    assert set(restored.keys()) == set(row.keys())
    for token_id, weight in row.items():
        assert abs(restored[token_id] - weight) < 1.5


def test_snn_transformer_save_and_load_with_turboquant():
    save_dir = model_path("tests", "snn_transformer_turboquant")
    os.makedirs(save_dir, exist_ok=True)

    config = SNNTransformerConfig(
        vocab_size=32,
        embed_dim=16,
        enable_turboquant=True,
        turboquant_main_bits=3,
    )
    model = SpikingTransformerModel(config)
    model.readout_synapses[4] = {5: (1.2, 0), 6: (2.4, 1), 7: (3.6, 2)}
    model.readout_synapses[9] = {3: (0.8, 1), 8: (1.6, 2)}
    model.adaptive_thresholds = {5: 1.1}
    model.target_counts = {6: 2}

    model.save_pretrained(save_dir)
    restored = SpikingTransformerModel.from_pretrained(save_dir)

    assert restored.adaptive_thresholds == model.adaptive_thresholds
    assert restored.target_counts == model.target_counts
    assert set(restored.readout_synapses[4].keys()) == set(model.readout_synapses[4].keys())
    assert set(restored.readout_synapses[9].keys()) == set(model.readout_synapses[9].keys())

    for post_id, (weight, branch_id) in model.readout_synapses[4].items():
        restored_weight, restored_branch = restored.readout_synapses[4][post_id]
        assert restored_branch == branch_id
        assert abs(restored_weight - weight) < 1.5
