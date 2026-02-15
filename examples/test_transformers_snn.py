_FILE_INFO = {
    "//": "ディレクトリパス: examples/test_transformers_snn.py",
    "//": "タイトル: SNN Transformer 結合テスト",
    "//": "目的: 仮想環境の古いパッケージとの競合を避け、確実に追加モジュールをテストする。"
}

import os
import sys
import numpy as np

# 何よりも先にローカルの src ディレクトリを sys.path の先頭に挿入する
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from sara_engine.core.transformer import PlasticTransformerBlock
from sara_engine.core.normalization import SpikeIntrinsicPlasticity
from sara_engine.core.dropout import SpikeDropout
from sara_engine.utils.visualizer import SNNVisualizer

def run_test():
    workspace_dir = os.path.join(os.path.dirname(__file__), "..", "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    
    viz = SNNVisualizer(workspace_dir=workspace_dir)
    
    d_model = 128
    num_heads = 4
    seq_len = 30
    
    transformer = PlasticTransformerBlock(d_model=d_model, num_heads=num_heads)
    layer_norm = SpikeIntrinsicPlasticity(d_model=d_model, target_rate=0.1)
    dropout = SpikeDropout(drop_rate=0.2)
    
    rng = np.random.RandomState(42)
    sequence = []
    for t in range(seq_len):
        spikes = rng.choice(d_model, size=max(1, int(d_model * 0.1)), replace=False).tolist()
        sequence.append(spikes)
        
    print("Training SNN Transformer components (Biological Equivalent)...")
    
    history_out = []
    attention_matrix = np.zeros((seq_len, seq_len))
    
    for t, spikes in enumerate(sequence):
        out_spikes = transformer.compute(spikes, pos=t, learning=True)
        
        for past_t in range(t):
            overlap = len(set(sequence[t]).intersection(set(sequence[past_t])))
            attention_matrix[t, past_t] = overlap
            
        drop_spikes = dropout.compute(out_spikes, learning=True)
        norm_spikes = layer_norm.compute(drop_spikes, learning=True)
        
        history_out.append(norm_spikes)
        
    print("Generating visualizations...")
    viz.plot_raster(history_out, title="Output Spike Raster (After IP Norm & Dropout)", filename="transformer_output_raster.png")
    viz.plot_attention_heatmap(attention_matrix, title="Spike Attention Overlap Heatmap", filename="attention_heatmap.png")
    
    print(f"Test complete. Please check the '{workspace_dir}' directory for logs and images.")

if __name__ == "__main__":
    run_test()
