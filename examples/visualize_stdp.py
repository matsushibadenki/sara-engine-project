_FILE_INFO = {
    "//": "ディレクトリパス: examples/visualize_stdp.py",
    "//": "タイトル: STDP学習と内部状態の可視化デモ",
    "//": "目的: 最新のSARA Engineアーキテクチャに合わせて、SDREncoderとDynamicLiquidLayerを用いた可視化デモに修正。"
}

import numpy as np
import os
import sys

# ローカルパスの優先追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sara_engine.memory.sdr import SDREncoder
from sara_engine.core.layers import DynamicLiquidLayer
from sara_engine.utils.visualizer import SaraVisualizer

def run_visualization():
    print("=== SARA Engine: STDP Visualization Demo ===")
    
    # 1. エンコーダの準備
    sdr_size = 1024
    encoder = SDREncoder(input_size=sdr_size, density=0.05, use_tokenizer=True)
    
    text = "hello sara artificial intelligence is fascinating"
    print(f"Input Text: '{text}'")
    encoder.tokenizer.train([text])
    
    # 2. レイヤーの準備 (3層のLiquid State Machineを模倣)
    print("Initializing layers in Python mode for visualization...")
    l1 = DynamicLiquidLayer(input_size=sdr_size, hidden_size=200, decay=0.9, use_rust=False)
    l2 = DynamicLiquidLayer(input_size=200, hidden_size=200, decay=0.9, use_rust=False)
    l3 = DynamicLiquidLayer(input_size=200, hidden_size=200, decay=0.9, use_rust=False)
    
    # 可視化ツールの初期化
    viz = SaraVisualizer(save_dir="workspace/stdp_logs")
    
    # 履歴バッファ
    spike_history_l1 = []
    spike_history_l2 = []
    spike_history_l3 = []
    attention_history = []
    
    # 実行ループ
    print("Processing spikes and applying STDP...")
    for word in text.split():
        sdr = encoder.encode(word)
        
        # 1単語につき複数ステップ（Echoを見るため）
        for _ in range(5):
            # 順伝播（learning=TrueでSTDPを有効化）
            out1 = l1.forward(active_inputs=sdr, prev_active_hidden=[], learning=True)
            out2 = l2.forward(active_inputs=out1, prev_active_hidden=[], learning=True)
            out3 = l3.forward(active_inputs=out2, prev_active_hidden=[], learning=True)
            
            # スパイク履歴の記録
            spike_history_l1.append(out1)
            spike_history_l2.append(out2)
            spike_history_l3.append(out3)
            
            # アテンション履歴 (可視化用のダミーデータ生成)
            dummy_attn = np.random.choice(60, 3, replace=False).tolist()
            attention_history.append(dummy_attn)

    # 最後のステップの膜電位分布を取得（Layer 2）
    v, thresh = l2.get_state()
    
    # === 可視化の実行 ===
    print("Generating plots...")
    
    # 1. Raster Plot (Layer 2)
    viz.plot_raster(spike_history_l2, title="Layer 2 Spike Raster (Liquid State)", filename="raster_l2.png")
    
    # 2. Membrane Potential Distribution
    viz.plot_membrane_potential_distribution(v, thresh, filename="potential_dist_l2.png")
    
    # 3. Attention Heatmap
    viz.plot_attention_heatmap(attention_history, memory_size=60, filename="attention_map.png")
    
    print(f"\nSuccess! Please check the '{viz.save_dir}' directory for images.")

if __name__ == "__main__":
    run_visualization()