# [配置するディレクトリのパス]: ./examples/visualize_stdp.py
# [ファイルの日本語タイトル]: STDP学習と内部状態の可視化デモ
# [ファイルの目的や内容]: 廃止されたget_state()の代わりに、レイヤーの膜電位(v)と動的閾値(dynamic_thresh)へ直接アクセスして可視化を行う。
_FILE_INFO = {
    "//": "ディレクトリパス: examples/visualize_stdp.py",
    "//": "タイトル: STDP学習と内部状態の可視化デモ",
    "//": "目的: 廃止されたget_state()の代わりに、レイヤーの膜電位(v)と動的閾値(dynamic_thresh)へ直接アクセスして可視化を行う。"
}

import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sara_engine.memory.sdr import SDREncoder
from sara_engine.core.layers import DynamicLiquidLayer
from sara_engine.utils.visualizer import SaraVisualizer

def run_visualization():
    print("=== SARA Engine: STDP Visualization Demo ===")
    
    sdr_size = 1024
    encoder = SDREncoder(input_size=sdr_size, density=0.05, use_tokenizer=True)
    
    text = "hello sara artificial intelligence is fascinating"
    print(f"Input Text: '{text}'")
    encoder.tokenizer.train([text])
    
    print("Initializing layers in Python mode for visualization...")
    l1 = DynamicLiquidLayer(input_size=sdr_size, hidden_size=200, decay=0.9, use_rust=False)
    l2 = DynamicLiquidLayer(input_size=200, hidden_size=200, decay=0.9, use_rust=False)
    l3 = DynamicLiquidLayer(input_size=200, hidden_size=200, decay=0.9, use_rust=False)
    
    viz = SaraVisualizer(save_dir="workspace/stdp_logs")
    
    spike_history_l1 = []
    spike_history_l2 = []
    spike_history_l3 = []
    attention_history = []
    
    print("Processing spikes and applying STDP...")
    for word in text.split():
        sdr = encoder.encode(word)
        
        for _ in range(5):
            out1 = l1.forward(active_inputs=sdr, prev_active_hidden=[], learning=True)
            out2 = l2.forward(active_inputs=out1, prev_active_hidden=[], learning=True)
            out3 = l3.forward(active_inputs=out2, prev_active_hidden=[], learning=True)
            
            spike_history_l1.append(out1)
            spike_history_l2.append(out2)
            spike_history_l3.append(out3)
            
            dummy_attn = np.random.choice(60, 3, replace=False).tolist()
            attention_history.append(dummy_attn)

    # get_state()が廃止されたため、直接プロパティから状態を取得し、numpy配列化する
    v = np.array(l2.v)
    thresh = np.array(l2.dynamic_thresh)
    
    print("Generating plots...")
    viz.plot_raster(spike_history_l2, title="Layer 2 Spike Raster (Liquid State)", filename="raster_l2.png")
    viz.plot_membrane_potential_distribution(v, thresh, filename="potential_dist_l2.png")
    viz.plot_attention_heatmap(attention_history, memory_size=60, filename="attention_map.png")
    
    print(f"\nSuccess! Please check the '{viz.save_dir}' directory for images.")

if __name__ == "__main__":
    run_visualization()