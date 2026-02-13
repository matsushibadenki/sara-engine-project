_FILE_INFO = {
    "//": "ディレクトリパス: examples/visualize_stdp.py",
    "//": "タイトル: STDP学習と内部状態の可視化デモ",
    "//": "目的: use_rust=False 指定によるPythonモード強制の修正。"
}

import numpy as np
import os
from sara_engine import SaraGPT, SaraVisualizer

def run_visualization():
    print("=== SARA Engine: STDP Visualization Demo ===")
    print("Note: Ensure 'sara-engine' is installed via pip (e.g., pip install -e .)")
    
    # モデルの初期化
    sdr_size = 1024
    engine = SaraGPT(sdr_size=sdr_size)
    
    # Pythonモードを強制（可視化データ取得のため）
    print("Initializing engine in Python mode for visualization...")
    for layer in engine.layers:
        layer.use_rust = False
        # [Fix] use_rust=False を明示的に指定
        layer.__init__(
            input_size=layer.input_size, 
            hidden_size=layer.size, 
            decay=layer.decay,
            density=layer.density,
            input_scale=layer.input_scale,
            rec_scale=layer.rec_scale,
            feedback_scale=layer.feedback_scale,
            use_rust=False,
            target_rate=0.05 # [Fix] 目標発火率を設定
        )
    
    # 可視化ツールの初期化
    viz = SaraVisualizer(save_dir="workspace/stdp_logs")
    
    # データの準備
    text = "hello sara artificial intelligence is fascinating"
    print(f"Input Text: '{text}'")
    
    # 履歴バッファ
    spike_history_l1 = []
    spike_history_l2 = []
    spike_history_l3 = []
    attention_history = []
    
    # 実行ループ
    print("Processing spikes and applying STDP...")
    for word in text.split():
        sdr = engine.encoder.encode(word)
        
        # 1単語につき複数ステップ（Echoを見るため）
        for _ in range(5):
            # 順伝播（Training=TrueでSTDPを有効化）
            predicted_sdr, all_spikes = engine.forward_step(sdr, training=True)
            
            # スパイク履歴の記録
            spike_history_l1.append(engine.prev_spikes[0])
            spike_history_l2.append(engine.prev_spikes[1])
            spike_history_l3.append(engine.prev_spikes[2])
            
            # アテンション履歴
            if engine.attention_active:
                dummy_attn = np.random.choice(60, 3, replace=False).tolist()
                attention_history.append(dummy_attn)

    # 最後のステップの膜電位分布を取得（Layer 2）
    v, thresh = engine.l2.get_state()
    
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