_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_transformer_snn.py",
    "//": "タイトル: SNN-based Transformer実行デモ",
    "//": "目的: 作成したSpikeTransformerBlockの動作確認と可視化ツールのテストを行う。"
}

import os
import sys
import numpy as np

# プロジェクトルートをPYTHONPATHに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sara_engine.core.transformer import SpikeTransformerBlock, SpikePositionalEncoding
from sara_engine.utils.visualizer import SaraVisualizer

def generate_dummy_sequence(seq_len: int, d_model: int, density: float = 0.05):
    """ダミーの時系列スパイク入力を生成"""
    sequence = []
    num_active = max(1, int(d_model * density))
    for _ in range(seq_len):
        spikes = np.random.choice(d_model, num_active, replace=False).tolist()
        sequence.append(spikes)
    return sequence

def main():
    # 設定
    d_model = 256
    num_heads = 4
    seq_len = 50
    memory_size = 100
    
    # 出力先ディレクトリ (workspace下)
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'workspace', 'logs', 'transformer_demo'))
    os.makedirs(workspace_dir, exist_ok=True)
    print(f"[{os.path.basename(__file__)}] Logs will be saved to: {workspace_dir}")
    
    # モジュールの初期化
    pos_encoder = SpikePositionalEncoding(d_model=d_model, max_len=seq_len)
    transformer_block = SpikeTransformerBlock(d_model=d_model, num_heads=num_heads, memory_size=memory_size)
    visualizer = SaraVisualizer(save_dir=workspace_dir)
    
    # ダミー入力データの生成
    inputs = generate_dummy_sequence(seq_len, d_model)
    
    # 記録用リスト
    input_history = []
    output_history = []
    attn_heatmap_data = [] 
    
    print("Running SNN Transformer Block...")
    for t in range(seq_len):
        raw_spikes = inputs[t]
        pos_spikes = pos_encoder.get_spikes(t)
        
        # 入力とPositional Encodingの結合 (Union)
        embedded_spikes = list(set(raw_spikes) | set(pos_spikes))
        input_history.append(embedded_spikes)
        
        # Transformer Block の計算
        out_spikes = transformer_block.compute(embedded_spikes)
        output_history.append(out_spikes)
        
        # アテンションの活性化状況として出力を記録
        attn_heatmap_data.append(out_spikes)
        
    print("Simulation finished. Generating visualizations...")
    
    # 1. ラスタープロット（入力と出力）
    visualizer.plot_raster(input_history, title="Input Spikes (with Positional Encoding)", filename="input_raster.png")
    visualizer.plot_raster(output_history, title="Transformer Block Output Spikes", filename="output_raster.png")
    
    # 2. アテンション・ヒートマップ (出力の活性化度合いを代用可視化)
    visualizer.plot_attention_heatmap(attn_heatmap_data, memory_size=d_model, filename="attention_activity.png")
    
    print("All visualizations saved successfully.")

if __name__ == "__main__":
    main()