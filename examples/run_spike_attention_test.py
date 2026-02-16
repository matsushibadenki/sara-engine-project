# パス: examples/run_spike_attention_test.py
# タイトル: スパイクアテンションの実行テスト
# 目的: ローカルのsrcディレクトリを優先して読み込み、SNNアテンション機構のテストを実行し、膜電位分布の分析を含む各種結果をworkspaceに出力する。

import random
import os
import sys

# 仮想環境の既存パッケージよりローカルのsrcディレクトリを優先して読み込む
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sara_engine.core.spike_attention import SpikeAttention
from sara_engine.utils.snn_visualizer import SNNVisualizer

def main():
    print("SNN SpikeAttention テストを開始します...")
    
    num_neurons = 8
    time_steps = 15
    firing_threshold = 1.5
    
    attention_layer = SpikeAttention(decay_rate=0.8, threshold=firing_threshold)
    visualizer = SNNVisualizer(workspace_dir="workspace")
    
    query_history = []
    output_history = []
    attention_scores_history = []
    all_potentials = [] # ヒストグラム診断用に全スコア（膜電位相当）を保持
    
    # 疑似的なスパイク列の生成と処理
    for t in range(time_steps):
        # 疎な発火をシミュレート
        q_spikes = [1 if random.random() < 0.2 else 0 for _ in range(num_neurons)]
        k_spikes = [1 if random.random() < 0.2 else 0 for _ in range(num_neurons)]
        v_spikes = [1 if random.random() < 0.2 else 0 for _ in range(num_neurons)]
        
        # 意図的な強いパターンの挿入
        if t == 5:
            q_spikes = [1, 0, 1, 0, 1, 0, 1, 0]
            k_spikes = [1, 0, 1, 0, 1, 0, 1, 0]
            v_spikes = [1, 1, 1, 1, 0, 0, 0, 0]
            
        if t == 10:
            q_spikes = [1, 0, 1, 0, 1, 0, 1, 0]
        
        query_history.append(q_spikes)
        
        # 1ステップの処理
        out_spikes, attn_scores = attention_layer.process_step(q_spikes, k_spikes, v_spikes)
        output_history.append(out_spikes)
        attention_scores_history.append(attn_scores)
        
        # 発生したスコアをすべて収集（ゼロのスコアも含めて分布を確認する）
        all_potentials.extend(attn_scores)

    # 可視化ファイルの出力
    visualizer.generate_ascii_raster_plot(query_history, "query_spikes.txt")
    visualizer.generate_ascii_raster_plot(output_history, "output_spikes.txt")
    visualizer.generate_ascii_attention_heatmap(attention_scores_history, "attention_heatmap.txt")
    
    # 膜電位（スコア）分布ヒストグラムの出力
    visualizer.generate_ascii_membrane_potential_histogram(
        potentials=all_potentials, 
        threshold=firing_threshold, 
        bins=15, 
        filename="membrane_potential_hist.txt"
    )
    
    print("テスト完了。結果は workspace/ ディレクトリを確認してください。")

if __name__ == "__main__":
    main()