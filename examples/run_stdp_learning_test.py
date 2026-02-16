# パス: examples/run_stdp_learning_test.py
# タイトル: STDP学習レイヤーの自己組織化テスト
# 目的: 教師データを用いずに、STDPレイヤーが入力パターンを自律的に学習し、特化していく様子を確認する。
# {
#     "//": "テスト結果はworkspaceディレクトリに出力される"
# }

import os
import sys
import random

# ローカルのsrcディレクトリを優先して読み込む
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sara_engine.learning.stdp import STDPLayer
from sara_engine.utils.snn_visualizer import SNNVisualizer

def print_weights(layer: STDPLayer, filename: str, workspace_dir: str):
    """学習された結合荷重（重み）の分布を可視化してテキスト出力する"""
    filepath = os.path.join(workspace_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=== STDP Synaptic Weights ===\n")
        f.write("Row: Input Neuron, Col: Output Neuron\n")
        f.write("Legend: [.]<0.2, [-]<0.5, [+]<0.8, [#]>=0.8\n\n")
        
        # ヘッダー
        header = "    " + "".join([f"O{j:02d} " for j in range(layer.num_outputs)])
        f.write(header + "\n")
        
        for i in range(layer.num_inputs):
            line = f"I{i:02d}| "
            for j in range(layer.num_outputs):
                w = layer.weights[i][j]
                char = "."
                if w >= 0.8: char = "#"
                elif w >= 0.5: char = "+"
                elif w >= 0.2: char = "-"
                
                line += f" {char}  "
            f.write(line + "\n")
    print(f"重みマップを保存しました: {filepath}")

def main():
    print("STDPレイヤーの自己組織化学習テストを開始します...")
    
    num_inputs = 10
    num_outputs = 4
    time_steps = 300 # 特徴抽出を安定させるため少し長めに学習
    
    # 閾値を低めに設定して発火と学習のサイクルを誘発
    stdp_layer = STDPLayer(num_inputs=num_inputs, num_outputs=num_outputs, threshold=1.2)
    visualizer = SNNVisualizer(workspace_dir="workspace")
    
    query_history = []
    output_history = []
    
    # 学習前のランダムな重みを保存
    print_weights(stdp_layer, "stdp_weights_before.txt", "workspace")
    
    # 隠された規則（頻出するパターン）を2種類定義
    # パターンA: 前半のニューロン(0~4)が発火
    pattern_A = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    # パターンB: 後半のニューロン(5~9)が発火
    pattern_B = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    
    # 教師データなし（オンライン自己組織化学習）
    for t in range(time_steps):
        rand_val = random.random()
        base_pattern = [0] * num_inputs
        
        # 40%でパターンA、40%でパターンB、20%は完全なランダムノイズ
        if rand_val < 0.4:
            base_pattern = pattern_A.copy()
        elif rand_val < 0.8:
            base_pattern = pattern_B.copy()
            
        # パターンにランダムなノイズ（揺らぎ）を10%付与
        input_spikes = []
        for bit in base_pattern:
            if random.random() < 0.1:
                input_spikes.append(1 - bit)
            else:
                input_spikes.append(bit)
                
        query_history.append(input_spikes)
        
        # SNNにデータを通すだけで自動的にSTDPがシナプスを調整する
        out_spikes, _ = stdp_layer.process_step(input_spikes)
        output_history.append(out_spikes)

    # 学習後の出力と可視化
    visualizer.generate_ascii_raster_plot(query_history, "stdp_input_spikes.txt")
    visualizer.generate_ascii_raster_plot(output_history, "stdp_output_spikes.txt")
    print_weights(stdp_layer, "stdp_weights_after.txt", "workspace")
    
    print("テスト完了。学習前後のシナプス結合荷重の変化は workspace/ の before と after を比較してください。")

if __name__ == "__main__":
    main()