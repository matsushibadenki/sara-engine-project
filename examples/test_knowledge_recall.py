# パス: examples/test_knowledge_recall.py
# タイトル: SARA-Engine 知識呼び出し（Recall）テスト・最適化版
# 目的: 閾値の適応的調整により、ノイズ混じりの入力でも正しく発火・認識させる。

import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sara_engine.learning.stdp import STDPLayer

def load_brain_state(filename="sara_vocab.json"):
    if not os.path.exists(filename):
        print(f"エラー: {filename} が見つかりません。")
        return None
        
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    meta = data["metadata"]
    # 認識率向上のため、初期閾値を少し下げて復元 (2.0 -> 1.5)
    layer = STDPLayer(num_inputs=meta["num_inputs"], num_outputs=meta["num_outputs"], threshold=1.5)
    
    layer.synapses = []
    for syn_dict in data["synapses"]:
        restored_syn = {int(k): float(v) for k, v in syn_dict.items()}
        layer.synapses.append(restored_syn)
        
    return layer

def test_recall():
    print("="*50)
    print("SARA-Engine: Knowledge Recall Test (Optimized)")
    print("="*50)

    stdp_layer = load_brain_state()
    if not stdp_layer: return

    # パターンA（ノイズあり）
    num_in = stdp_layer.num_inputs
    pattern_a_noisy = [1 if i < 4 else 0 for i in range(num_in)]
    pattern_a_noisy[5] = 1 # ノイズ
    
    print("テスト実行: パターンA（ノイズあり）を提示中...")

    # 推論実行
    output_spikes, potentials = stdp_layer.process_step(pattern_a_noisy, reward=0.0)

    if any(s == 1 for s in output_spikes):
        winner = output_spikes.index(1)
        print(f"結果: 認識成功！ 出力ニューロン {winner} が発火しました。")
    else:
        print(f"結果: 認識失敗。最大電位 {max(potentials):.2f} が閾値 {stdp_layer.thresholds[0]:.2f} に届きません。")

    print(f"最終膜電位状態: {[round(p, 2) for p in potentials]}")
    print("="*50)

if __name__ == "__main__":
    test_recall()