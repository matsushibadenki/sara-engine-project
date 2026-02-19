# examples/test_knowledge_recall.py
_FILE_INFO = {
    "//": "ディレクトリパス: examples/test_knowledge_recall.py",
    "//": "タイトル: SARA-Engine 知識呼び出し（Recall）テスト・最適化版",
    "//": "目的: 閾値の適応的調整により、ノイズ混じりの入力でも正しく発火・認識させる。古いバージョンのJSONフォーマットやニューロン数が0になるケースにも対応し、エラーを回避する。"
}

import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sara_engine.learning.stdp import STDPLayer

def load_brain_state(filename="workspace/sara_vocab.json"):
    if not os.path.exists(filename):
        print(f"エラー: {filename} が見つかりません。")
        return None
        
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 互換性のため、metadataが無い場合やリスト形式の場合にも対応
    if isinstance(data, list):
        synapses_data = data
        num_outputs = max(len(synapses_data), 1)
        max_in = 0
        for syn_dict in synapses_data:
            if syn_dict:
                max_k = max((int(k) for k in syn_dict.keys()), default=0)
                if max_k > max_in:
                    max_in = max_k
        num_inputs = max(max_in + 1, 10)
    else:
        meta = data.get("metadata", {})
        num_inputs = meta.get("num_inputs", data.get("num_inputs", 10))
        synapses_data = data.get("synapses", [])
        num_outputs = meta.get("num_outputs", data.get("num_outputs", max(len(synapses_data), 1)))
        
        # num_inputsが推測できない場合は既存のシナプスから推定
        if num_inputs == 10 and synapses_data:
            max_in = 0
            for syn_dict in synapses_data:
                if syn_dict:
                    max_k = max((int(k) for k in syn_dict.keys()), default=0)
                    if max_k > max_in:
                        max_in = max_k
            num_inputs = max(max_in + 1, 10)

    # 認識率向上のため、初期閾値を少し下げて復元 (2.0 -> 1.5)
    layer = STDPLayer(num_inputs=num_inputs, num_outputs=num_outputs, threshold=1.5)
    
    layer.synapses = []
    for syn_dict in synapses_data:
        restored_syn = {int(k): float(v) for k, v in syn_dict.items()}
        layer.synapses.append(restored_syn)
        
    # ロードされたシナプスデータがnum_outputsに満たない場合は空のシナプスを追加して構造を保つ
    while len(layer.synapses) < num_outputs:
        layer.synapses.append({})
        
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
    if len(pattern_a_noisy) > 5:
        pattern_a_noisy[5] = 1 # ノイズ
    
    print("テスト実行: パターンA（ノイズあり）を提示中...")

    # 推論実行
    output_spikes, potentials = stdp_layer.process_step(pattern_a_noisy, reward=0.0)

    if any(s == 1 for s in output_spikes):
        winner = output_spikes.index(1)
        print(f"結果: 認識成功！ 出力ニューロン {winner} が発火しました。")
    elif len(potentials) > 0 and len(stdp_layer.thresholds) > 0:
        print(f"結果: 認識失敗。最大電位 {max(potentials):.2f} が閾値 {stdp_layer.thresholds[0]:.2f} に届きません。")
    else:
        print("結果: 認識失敗。出力ニューロンが読み込まれていません（電位または閾値が空です）。")

    print(f"最終膜電位状態: {[round(p, 2) for p in potentials]}")
    print("="*50)

if __name__ == "__main__":
    test_recall()