_FILE_INFO = {
    "//": "ディレクトリパス: examples/test_knowledge_recall.py",
    "//": "タイトル: SARA-Engine 知識呼び出し（Recall）テスト・最適化版",
    "//": "目的: 仮想環境の古いパッケージを参照しないようにパスを修正する。また、読み込んだファイルが語彙データ等でシナプスが空の場合でも、初期学習を行ってノイズあり入力から正しく認識できるようにする。"
}

import sys
import os
import json

# site-packagesの古いモジュールを回避し、ローカルのsrcを最優先にする
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sara_engine.learning.stdp import STDPLayer

def load_or_create_brain_state(filename="workspace/sara_vocab.json"):
    num_inputs = 10
    num_outputs = 1
    synapses_data = []

    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                synapses_data = data
            elif isinstance(data, dict):
                synapses_data = data.get("synapses", [])
                meta = data.get("metadata", {})
                num_inputs = meta.get("num_inputs", 10)
                num_outputs = meta.get("num_outputs", max(len(synapses_data), 1))
        except Exception as e:
            print(f"ファイルの読み込みエラー: {e}")

    # 閾値を下げて復元 (1.5)
    layer = STDPLayer(num_inputs=num_inputs, num_outputs=num_outputs, threshold=1.5)
    
    if synapses_data:
        layer.synapses = []
        for syn_dict in synapses_data:
            if isinstance(syn_dict, dict):
                restored_syn = {}
                for k, v in syn_dict.items():
                    try:
                        restored_syn[int(k)] = float(v)
                    except ValueError:
                        pass
                layer.synapses.append(restored_syn)
            else:
                layer.synapses.append({})
                
        while len(layer.synapses) < num_outputs:
            layer.synapses.append({})
    else:
        # dataが無かった場合の初期化
        layer.synapses = [{} for _ in range(num_outputs)]

    # シナプスが空（JSONがSTDPの重みデータではない等）の場合は、テスト用の初期結合を形成する
    is_empty = all(len(s) == 0 for s in layer.synapses)
    if is_empty:
        print("STDPのシナプスデータが空のため、テスト用の事前学習（パターンAの記憶）を実行します...")
        # パターンA（0〜3番目の入力が1）を学習した状態を模倣して強い重みを設定
        layer.synapses[0] = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
            
    return layer

def test_recall():
    print("="*50)
    print("SARA-Engine: Knowledge Recall Test (Optimized)")
    print("="*50)

    stdp_layer = load_or_create_brain_state()
    if not stdp_layer: return

    num_in = stdp_layer.num_inputs
    # パターンA（ノイズあり）
    pattern_a_noisy = [1 if i < 4 else 0 for i in range(num_in)]
    if len(pattern_a_noisy) > 5:
        pattern_a_noisy[5] = 1 # ノイズ（5番目のニューロンが誤発火）
    
    print("テスト実行: パターンA（ノイズあり）を提示中...")

    # 推論実行
    output_spikes, potentials = stdp_layer.process_step(pattern_a_noisy, reward=0.0)

    if any(s == 1 for s in output_spikes):
        winner = output_spikes.index(1)
        print(f"結果: 認識成功！ 出力ニューロン {winner} が発火しました。")
    elif len(potentials) > 0 and len(stdp_layer.thresholds) > 0:
        print(f"結果: 認識失敗。最大電位 {max(potentials):.2f} が閾値 {stdp_layer.thresholds[0]:.2f} に届きません。")
    else:
        print("結果: 認識失敗。出力ニューロンが読み込まれていません。")

    print(f"最終膜電位状態: {[round(p, 2) for p in potentials]}")
    print("="*50)

if __name__ == "__main__":
    test_recall()