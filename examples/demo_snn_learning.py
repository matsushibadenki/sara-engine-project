_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_learning.py",
    "//": "タイトル: SARA-Engine SNN学習・睡眠・永続化デモ",
    "//": "目的: 最新のSpikeReadoutLayerとDynamicLiquidLayerのAPI(forward等)に適合させ、エラーを解消する。"
}

import sys
import os
import time
import random
import json

# プロジェクトルートをパスの先頭に追加し、ローカルのsrcを確実に優先させる
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sara_engine.core.layers import DynamicLiquidLayer
from src.sara_engine.models.readout_layer import SpikeReadoutLayer

def save_brain_state(readout_layer, filename="workspace/sara_readout_state.json"):
    """学習した読み出し層の重みをJSONとして保存する"""
    if not os.path.exists("workspace"):
        os.makedirs("workspace")
        
    state = {
        "W": [dict((int(k), float(v)) for k, v in w.items()) for w in readout_layer.W],
        "b": readout_layer.b,
        "metadata": {
            "input_size": readout_layer.d_model,
            "output_size": readout_layer.vocab_size,
            "timestamp": time.time()
        }
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)
    print(f"\n[記憶完了] 学習済みデータを {filename} に保存しました。")

def run_snn_learning_demo():
    print("="*50)
    print("SARA-Engine: Supervised Learning & Sleep Phase Demo")
    print("="*50)

    num_inputs = 20
    hidden_size = 50
    num_classes = 2
    epochs = 1000
    
    # 1. リザーバー層の初期化
    liquid_layer = DynamicLiquidLayer(
        input_size=num_inputs, 
        hidden_size=hidden_size, 
        decay=0.85, 
        target_rate=0.05,
        density=0.2,       
        input_scale=2.0,   
        use_rust=False
    )
    
    # 2. 読み出し層の初期化
    readout_layer = SpikeReadoutLayer(
        d_model=hidden_size, 
        vocab_size=num_classes,
        learning_rate=0.005,
        use_refractory=False
    )

    # 2つの異なる入力パターン (AとB)
    pattern_a = [1 if i < 4 else 0 for i in range(num_inputs)]
    pattern_b = [1 if i > 16 else 0 for i in range(num_inputs)]

    print(f"学習開始 (エポック数: {epochs})...")
    start_time = time.time()

    for epoch in range(epochs):
        is_pattern_a = random.random() < 0.5
        current_pattern = pattern_a if is_pattern_a else pattern_b
        target_label = 0 if is_pattern_a else 1
        
        active_inputs = [i for i, v in enumerate(current_pattern) if v == 1]
        
        # 学習時の局所的な膜電位リセット
        liquid_layer.v = [0.0] * hidden_size
        liquid_layer.refractory = [0.0] * hidden_size
        
        accumulated_fired = set()
        
        # SNNの時間発展 (5タイムステップ)
        for _ in range(5):
            fired_hidden = liquid_layer.forward(
                active_inputs=active_inputs,
                prev_active_hidden=[]
            )
            accumulated_fired.update(fired_hidden)
        
        if accumulated_fired:
            readout_layer.forward(
                list(accumulated_fired), 
                target_token=target_label,
                learning=True
            )
            
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} 完了")

    end_time = time.time()
    print(f"学習完了 | 所要時間: {end_time - start_time:.4f}s")
    
    # 3. 睡眠フェーズの実行
    print("\n--- 睡眠フェーズ開始 ---")
    pruned_count = 0
    for s_idx in range(len(readout_layer.W)):
        to_delete = [t_id for t_id, weight in readout_layer.W[s_idx].items() if abs(weight) < 0.05]
        for t_id in to_delete:
            del readout_layer.W[s_idx][t_id]
            pruned_count += 1
    print(f"睡眠フェーズ完了: 全ての層から {pruned_count} 個のシナプスを枝刈りしました。")
    
    # 4. 推論テスト
    print("\n--- 推論テスト ---")
    
    liquid_layer.reset()
    
    for name, pattern, true_label in [("Pattern A", pattern_a, 0), ("Pattern B", pattern_b, 1)]:
        active_in = [i for i, v in enumerate(pattern) if v == 1]
        
        liquid_layer.v = [0.0] * hidden_size
        liquid_layer.refractory = [0.0] * hidden_size
        
        accumulated_fired = set()
        for _ in range(10):
            fired_h = liquid_layer.forward(active_inputs=active_in, prev_active_hidden=[])
            accumulated_fired.update(fired_h)
            
        potentials_dict = {i: readout_layer.b[i] for i in range(num_classes)}
        for s in accumulated_fired:
            if s < len(readout_layer.W):
                for tid, w in readout_layer.W[s].items():
                    potentials_dict[tid] += w
        potentials = [potentials_dict[i] for i in range(num_classes)]
        
        predicted = readout_layer.forward(list(accumulated_fired), learning=False)
        formatted_potentials = [round(p, 3) for p in potentials]
        
        print(f"{name}: 予測クラス={predicted} (正解={true_label}) | ポテンシャル={formatted_potentials}")

    # 5. 学習結果の永続化
    save_brain_state(readout_layer)
    
    print("="*50)

if __name__ == "__main__":
    run_snn_learning_demo()