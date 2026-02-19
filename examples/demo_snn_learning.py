# [配置するディレクトリのパス]: examples/demo_snn_learning.py
# [ファイルの日本語タイトル]: SARA-Engine SNN学習・睡眠・永続化デモ
# [ファイルの目的や内容]:
# ホメオスタシスによる過剰な抑制（閾値上昇）をリセットし、推論時に安定して発火・予測させる。
# インポートエラーの解消と、インデントエラー(スペース/タブ混在)を厳密な4スペースで修正。

import sys
import os
import time
import random
import json

# プロジェクトルートをパスの先頭に追加し、ローカルのsrcを確実に優先させる
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# cortexではなくlayersから正しくインポートするように修正
from src.sara_engine.core.layers import DynamicLiquidLayer
from src.sara_engine.models.readout_layer import ReadoutLayer

def save_brain_state(readout_layer, filename="sara_readout_state.json"):
    """学習した読み出し層の重みをJSONとして保存する"""
    state = {
        "weights": [dict((int(k), float(v)) for k, v in w.items()) for w in readout_layer.weights],
        "metadata": {
            "input_size": readout_layer.input_size,
            "output_size": readout_layer.output_size,
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
    readout_layer = ReadoutLayer(
        input_size=hidden_size, 
        output_size=num_classes,
        learning_rate=0.005
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
            fired_hidden = liquid_layer.forward_with_feedback(
                active_inputs=active_inputs,
                prev_active_hidden=[]
            )
            accumulated_fired.update(fired_hidden)
        
        if accumulated_fired:
            readout_layer.train_step(
                active_hidden_indices=list(accumulated_fired), 
                target_label=target_label
            )
            
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} 完了")

    end_time = time.time()
    print(f"学習完了 | 所要時間: {end_time - start_time:.4f}s")
    
    # 3. 睡眠フェーズの実行
    print("\n--- 睡眠フェーズ開始 ---")
    sleep_result = readout_layer.sleep_phase(prune_rate=0.05)
    print(sleep_result)
    
    # 4. 推論テスト
    print("\n--- 推論テスト ---")
    
    # 【重要修正】: 学習によって上がりきった発火閾値や活動履歴をリセット
    liquid_layer.reset()
    
    for name, pattern, true_label in [("Pattern A", pattern_a, 0), ("Pattern B", pattern_b, 1)]:
        active_in = [i for i, v in enumerate(pattern) if v == 1]
        
        # テスト時もパターン提示前に膜電位をリセット
        liquid_layer.v = [0.0] * hidden_size
        liquid_layer.refractory = [0.0] * hidden_size
        
        accumulated_fired = set()
        # テスト時は発火を確実にするためステップ数を10に増加
        for _ in range(10):
            fired_h = liquid_layer.forward_with_feedback(active_in, [])
            accumulated_fired.update(fired_h)
        
        potentials = readout_layer.predict(list(accumulated_fired))
        predicted = potentials.index(max(potentials)) if accumulated_fired else -1
        
        formatted_potentials = [round(p, 3) for p in potentials]
        
        print(f"{name}: 予測クラス={predicted} (正解={true_label}) | ポテンシャル={formatted_potentials}")

    # 5. 学習結果の永続化
    save_brain_state(readout_layer)
    
    print("="*50)

if __name__ == "__main__":
    run_snn_learning_demo()