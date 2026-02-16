# パス: examples/demo_snn_learning.py
# タイトル: SARA-Engine SNN学習・永続化デモ
# 目的: 学習したシナプス構造をJSONとして保存し、知識の再利用を可能にする。

import sys
import os
import time
import random
import json

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sara_engine.learning.stdp import STDPLayer
from src.sara_engine.core.layers import DynamicLiquidLayer

def save_brain_state(stdp_layer, filename="sara_vocab.json"):
    """学習したシナプス接続を保存する"""
    state = {
        "synapses": [dict((int(k), float(v)) for k, v in syn.items()) for syn in stdp_layer.synapses],
        "metadata": {
            "num_inputs": stdp_layer.num_inputs,
            "num_outputs": stdp_layer.num_outputs,
            "timestamp": time.time()
        }
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)
    print(f"\n[記憶完了] 学習済みデータを {filename} に保存しました。")

def run_snn_learning_demo():
    print("="*50)
    print("SARA-Engine: Learning & Knowledge Persistence")
    print("="*50)

    num_inputs = 20
    num_outputs = 5
    epochs = 1000
    
    liquid_layer = DynamicLiquidLayer(
        input_size=num_inputs, 
        hidden_size=20, 
        decay=0.85, 
        target_rate=0.02,
        use_rust=False
    )
    
    stdp_layer = STDPLayer(num_inputs=num_inputs, num_outputs=num_outputs)
    stdp_layer.A_minus = 0.08
    stdp_layer.prune_threshold = 0.05

    pattern_a = [1 if i < 4 else 0 for i in range(num_inputs)]
    pattern_b = [1 if i > 16 else 0 for i in range(num_inputs)]

    print(f"学習開始...")
    start_time = time.time()

    for epoch in range(epochs):
        current_pattern = pattern_a if random.random() < 0.5 else pattern_b
        
        liquid_fired = liquid_layer.forward_with_feedback(
            active_inputs=[i for i, v in enumerate(current_pattern) if v == 1],
            prev_active_hidden=[]
        )
        
        if liquid_fired:
            stdp_input = [1 if i in liquid_fired else 0 for i in range(num_inputs)]
            stdp_layer.process_step(stdp_input, reward=1.2)

    end_time = time.time()
    
    # 学習結果の保存
    save_brain_state(stdp_layer)
    
    print("="*50)
    print(f"完了 | 所要時間: {end_time - start_time:.4f}s")
    print(f"最終シナプス密度: {(sum(len(s) for s in stdp_layer.synapses) / (num_inputs * num_outputs)) * 100:.1f}%")
    print("="*50)

if __name__ == "__main__":
    run_snn_learning_demo()