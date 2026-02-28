_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_rust_snn_no_numpy.py",
    "//": "タイトル: Numpy非依存のSNNエンジン連携デモ",
    "//": "目的: 未実装のRust関数への直接参照を避け、公式のPythonラッパー(DynamicLiquidLayer, SpikeAttention)を利用する。"
}

import random
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sara_engine.core.layers import DynamicLiquidLayer
from src.sara_engine.core.attention import SpikeAttention

def generate_random_spikes(num_neurons: int, probability: float) -> list[int]:
    """
    numpyを使わずに、指定した確率で発火したニューロンのインデックスリスト（SDR）を生成する。
    """
    spikes = []
    for i in range(num_neurons):
        if random.random() < probability:
            spikes.append(i)
    return spikes

def run_demo():
    input_size = 1000
    hidden_size = 500
    
    print("=== SNN Engine Demo (No Numpy, No Matrix Math) ===")
    
    print("Initializing DynamicLiquidLayer...")
    layer = DynamicLiquidLayer(
        input_size=input_size,
        hidden_size=hidden_size,
        decay=0.9,
        density=0.1,
        feedback_scale=0.5
    )
    
    print("Initializing SpikeAttention...")
    attention = SpikeAttention(
        input_size=hidden_size,
        hidden_size=256,
        num_heads=4,
        memory_size=10
    )

    prev_active_hidden: list[int] = []
    feedback_active: list[int] = []
    
    start_time = time.time()
    
    # 100ステップのシミュレーション
    for step in range(100):
        # 1. 外部からの入力スパイク（インデックスのリスト）を生成
        active_inputs = generate_random_spikes(input_size, probability=0.05)
        
        # 2. Liquid Layerへのフォワードパス（状態更新とSTDP学習）
        attention_signal: list[int] = []
        active_hidden = layer.forward(
            active_inputs=active_inputs,
            prev_active_hidden=prev_active_hidden,
            feedback_active=feedback_active,
            attention_signal=attention_signal,
            learning=True,
            reward=1.0
        )
        
        # 3. Attention機構の計算（SDRの共通集合による類似度計算）
        context_spikes = attention.compute(active_hidden)
        
        # 次のステップのための状態更新
        prev_active_hidden = active_hidden
        feedback_active = context_spikes
        
        if step % 20 == 0:
            print(f"Step {step}: Input Spikes={len(active_inputs)}, Hidden Spikes={len(active_hidden)}, Context Spikes={len(context_spikes)}")

    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.4f} seconds.")

if __name__ == "__main__":
    run_demo()