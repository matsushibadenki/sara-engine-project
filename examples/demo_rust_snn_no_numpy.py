"""
{
    "//": "配置パス: examples/demo_rust_snn_no_numpy.py",
    "//": "日本語タイトル: Numpy非依存のRust版SNNエンジン連携デモ",
    "//": "目的: 行列演算やnumpyを使わず、Python標準のリスト処理だけでRustの計算エンジンを高速に呼び出す実証を行う。"
}
"""

import random
import time
from sara_engine.sara_rust_core import RustLiquidLayer, RustSpikeAttention

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
    
    print("=== Rust SNN Engine Demo (No Numpy, No Matrix Math) ===")
    
    # Rust側で初期化（重みや状態はすべてRust側のメモリに保持される）
    print("Initializing RustLiquidLayer...")
    layer = RustLiquidLayer(
        input_size=input_size,
        hidden_size=hidden_size,
        decay=0.9,
        density=0.1,
        feedback_scale=0.5
    )
    
    print("Initializing RustSpikeAttention...")
    attention = RustSpikeAttention(
        input_size=hidden_size,
        hidden_size=256,
        num_heads=4,
        memory_size=10
    )

    prev_active_hidden = []
    feedback_active = []
    
    start_time = time.time()
    
    # 100ステップのシミュレーション
    for step in range(100):
        # 1. 外部からの入力スパイク（インデックスのリスト）を生成
        active_inputs = generate_random_spikes(input_size, probability=0.05)
        
        # 2. Liquid Layerへのフォワードパス（状態更新とSTDP学習）
        attention_signal = []
        active_hidden = layer.forward(
            active_inputs,
            prev_active_hidden,
            feedback_active,
            attention_signal,
            learning=True
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