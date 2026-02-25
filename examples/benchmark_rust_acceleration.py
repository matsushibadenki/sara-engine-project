_FILE_INFO = {
    "//": "ディレクトリパス: examples/benchmark_rust_acceleration.py",
    "//": "ファイルの日本語タイトル: Rustコア加速ベンチマーク",
    "//": "ファイルの目的や内容: Python純粋実装とRustコア実装のスパイク処理速度を比較し、Phase 3で目標とする「大規模スパイク処理」の実現可能性を検証する。"
}

import time
import random
from sara_engine import sara_rust_core
from sara_engine.nn.attention import SpikeSelfAttention

def run_performance_comparison():
    print("=== SARA Engine: Rust Core Acceleration Benchmark ===\n")
    
    embed_dim = 1024  # 大規模な次元数
    num_spikes = 100  # 1ステップあたりの発火数
    iterations = 50
    
    # 1. Python純粋実装の準備
    py_attention = SpikeSelfAttention(embed_dim=embed_dim, density=0.1)
    
    # 2. Rustエンジンの準備
    rust_engine = sara_rust_core.SpikeEngine()
    # Python側の初期重みをRustへ転送 (互換性テスト)
    rust_engine.set_weights(py_attention.q_weights)
    
    test_input = random.sample(range(embed_dim), num_spikes)

    # --- Python Performance Test ---
    print(f"[*] Testing Python implementation ({iterations} iterations)...")
    start_py = time.time()
    for _ in range(iterations):
        # SpikeSelfAttention内部の _sparse_propagate を模倣
        _ = py_attention._sparse_propagate(test_input, py_attention.q_weights, embed_dim)
    end_py = time.time()
    py_duration = end_py - start_py
    print(f"  -> Python Total Time: {py_duration:.4f}s")

    # --- Rust Performance Test ---
    print(f"\n[*] Testing Rust core implementation ({iterations} iterations)...")
    start_rust = time.time()
    for _ in range(iterations):
        # Rustの高速メソッドを直接呼び出し
        _ = rust_engine.propagate(test_input, threshold=0.5, max_out=int(embed_dim * 0.15))
    end_rust = time.time()
    rust_duration = end_rust - start_rust
    print(f"  -> Rust Total Time: {rust_duration:.4f}s")

    # --- Conclusion ---
    speedup = py_duration / rust_duration if rust_duration > 0 else 0
    print(f"\n[Result]")
    print(f"  Acceleration: {speedup:.2f}x faster than Python")
    
    if speedup > 1.0:
        print("\n=> SUCCESS: Rust core significantly reduces overhead for spike propagation!")
    else:
        print("\n=> NOTE: Overhead of data conversion might be dominant for small spike counts.")

if __name__ == "__main__":
    run_performance_comparison()