_FILE_INFO = {
    "//": "ディレクトリパス: examples/benchmark_rust_acceleration.py",
    "//": "ファイルの日本語タイトル: Rustコア加速ベンチマーク (引数仕様修正版)",
    "//": "ファイルの目的や内容: SpikeSelfAttentionの最新の引数仕様に対応し、Python純粋モードとRust加速モードの速度差を正確に計測する。"
}

import time
import random
from sara_engine.nn.attention import SpikeSelfAttention

def run_performance_comparison():
    print("=== SARA Engine: Rust Core Acceleration Benchmark ===\n")
    
    embed_dim = 1024  # 大規模な次元数
    num_spikes = 100  # 1ステップあたりの発火数
    iterations = 50
    threshold = 0.5
    max_out = int(embed_dim * 0.15)
    
    # 1. Python純粋モードのインスタンス (use_rust=False)
    py_attention = SpikeSelfAttention(embed_dim=embed_dim, density=0.1, use_rust=False)
    
    # 2. Rust加速モードのインスタンス (use_rust=True)
    # 内部で自動的に sara_rust_core.SpikeEngine が生成され、重みが同期される
    rust_attention = SpikeSelfAttention(embed_dim=embed_dim, density=0.1, use_rust=True)
    
    test_input = random.sample(range(embed_dim), num_spikes)

    # --- Python Performance Test ---
    print(f"[*] Testing Python implementation ({iterations} iterations)...")
    start_py = time.time()
    for _ in range(iterations):
        # 最新の引数仕様 (threshold, max_out) を指定して呼び出し
        _ = py_attention._sparse_propagate(test_input, py_attention.q_weights, embed_dim, threshold, max_out)
    end_py = time.time()
    py_duration = end_py - start_py
    print(f"  -> Python Total Time: {py_duration:.4f}s")

    # --- Rust Performance Test ---
    print(f"\n[*] Testing Rust core implementation ({iterations} iterations)...")
    # Rustモードが有効であることを確認
    if not rust_attention.use_rust:
        print("  -> ERROR: Rust core is not available. Check your build.")
        return

    start_rust = time.time()
    for _ in range(iterations):
        # Rustエンジンによる高速伝播を呼び出し
        _ = rust_attention.q_engine.propagate(test_input, threshold, max_out)
    end_rust = time.time()
    rust_duration = end_rust - start_rust
    print(f"  -> Rust Total Time: {rust_duration:.4f}s")

    # --- Conclusion ---
    speedup = py_duration / rust_duration if rust_duration > 0 else 0
    print(f"\n[Result]")
    print(f"  Acceleration: {speedup:.2f}x faster than Python")
    
    if speedup > 1.0:
        print("\n=> SUCCESS: Rust core integration confirmed with high performance!")
    else:
        print("\n=> NOTE: Performance gain might be low for small scale tasks.")

if __name__ == "__main__":
    run_performance_comparison()