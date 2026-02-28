from src.sara_engine.nn.attention import SpikeSelfAttention
from src.sara_engine.core.layers import DynamicLiquidLayer
from src.sara_engine.core.hal import HardwareManager
import multiprocessing
import random
import time
import sys
import os
_FILE_INFO = {
    "//": "ディレクトリパス: examples/integrated_rust_acceleration_benchmark.py",
    "//": "ファイルの日本語タイトル: 統合Rustアクセラレーション・ベンチマーク",
    "//": "ファイルの目的や内容: Python実装とRustコア(sara_rust_core)の性能を比較。LIFネットワーク、シナプス学習、大規模行列演算の各項目で加速効果を測定する。"
}


# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def run_performance_test(name, task_fn, iterations=1):
    start = time.time()
    for _ in range(iterations):
        task_fn()
    duration = time.time() - start
    return duration


def main():
    print("=== SARA Engine: Integrated Rust Acceleration Benchmark ===")
    print(f"Detected CPU Cores: {multiprocessing.cpu_count()}")

    workspace_dir = os.path.join(os.path.dirname(__file__), "workspace")
    os.makedirs(workspace_dir, exist_ok=True)

    # 1. HAL (Hardware Abstraction Layer) スイッチングテスト
    print("\n[1] Testing HAL Backend Switching...")
    for bg_type in ["python", "rust"]:
        manager = HardwareManager(preferred=bg_type)
        backend = manager.get_backend()
        print(
            f"  - Preferred: {bg_type:>6} -> Active Backend: {backend.get_name()}")

    # 2. LIF Network & Causal Synapses (Rust Direct vs Python)
    print("\n[2] Massive Synapse Propagation Benchmark (1M Synapses)")
    num_neurons = 10000
    synapses_per_neuron = 100
    active_spikes = random.sample(range(num_neurons), 1000)

    # ネットワークデータの準備
    weights = []
    for _ in range(num_neurons):
        targets = {random.randint(0, num_neurons - 1): random.uniform(0.1, 1.0)
                   for _ in range(synapses_per_neuron)}
        weights.append(targets)

    # --- Python Mode ---
    py_layer = DynamicLiquidLayer(
        input_size=num_neurons, hidden_size=num_neurons, decay=0.95, use_rust=False)
    py_layer.in_weights = weights  # 重みのセット

    def py_task(): py_layer.forward(active_inputs=active_spikes, prev_active_hidden=[])
    py_time = run_performance_test("Python", py_task, iterations=5) / 5
    print(f"  - Python Avg Time: {py_time * 1000:8.2f} ms")

    # --- Rust Mode ---
    rust_layer = DynamicLiquidLayer(
        input_size=num_neurons, hidden_size=num_neurons, decay=0.95, use_rust=True)
    if not rust_layer.use_rust:
        print("  - Rust core not available. Skipping Rust benchmark.")
        return

    rust_layer.core.set_weights(weights)

    def rust_task(): rust_layer.forward(
        active_inputs=active_spikes, prev_active_hidden=[])
    rust_time = run_performance_test("Rust", rust_task, iterations=50) / 50
    print(f"  - Rust   Avg Time: {rust_time * 1000:8.2f} ms")

    speedup = py_time / rust_time if rust_time > 0 else 0
    print(f"  => Acceleration: {speedup:.2f}x faster")

    # 3. SpikeSelfAttention (Latest API Specification)
    print("\n[3] Attention Mechanism Benchmark (Embed Dim: 1024)")
    embed_dim = 1024
    test_input = random.sample(range(embed_dim), 100)

    py_attn = SpikeSelfAttention(embed_dim=embed_dim, use_rust=False)
    rust_attn = SpikeSelfAttention(embed_dim=embed_dim, use_rust=True)

    def py_attn_task(): py_attn._sparse_propagate(
        test_input, py_attn.q_weights, embed_dim, 0.5, 50)

    def rust_attn_task(): rust_attn.q_engine.propagate(test_input, 0.5, 50)

    py_at_time = run_performance_test("Py-Attn", py_attn_task, iterations=50)
    rust_at_time = run_performance_test(
        "Rust-Attn", rust_attn_task, iterations=50)

    print(f"  - Python Total Time: {py_at_time:.4f} s")
    print(f"  - Rust   Total Time: {rust_at_time:.4f} s")
    print(f"  => Acceleration: {py_at_time/rust_at_time:.2f}x faster")

    print("\n=== Benchmark Completed Successfully ===")
    if speedup > 1.0:
        print("NOTE: Native Acceleration Active and verified.")
    else:
        print("WARNING: Rust core performance did not exceed Python. Check build optimization.")


if __name__ == "__main__":
    main()
