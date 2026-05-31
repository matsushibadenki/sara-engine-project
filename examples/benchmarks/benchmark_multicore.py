_FILE_INFO = {
    "//": "ディレクトリパス: examples/benchmark_multicore.py",
    "//": "ファイルの日本語タイトル: マルチコア分散処理ベンチマーク",
    "//": "ファイルの目的や内容: 新たに並列最適化されたRustコア (SpikeEngine) に、極めて大規模なシナプスネットワークとスパイクを与え、行列演算やGPUなしでの伝播速度を計測する。"
}

import time
import random
import os
import multiprocessing

try:
    from sara_engine import sara_rust_core
except ImportError:
    print("Error: Please compile the rust core using 'maturin develop --release' first.")
    exit(1)

def run_benchmark():
    workspace_dir = os.path.join(os.path.dirname(__file__), "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    log_file_path = os.path.join(workspace_dir, "multicore_benchmark.log")
    
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== Multi-core CPU Spike Propagation Benchmark ===\n")
        f.write(f"Detected CPU Cores: {multiprocessing.cpu_count()}\n\n")
        print(f"Detected CPU Cores: {multiprocessing.cpu_count()}")
        
        engine = sara_rust_core.SpikeEngine()
        
        # 大規模ネットワークの設定: 10,000 ニューロン, 各100シナプス (合計100万シナプス)
        num_neurons = 10000
        synapses_per_neuron = 100
        
        print("Initializing massive synapse network (1 million synapses)...")
        start = time.time()
        
        weights = []
        for i in range(num_neurons):
            targets = {}
            # ランダムな後方ニューロンへの結合
            for _ in range(synapses_per_neuron):
                post = random.randint(0, num_neurons - 1)
                targets[post] = random.uniform(0.1, 1.0)
            weights.append(targets)
            
        engine.set_weights(weights)
        f.write(f"Network initialization took {time.time() - start:.2f} seconds.\n")
        
        # 1000個のニューロンが同時に発火（スパイク）したという極端な状況
        active_spikes = random.sample(range(num_neurons), 1000)
        
        print("Running parallel propagation (Rayon Map-Reduce)...")
        # ウォームアップ
        engine.propagate(active_spikes, 1.5, 50)
        
        # ベンチマーク本番
        start = time.time()
        iterations = 100
        for _ in range(iterations):
            out = engine.propagate(active_spikes, 1.5, 50)
            
        total_time = time.time() - start
        avg_time = total_time / iterations
        
        f.write(f"\n[Results]\n")
        f.write(f"Processed 1,000 active spikes across 1,000,000 possible synapses.\n")
        f.write(f"Iterations: {iterations}\n")
        f.write(f"Total Time: {total_time:.4f} seconds\n")
        f.write(f"Average Time per Propagation: {avg_time * 1000:.2f} ms\n")
        
        print(f"Success! Average Time per Propagation: {avg_time * 1000:.2f} ms")
        print(f"Log generated at: {log_file_path}")

if __name__ == "__main__":
    run_benchmark()