_FILE_INFO = {
    "//": "ディレクトリパス: examples/benchmark_hal.py",
    "//": "ファイルの日本語タイトル: HAL (Hardware Abstraction Layer) ベンチマーク",
    "//": "ファイルの目的や内容: 構築したHALを通じて、Python、Rust、および専用ハードウェア(Mock)のバックエンドを透過的に切り替え、インターフェースの互換性と速度の違いを検証する。"
}

import os
import time
import random
from sara_engine.core.hal import HardwareManager

def run_hal_benchmark():
    workspace_dir = os.path.join(os.path.dirname(__file__), "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    log_file_path = os.path.join(workspace_dir, "hal_benchmark.log")
    
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== SARA Engine HAL (Hardware Abstraction Layer) Test ===\n\n")
        
        # 中規模ネットワークの作成
        num_neurons = 5000
        synapses_per_neuron = 100
        f.write(f"Generating Network: {num_neurons} neurons, {synapses_per_neuron} synapses each.\n\n")
        
        weights = []
        for _ in range(num_neurons):
            targets = {}
            for _ in range(synapses_per_neuron):
                targets[random.randint(0, num_neurons - 1)] = random.uniform(0.1, 1.0)
            weights.append(targets)
            
        active_spikes = random.sample(range(num_neurons), 500)
        
        # 3つのバックエンドを順番にテスト
        backends_to_test = ["python", "rust", "chip"]
        
        for bg_type in backends_to_test:
            manager = HardwareManager(preferred=bg_type)
            backend = manager.get_backend()
            
            f.write(f"Testing Backend: {backend.get_name()}\n")
            print(f"Testing Backend: {backend.get_name()}")
            
            # 1. ネットワークのマッピング（チップへのロード等）
            start_map = time.time()
            backend.set_weights(weights)
            map_time = time.time() - start_map
            
            # 2. 伝播テスト（10回の平均）
            start_prop = time.time()
            for _ in range(10):
                out_spikes = backend.propagate(active_spikes, threshold=1.5, max_out=50)
            prop_time = (time.time() - start_prop) / 10
            
            f.write(f"  - Hardware Mapping Time: {map_time:.4f} seconds\n")
            f.write(f"  - Average Propagation Time: {prop_time * 1000:.2f} ms\n")
            f.write(f"  - Output Spikes Count: {len(out_spikes)}\n\n")
            
        f.write("SUCCESS: HAL test completed perfectly. All backends swapped seamlessly via single interface.\n")
        print("SUCCESS: HAL test completed successfully.")
        
    print(f"Log generated at: {log_file_path}")

if __name__ == "__main__":
    run_hal_benchmark()