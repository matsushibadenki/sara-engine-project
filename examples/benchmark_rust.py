_FILE_INFO = {
    "//": "ディレクトリパス: examples/benchmark_rust.py",
    "//": "タイトル: Rust vs Python ベンチマーク",
    "//": "目的: DynamicLiquidLayerのforward_with_feedback廃止に伴い、新しいforwardメソッドへ移行する。"
}

import time
import sys
import os

# プロジェクトルートをパスの先頭に追加し、site-packagesの古いモジュールより優先させる
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# src配下から明示的にインポートするように修正
from src.sara_engine.core.layers import DynamicLiquidLayer

def run_benchmark():
    print("=== SARA Engine: Rust vs Python Benchmark ===")
    
    # パッケージからレイヤーを初期化
    # Rust実装が利用可能な場合は自動的に使用される
    layer = DynamicLiquidLayer(input_size=1024, hidden_size=2000, decay=0.5, density=0.05)
    
    mode = "Rust" if layer.use_rust else "Python"
    print(f"Running in {mode} mode.")
    
    steps = 1000
    start_time = time.time()
    
    print(f"Executing {steps} steps...")
    input_spikes: list[int] = list(range(10)) 
    prev_spikes: list[int] = []
    
    for i in range(steps):
        if i % 10 == 0:
            input_spikes = [(x + 1) % 1024 for x in input_spikes]
            
        # 順伝播
        spikes = layer.forward(active_inputs=input_spikes, prev_active_hidden=prev_spikes)
        prev_spikes = spikes
        
    duration = time.time() - start_time
    fps = steps / duration
    
    print(f"\nResult ({mode}):")
    print(f"  Total Time: {duration:.4f} sec")
    print(f"  Speed: {fps:.2f} steps/sec")
    
    if mode == "Rust":
        print("\nOptimization Status: Native Acceleration Active.")
    else:
        print("\nOptimization Status: Using Python Fallback.")

if __name__ == "__main__":
    run_benchmark()