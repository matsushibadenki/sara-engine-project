_FILE_INFO = {
    "//": "ディレクトリパス: examples/benchmark_rust.py",
    "//": "タイトル: Rust vs Python ベンチマーク",
    "//": "目的: 計算コアの速度比較を行う。"
}

import sys
import os
import time
import numpy as np

from sara_engine.sara_gpt_core import DynamicLiquidLayer

def run_benchmark():
    print("=== Rust vs Python Benchmark ===")
    
    # 強制的にPythonモードで計測したい場合はソースコードの一時変更が必要だが、
    # ここでは現在のモード（Rustが入っていればRust）の速度を測る。
    
    layer = DynamicLiquidLayer(1024, 2000, 0.5, 0.05)
    mode = "Rust" if layer.use_rust else "Python"
    print(f"Running in {mode} mode.")
    
    steps = 1000
    start_time = time.time()
    
    print(f"Executing {steps} steps...")
    input_spikes = list(range(10)) # Dummy input
    prev_spikes = []
    
    for i in range(steps):
        # 入力を少し変える
        if i % 10 == 0:
            input_spikes = [(x + 1) % 1024 for x in input_spikes]
            
        spikes = layer.forward_with_feedback(input_spikes, prev_spikes)
        prev_spikes = spikes
        
    end_time = time.time()
    duration = end_time - start_time
    fps = steps / duration
    
    print(f"\nResult ({mode}):")
    print(f"  Total Time: {duration:.4f} sec")
    print(f"  Speed: {fps:.2f} steps/sec")
    
    if mode == "Rust":
        print("\nExpectation: Python mode handles ~50-100 steps/sec.")
        print("Rust mode should be significantly faster (1000+ steps/sec).")

if __name__ == "__main__":
    run_benchmark()