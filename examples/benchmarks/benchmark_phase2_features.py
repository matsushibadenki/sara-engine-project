# ディレクトリパス: examples/benchmarks/benchmark_phase2_features.py
# ファイルの日本語タイトル: フェーズ2機能ベンチマーク
# ファイルの目的や内容: 曖昧さの許容(Fuzzy Recall)と階層処理の動作確認。

import os
import time
from sara_engine.models.hierarchical_snn import HierarchicalSNN
try:
    from sara_engine.sara_rust_core import calculate_sdr_overlap
except ImportError:
    calculate_sdr_overlap = None

def run_benchmark():
    # 結果保存用ディレクトリ
    work_dir = "workspace/logs"
    os.makedirs(work_dir, exist_ok=True)
    
    print("--- [1] Testing Fuzzy Recall (Overlap Logic) ---")
    if calculate_sdr_overlap:
        sdr1 = [1, 5, 10, 15, 20]
        sdr2 = [1, 6, 10, 16, 20] # 60% overlap
        overlap = calculate_sdr_overlap(sdr1, sdr2)
        print(f"Overlap Score: {overlap:.2f}")
    else:
        print("Rust core function 'calculate_sdr_overlap' not found.")

    print("\n--- [2] Testing Hierarchical Processing ---")
    configs = [
        {"embed_dim": 256}, # Layer 0
        {"embed_dim": 128}, # Layer 1
        {"embed_dim": 64}   # Layer 2
    ]
    model = HierarchicalSNN(layer_configs=configs)
    
    test_input = [10, 45, 102, 200]
    start = time.time()
    output = model.forward(test_input, learning=True)
    end = time.time()
    
    print(f"Input Spikes: {test_input}")
    print(f"Output Spikes: {output}")
    print(f"Time: {(end - start)*1000:.2f} ms")

    # ログ出力
    with open(os.path.join(work_dir, "phase2_report.log"), "w") as f:
        f.write(f"Inference result: {output}\n")
        f.write(f"Latency: {(end - start)*1000} ms\n")

if __name__ == "__main__":
    run_benchmark()