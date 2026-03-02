_FILE_INFO = {
    "//": "ディレクトリパス: examples/benchmarks/benchmark_predictive_coding.py",
    "//": "ファイルの日本語タイトル: 予測符号化ベンチマーク",
    "//": "ファイルの目的や内容: パターン入力に対する予測誤差率の減衰（自律学習と省エネの証明）をテストする。"
}

import os
from sara_engine.nn.predictive import PredictiveCodingLayer

def run_benchmark():
    work_dir = "workspace/logs"
    os.makedirs(work_dir, exist_ok=True)
    log_file = os.path.join(work_dir, "predictive_coding_benchmark.log")
    
    # max_delayを長くしてパターンを記憶しやすくする
    layer = PredictiveCodingLayer(max_delay=5, learning_rate=0.2, threshold=0.3)
    
    # 周期的な繰り返しパターンをストリームとして流す
    # [10, 20] の次には必ず [30, 40] が来るというルール
    pattern_sequence = [
        [10, 20],
        [30, 40],
        [50, 60]
    ]
    
    epochs = 10
    
    print("--- 予測符号化 (Predictive Routing) ベンチマーク ---")
    
    with open(log_file, "w") as f:
        f.write("Predictive Coding Learning Log\n")
        f.write("==============================\n")
        
        for epoch in range(epochs):
            total_error_rate = 0.0
            print(f"\n[Epoch {epoch+1}]")
            f.write(f"\n[Epoch {epoch+1}]\n")
            
            for step, actual_spikes in enumerate(pattern_sequence):
                error_spikes, error_rate = layer.forward(actual_spikes, learning=True)
                total_error_rate += error_rate
                
                log_msg = f"  Step {step}: Input {actual_spikes} -> Error Spikes {error_spikes} (Error Rate: {error_rate:.2f})"
                print(log_msg)
                f.write(log_msg + "\n")
                
            avg_error = total_error_rate / len(pattern_sequence)
            print(f"  => Epoch Avg Error Rate: {avg_error:.2f}")
            f.write(f"  => Epoch Avg Error Rate: {avg_error:.2f}\n")
            
            # 誤差が0になったら完全に予測モデルが構築されたことを意味する
            if avg_error == 0.0:
                print("\n[SUCCESS] 完全にパターンを予測できました。これ以降は上位レイヤーへのスパイク伝播（無駄な計算）がスキップされます。")
                f.write("\n[SUCCESS] Perfect Prediction Achieved.\n")
                break

if __name__ == "__main__":
    run_benchmark()