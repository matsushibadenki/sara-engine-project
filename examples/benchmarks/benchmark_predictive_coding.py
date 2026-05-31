# ディレクトリパス: examples/benchmarks/benchmark_predictive_coding.py
# ファイルの日本語タイトル: 予測符号化ベンチマーク
# ファイルの目的や内容: パターン入力に対する予測誤差率の減衰（自律学習と省エネの証明）をテストする。最新のPredictiveCodingManagerを用いて、時系列シーケンスの自己組織化学習を検証する。エポック間で状態メモリを保持し、連続的なストリームとしての予測を可能にする。

import os
from typing import List, Dict, Set
from sara_engine.learning.predictive_coding import PredictiveCodingManager

def run_benchmark():
    work_dir = "workspace/logs"
    os.makedirs(work_dir, exist_ok=True)
    log_file = os.path.join(work_dir, "predictive_coding_benchmark.log")
    
    manager = PredictiveCodingManager(learning_rate=0.2)
    
    # ネットワークの重み（辞書によるスパース結合）
    # 過去の状態(pre)から現在の入力(post)を予測するトップダウン結合
    backward_weights: List[Dict[int, float]] = [{} for _ in range(100)]
    
    # 周期的な繰り返しパターンをストリームとして流す
    # [10, 20] -> [30, 40] -> [50, 60] -> [10, 20] ...
    pattern_sequence = [
        [10, 20],
        [30, 40],
        [50, 60]
    ]
    
    epochs = 15
    print("--- 予測符号化 (Predictive Routing) ベンチマーク ---")
    
    # 状態メモリ（1ステップ前の入力スパイクを保持）
    # ループの外に出すことで、エポックを跨いで連続したストリームとして予測を継続する
    prev_spikes: List[int] = []
    
    with open(log_file, "w") as f:
        f.write("Predictive Coding Learning Log\n")
        f.write("==============================\n")
        
        for epoch in range(epochs):
            total_error_rate = 0.0
            print(f"\n[Epoch {epoch+1}]")
            f.write(f"\n[Epoch {epoch+1}]\n")
            
            for step, actual_spikes in enumerate(pattern_sequence):
                # 1. 過去の状態から現在の入力を予測 (Top-down prediction)
                predicted_spikes: Set[int] = set()
                if prev_spikes:
                    for p_spike in prev_spikes:
                        for target, weight in backward_weights[p_spike].items():
                            if weight > 0.5: # 予測閾値
                                predicted_spikes.add(target)
                
                # 2. 予測誤差(Surprise)の計算
                actual_set = set(actual_spikes)
                # 予測できなかった未知の入力（上位層へ送られる誤差スパイク）
                surprise_spikes = actual_set - predicted_spikes
                
                # 誤差率の計算 (0.0 なら完全予測)
                error_rate = len(surprise_spikes) / len(actual_spikes) if actual_spikes else 0.0
                total_error_rate += error_rate
                manager.record_error(error_rate)
                
                # 3. 局所学習則の適用 (Predictive Coding update)
                if prev_spikes:
                    manager.update_backward(
                        backward_weights=backward_weights,
                        prev_state_spikes=prev_spikes,
                        current_in_spikes=actual_spikes,
                        predicted_in_spikes=predicted_spikes,
                        lr=manager.learning_rate
                    )
                
                log_msg = f"  Step {step}: Input {actual_spikes} -> Predicted {list(predicted_spikes)} | Error Spikes {list(surprise_spikes)} (Error Rate: {error_rate:.2f})"
                print(log_msg)
                f.write(log_msg + "\n")
                
                # 状態を更新
                prev_spikes = actual_spikes
                
            avg_error = total_error_rate / len(pattern_sequence)
            accuracy = manager.get_prediction_accuracy()
            print(f"  => Epoch Avg Error Rate: {avg_error:.2f} | Model Accuracy Metric: {accuracy:.2f}")
            f.write(f"  => Epoch Avg Error Rate: {avg_error:.2f}\n")
            
            # 誤差が完全に無くなったら学習完了
            if avg_error == 0.0:
                print("\n[SUCCESS] 完全にパターンを予測できました。これ以降は上位レイヤーへのスパイク伝播（無駄な計算）が抑制されます。")
                f.write("\n[SUCCESS] Perfect Prediction Achieved.\n")
                break

if __name__ == "__main__":
    run_benchmark()