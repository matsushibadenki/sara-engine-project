_FILE_INFO = {
    "//": "ディレクトリパス: examples/benchmarks/benchmark_predictive_lm_phase2.py",
    "//": "ファイルの日本語タイトル: 予測符号化LMの学習・生成ベンチマーク",
    "//": "ファイルの目的や内容: 教師なしのエラー主導学習（Predictive Coding）により、系列データを繰り返すことで予測誤差(Surprise)が減少することを確認する。また、学習結果に基づくトークンの自己回帰生成(generate)をテストする。"
}

import os
import sys
import time

# ローカルのsrcを優先インポート
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from sara_engine.models.spiking_predictive_lm import SpikingPredictiveLM

def run_predictive_lm_benchmark():
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "workspace", "logs"))
    os.makedirs(workspace_dir, exist_ok=True)
    log_file = os.path.join(workspace_dir, "predictive_lm_report.log")
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== SARA Engine Phase 2: Predictive Coding LM Benchmark ===\n\n")

    print("=== [1] 初始化 (Initialization) ===")
    # 階層モデル（エンコーダ）のダミー設定
    layer_configs = [{"embed_dim": 128}, {"embed_dim": 64}]
    
    # モデルのインスタンス化 (学習率を高めにして収束を早くする)
    model = SpikingPredictiveLM(
        vocab_size=1000, 
        layer_configs=layer_configs, 
        max_delay=5, 
        learning_rate=0.5, 
        predict_threshold=0.1
    )
    
    print("モデルの初期化が完了しました。行列演算は使用されていません。")
    
    # テスト用のシンプルな系列パターン (例: 単語のIDの並び)
    # 10 -> 20 -> 30 -> 40 -> 50 という法則性を持つ
    sequence = [10, 20, 30, 40, 50]
    epochs = 5
    
    print(f"\n=== [2] 系列の学習 (Unsupervised Sequence Learning) ===")
    print(f"Target Sequence: {sequence}")
    print("各エポックで発生した『予測誤差(Surprise)スパイクの数』を計測します。学習が進むと誤差は減少するはずです。\n")
    
    for epoch in range(1, epochs + 1):
        model.reset_state() # 系列の先頭に戻るため状態をリセット
        total_errors = 0
        
        start_time = time.time()
        for token_id in sequence:
            # 予測誤差（今回予測できなかったスパイクのリスト）を取得
            error_spikes = model.forward([token_id], learning=True)
            total_errors += len(error_spikes)
            
        elapsed = (time.time() - start_time) * 1000
        
        msg = f"Epoch {epoch}: Total Surprise Spikes = {total_errors} (Latency: {elapsed:.2f} ms)"
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            
    print("\n=== [3] 自己回帰生成 (Autoregressive Generation) ===")
    # 学習した法則を利用して、プロンプトの続きを予測・生成させる
    prompt = [10, 20]
    print(f"Prompt: {prompt}")
    
    model.reset_state()
    start_time = time.time()
    # 続きを3トークン生成させる
    generated_sequence = model.generate(prompt_tokens=prompt, max_length=3)
    elapsed = (time.time() - start_time) * 1000
    
    msg_gen = (
        f"Generated Sequence: {generated_sequence}\n"
        f"(Expected to match or be close to {sequence})\n"
        f"Generation Latency: {elapsed:.2f} ms"
    )
    print(msg_gen)
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n--- Generation Test ---\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(msg_gen + "\n")
        f.write(f"\nLog saved successfully to {log_file}\n")

    print(f"\nベンチマーク完了。ログは {log_file} に保存されました。")

if __name__ == "__main__":
    run_predictive_lm_benchmark()