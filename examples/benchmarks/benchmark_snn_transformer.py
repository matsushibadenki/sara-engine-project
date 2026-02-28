_FILE_INFO = {
    "//": "ディレクトリパス: examples/benchmark_snn_transformer.py",
    "//": "ファイルの日本語タイトル: スパイキング・トランスフォーマー ベンチマークテスト",
    "//": "ファイルの目的や内容: SNN版トランスフォーマーモデルの学習、テキスト生成(推論)、およびモデルの保存・読み込みの完全なパイプラインをテストし、結果とモデル状態をworkspaceに出力する。"
}

import os
import time
from sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig

def run_benchmark():
    # ワークスペースディレクトリの設定 (テストアーティファクトの出力先)
    workspace_dir = os.path.join(os.path.dirname(__file__), "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    
    log_file_path = os.path.join(workspace_dir, "snn_transformer_benchmark.log")
    model_save_dir = os.path.join(workspace_dir, "saved_snn_transformer")

    print("Starting SNN Transformer Benchmark...")

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== SNN Transformer Benchmark Log ===\n")
        
        # 1. モデルの初期化
        f.write("\n[1] Initializing Spiking Transformer Model...\n")
        start_time = time.time()
        
        # テスト用に軽量な設定を使用（ただし文字レベルのUnicode語彙をサポート）
        config = SNNTransformerConfig(
            vocab_size=65536,
            embed_dim=64,
            num_layers=2,
            ffn_dim=128,
            num_pathways=2,
            dropout_p=0.1,
            target_spikes_ratio=0.25
        )
        model = SpikingTransformerModel(config)
        f.write(f"Initialization took {time.time() - start_time:.2f} seconds.\n")

        # 2. 学習フェーズ (BPなしのSTDP局所学習)
        f.write("\n[2] Starting STDP Learning Phase...\n")
        training_data = "SARA is a purely spike-driven biological AI engine."
        f.write(f"Training sequence: '{training_data}'\n")
        
        start_time = time.time()
        model.learn_sequence(training_data)
        f.write(f"Learning took {time.time() - start_time:.2f} seconds.\n")

        # 3. 推論フェーズ (テキスト生成)
        f.write("\n[3] Starting Generation Phase...\n")
        prompt = "SARA is"
        f.write(f"Prompt: '{prompt}'\n")
        
        start_time = time.time()
        # 学習させた内容の続きを生成する
        generated_text = model.generate(prompt, max_length=50)
        f.write(f"Generation took {time.time() - start_time:.2f} seconds.\n")
        f.write(f"Generated Result: '{generated_text}'\n")

        # 4. モデルの保存
        f.write("\n[4] Testing Model Saving...\n")
        start_time = time.time()
        model.save_pretrained(model_save_dir)
        f.write(f"Model saved to '{model_save_dir}' in {time.time() - start_time:.2f} seconds.\n")

        # 5. モデルの読み込み
        f.write("\n[5] Testing Model Loading...\n")
        start_time = time.time()
        loaded_model = SpikingTransformerModel.from_pretrained(model_save_dir)
        f.write(f"Model loaded successfully in {time.time() - start_time:.2f} seconds.\n")

        # 6. 読み込んだモデルでの再生成テスト
        f.write("\n[6] Generation Test with Loaded Model...\n")
        start_time = time.time()
        generated_text_loaded = loaded_model.generate(prompt, max_length=50)
        f.write(f"Loaded Model Generation took {time.time() - start_time:.2f} seconds.\n")
        f.write(f"Loaded Model Generated Result: '{generated_text_loaded}'\n")
        
        # 決定論的なスパイク計算により、状態が正しく復元されていれば完全に同じ出力になる
        if generated_text == generated_text_loaded:
            f.write("\nSUCCESS: Saved and loaded models produce identical outputs.\n")
            print("SUCCESS: State Dictionary and Load tests passed.")
        else:
            f.write("\nWARNING: Loaded model output differs from original.\n")
            print("WARNING: Loaded model behavior differs.")

    print(f"Benchmark completed successfully.")
    print(f"Log saved at: {log_file_path}")
    print(f"Model state saved at: {model_save_dir}")

if __name__ == "__main__":
    run_benchmark()