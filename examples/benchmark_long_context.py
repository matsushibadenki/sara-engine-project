_FILE_INFO = {
    "//": "ディレクトリパス: examples/benchmark_long_context.py",
    "//": "ファイルの日本語タイトル: 長距離依存性（ロングコンテキスト）ベンチマーク",
    "//": "ファイルの目的や内容: ROADMAP Phase 2における長距離依存性のテスト。大量のノイズ（数千ステップ）の後に過去の記憶をSTDPで保持・想起できるかを検証する。"
}

import os
import time
import random
from sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig

def run_long_context_benchmark():
    workspace_dir = os.path.join(os.path.dirname(__file__), "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    log_file_path = os.path.join(workspace_dir, "long_context_benchmark.log")

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== SNN Long-Context Retrieval Benchmark ===\n\n")
        
        # モデル初期化 (コンテキストウィンドウは極端に短く設定)
        config = SNNTransformerConfig(embed_dim=128, num_layers=2)
        model = SpikingTransformerModel(config)
        # SNNTransformerModelのリングバッファ(短期記憶)の長さを意図的に狭くする
        model.context_length = 16 
        
        f.write(f"Model initialized. Short-term memory buffer (context_length) is strictly limited to: {model.context_length} tokens.\n\n")

        # 1. 重要な事実（パスワード）の学習
        secret_fact = "The secret vault password is: OMEGA42."
        f.write(f"[Step 1] Injecting target fact: '{secret_fact}'\n")
        model.learn_sequence(secret_fact)

        # 2. 大量のノイズデータのストリーム学習（時間の経過と干渉のシミュレート）
        # コンテキストウィンドウ(16)を遥かに超える長さの無関係なテキストを学習させる
        noise_length = 2000
        f.write(f"[Step 2] Simulating passage of time... Streaming {noise_length} tokens of random noise.\n")
        
        start_time = time.time()
        noise_tokens = [random.randint(65, 90) for _ in range(noise_length)] # Random A-Z
        for tok in noise_tokens:
            # ノイズを学習させることで、過去の短期記憶バッファは完全に上書き・消失する
            model.forward_step(tok, learning=True, target_id=random.choice(noise_tokens))
        
        f.write(f"Noise processing completed in {time.time() - start_time:.2f} seconds.\n")
        f.write(f"Current delay buffer state (should contain only noise): {[chr(c) for c in model.delay_buffer]}\n\n")

        # 3. 想起テスト（Long-Context Retrieval）
        prompt = "The secret vault password is:"
        f.write(f"[Step 3] Retrieval Test Prompt: '{prompt}'\n")
        
        # 短期記憶はノイズで消えているが、STDPによってシナプスに「構造」として残っていれば思い出せる
        generated = model.generate(prompt, max_length=10)
        f.write(f"Generated Output: '{generated}'\n\n")

        # 判定
        if "OMEGA42" in generated:
            f.write("RESULT: SUCCESS (PASS) \n")
            f.write("The model successfully retrieved the information outside its short-term context window using synaptic structural plasticity (STDP). Long-Range Dependency is maintained!\n")
            print("SUCCESS: Long-Context Retrieval Test Passed!")
        else:
            f.write("RESULT: FAILED\n")
            f.write("The model suffered from catastrophic forgetting.\n")
            print("FAILED: Model forgot the secret fact.")

    print(f"Benchmark finished. Log saved to: {log_file_path}")

if __name__ == "__main__":
    run_long_context_benchmark()