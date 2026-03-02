_FILE_INFO = {
    "//": "ディレクトリパス: examples/benchmarks/benchmark_predictive_lm.py",
    "//": "ファイルの日本語タイトル: 予測符号化LMベンチマーク",
    "//": "ファイルの目的や内容: 階層モデルと予測符号化を統合したLLM代替モデルを用いて、系列パターンの学習と自律的なテキスト（トークン）生成をテストする。最初のトークンの誤差を除外して評価するよう修正。"
}

import os
from sara_engine.models.spiking_predictive_lm import SpikingPredictiveLM

def run_benchmark():
    work_dir = "workspace/logs"
    os.makedirs(work_dir, exist_ok=True)
    log_file = os.path.join(work_dir, "predictive_lm_benchmark.log")
    
    # 階層構成：入力を 128次元 -> 64次元 に抽象化
    configs = [{"embed_dim": 128}, {"embed_dim": 64}]
    model = SpikingPredictiveLM(vocab_size=1000, layer_configs=configs, max_delay=5, learning_rate=0.2, predict_threshold=0.1)
    
    # 学習データ（擬似的な文章のトークン系列）
    # 想定文章: "I"(10) -> "eat"(20) -> "an"(30) -> "apple"(40)
    sequence = [
        [10], # I
        [20], # eat
        [30], # an
        [40], # apple
    ]
    
    print("--- 1. 予測符号化LMの自己回帰学習 (Training) ---")
    epochs = 15
    with open(log_file, "w") as f:
        f.write("Predictive LM Generation Benchmark\n==================================\n\n")
        
        for epoch in range(epochs):
            model.reset_state()
            total_errors = 0
            
            # 系列を順番に読ませる
            for step, tokens in enumerate(sequence):
                error_spikes = model.forward(tokens, learning=True)
                # 最初の単語は予測不能で必ず誤差になるため、評価(合否判定)から除外する
                if step > 0:
                    total_errors += len(error_spikes)
                
            log_msg = f"[Epoch {epoch+1}] 累積誤差スパイク数(Step 1以降): {total_errors}"
            print(log_msg)
            f.write(log_msg + "\n")
            
            # 続きの文脈(Step 1〜)の誤差がゼロになれば、系列を完璧に記憶・予測できたことを意味する
            if total_errors == 0:
                success_msg = "=> 系列の学習が完了し、予測誤差がゼロに収束しました！"
                print(success_msg)
                f.write(success_msg + "\n")
                break
                
    print("\n--- 2. テキストの自律生成 (Text Generation) ---")
    # 状態をリセットし、プロンプトだけを与えて続きを生成させる
    model.reset_state()
    prompt = [10] # "I"
    print(f"プロンプト (Prompt Tokens): {prompt}")
    
    # 最大3つのトークンを生成
    generated_sequence = model.generate(prompt_tokens=prompt, max_length=3)
    print(f"生成結果 (Generated Sequence): {generated_sequence}")
    
    if generated_sequence == [10, 20, 30, 40]:
        print("=> 成功: プロンプトから正確に次の文脈を予測し、自律的に文章を生成しました！")

if __name__ == "__main__":
    run_benchmark()