_FILE_INFO = {
    "//": "ディレクトリパス: examples/benchmarks/benchmark_predictive_lm.py",
    "//": "ファイルの日本語タイトル: 予測符号化言語モデルの生成ベンチマーク",
    "//": "ファイルの目的や内容: SpikingPredictiveLM が純粋なSDRと予測誤差を用いて、単語の系列（文法）を自己組織化し、未知のプロンプトから正しい続きの単語を生成できるかテストする。HierarchicalSNNの必須パラメータを追加修正。"
}

import os
from sara_engine.models.spiking_predictive_lm import SpikingPredictiveLM

def run_lm_benchmark():
    print("=== SARA Engine: Spiking Predictive LM Benchmark ===\n")
    
    # 語彙の定義 (簡易的な自然言語シーケンス)
    vocab = {"<PAD>": 0, "I": 1, "am": 2, "a": 3, "cat": 4, ".": 5, "You": 6, "are": 7, "dog": 8}
    reverse_vocab = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    
    # 階層モデルの設定（KeyErrorを解消するため embed_dim 等を追加）
    # 念のため num_heads などのよく使われるトランスフォーマー系パラメータも付与しておきます
    layer_configs = [
        {"name": "L1", "embed_dim": 128, "size": 128, "num_heads": 4, "threshold": 0.5, "decay": 0.9},
        {"name": "L2", "embed_dim": 64,  "size": 64,  "num_heads": 2, "threshold": 0.6, "decay": 0.8}
    ]
    
    model = SpikingPredictiveLM(
        vocab_size=vocab_size,
        layer_configs=layer_configs,
        max_delay=5,
        learning_rate=0.2,
        predict_threshold=0.1
    )
    
    # 学習用コーパス（2つの文章を交互に学習させる）
    corpus = [
        [vocab["I"], vocab["am"], vocab["a"], vocab["cat"], vocab["."]],
        [vocab["You"], vocab["are"], vocab["a"], vocab["dog"], vocab["."]]
    ]
    
    epochs = 20
    print("[*] Training LM with Predictive Coding (No BP)...")
    
    for epoch in range(epochs):
        total_error = 0
        for sentence in corpus:
            # 文章ごとに内部の膜電位や発火履歴をリセット（文脈を区切る）
            model.reset_state()
            
            # 時系列データとして1トークンずつ入力
            for token in sentence:
                # SpikingPredictiveLM の forward は誤差スパイクを返す
                error_spikes = model.forward([token], learning=True)
                total_error += len(error_spikes)
                
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} - Total Error Spikes: {total_error}")
            
    print("\n[*] Evaluation: Autoregressive Generation")
    
    # テスト1: "I am" からの生成 ("a cat ." と続くか)
    model.reset_state()
    prompt1 = [vocab["I"], vocab["am"]]
    print(f"\nPrompt 1: {[reverse_vocab[t] for t in prompt1]}")
    gen1 = model.generate(prompt1, max_length=4)
    print(f"Generated 1: {[reverse_vocab[t] for t in gen1]}")
    
    # テスト2: "You are" からの生成 ("a dog ." と続くか)
    model.reset_state()
    prompt2 = [vocab["You"], vocab["are"]]
    print(f"\nPrompt 2: {[reverse_vocab[t] for t in prompt2]}")
    gen2 = model.generate(prompt2, max_length=4)
    print(f"Generated 2: {[reverse_vocab[t] for t in gen2]}")

if __name__ == "__main__":
    run_lm_benchmark()