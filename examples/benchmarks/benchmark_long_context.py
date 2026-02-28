_FILE_INFO = {
    "//1": "ディレクトリパス: examples/benchmark_long_context.py",
    "//2": "ファイルの日本語タイトル: LIF長文脈推論ベンチマーク",
    "//3": "ファイルの目的や内容: LIF Attentionにより、数十トークン前に出現した情報（主語）を膜電位として保持し、現在のトークン（動詞など）と結びつけて推論できるかを検証する。"
}

import os
from sara_engine.models.spiking_llm import SpikingLLM

def main():
    print("[INFO] Starting Long Context Benchmark (LIF Attention)...")
    
    # モデルの初期化 (LIF Attention と MoE が組み込まれている)
    model = SpikingLLM(num_layers=2, sdr_size=256, vocab_size=5000)
    
    # ダミーの辞書 (簡易的なトークナイザー)
    vocab = {"<pad>": 0, "アリス": 1, "は": 2, "とても": 3, "長い": 4, "森": 5, "の": 6, "中": 7, 
             "を": 8, "歩い": 9, "て": 10, "い": 11, "まし": 12, "た": 13, "。": 14, 
             "彼女": 15, "名前": 16}
    inv_vocab = {v: k for k, v in vocab.items()}
    
    # 長い文脈の学習データを作成
    # 意図: 最初に「アリス」という情報があり、間にノイズ（森の中を歩く描写）が長く続く。
    # 最後に「彼女 の 名前 は [アリス]」と予測させたい。
    noise_sequence = ["とても", "長い", "森", "の", "中", "を", "歩い", "て", "い", "まし", "た", "。"] * 3
    
    text_sequence = ["アリス", "は"] + noise_sequence + ["彼女", "の", "名前", "は", "アリス"]
    token_ids = [vocab.get(word, 0) for word in text_sequence]
    
    print("\n[TRAINING] Learning sequence with long context...")
    print(f"Sequence length: {len(token_ids)} tokens")
    
    # 数回学習させて強固なシナプスを形成する
    for _ in range(5):
        model.learn_sequence(token_ids)
        
    print("[TRAINING] Done.")
    
    # 推論テスト: 「彼女 の 名前 は」の続きを予測させる
    prompt_text = ["アリス", "は"] + noise_sequence + ["彼女", "の", "名前", "は"]
    prompt_ids = [vocab.get(word, 0) for word in prompt_text]
    
    print("\n[INFERENCE] Predicting next token...")
    # プロンプトを入力して次トークンを推論
    generated_ids = model.generate(prompt_tokens=prompt_ids, max_new_tokens=1, top_k=1, temperature=0.1)
    
    if generated_ids:
        predicted_word = inv_vocab.get(generated_ids[0], "<unknown>")
        print(f"Prompt end: ... {' '.join(prompt_text[-4:])}")
        print(f"Predicted Token: {predicted_word}")
        
        if predicted_word == "アリス":
            print("\n[SUCCESS] Model successfully retained context across the long sequence using LIF potentials!")
        else:
            print(f"\n[FAILED] Model predicted '{predicted_word}' instead of 'アリス'. Context was lost.")
    else:
        print("\n[FAILED] Model did not generate any tokens.")

if __name__ == "__main__":
    main()