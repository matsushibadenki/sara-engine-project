# ディレクトリパス: scripts/train/train_snn_lm.py
# ファイルの日本語タイトル: SNN言語モデル 実データ事前学習スクリプト
# ファイルの目的や内容: バイト単位の学習による文字化けを防ぐため、Unicode文字単位（コードポイント）での学習に修正。ストリーム学習に対応。

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig

def train_snn_language_model(corpus_path: str, save_dir: str, epochs: int = 1):
    print("=" * 60)
    print("SARA-Engine: SNN Language Model Pre-training (Character-Level)")
    print("=" * 60)
    
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found at {corpus_path}")
        return

    config = SNNTransformerConfig(vocab_size=1114112) # Unicodeの最大値まで許容
    model = SpikingTransformerModel(config)
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # バイトではなく文字のID（Unicodeコードポイント）に変換
    encoded_tokens = [ord(c) for c in text]
    total_tokens = len(encoded_tokens)
    print(f"Total tokens to process: {total_tokens}")

    start_time = time.time()
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        model.reset_state()
        
        for i in range(total_tokens - 1):
            curr_token = encoded_tokens[i]
            target_token = encoded_tokens[i + 1]
            
            # 修正箇所: チャンクごとにリセットするのではなく、1文字ずつ文脈を維持してストリーム学習
            model.forward_step(curr_token, learning=True, target_id=target_token)
            
            if i > 0 and i % 5000 == 0:
                progress = (i / total_tokens) * 100
                print(f"Progress: {progress:.1f}% ({i}/{total_tokens} tokens)")

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds.")
    model.save_pretrained(save_dir)
    print("=" * 60)

if __name__ == "__main__":
    CORPUS_FILE = "data/processed/corpus.txt"
    SAVE_DIRECTORY = "models/snn_lm_pretrained"
    train_snn_language_model(corpus_path=CORPUS_FILE, save_dir=SAVE_DIRECTORY)