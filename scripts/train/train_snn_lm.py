# ディレクトリパス: scripts/train/train_snn_lm.py
# ファイルの日本語タイトル: SNN言語モデル 実データ事前学習スクリプト
# ファイルの目的や内容: バイト単位の学習による文字化けを防ぐため、Unicode文字単位（コードポイント）での学習に修正。

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig

def train_snn_language_model(corpus_path: str, save_dir: str, chunk_size: int = 256, epochs: int = 1):
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
    
    # 修正箇所: バイトではなく文字のID（Unicodeコードポイント）に変換
    encoded_tokens = [ord(c) for c in text]
    total_tokens = len(encoded_tokens)
    print(f"Total tokens to process: {total_tokens}")

    start_time = time.time()
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        model.reset_state()
        processed = 0
        while processed < total_tokens - 1:
            chunk = encoded_tokens[processed : processed + chunk_size]
            if len(chunk) < 2:
                break
            model.learn_sequence(chunk)
            processed += len(chunk)
            if processed % (chunk_size * 10) == 0 or processed >= total_tokens - 1:
                progress = (processed / total_tokens) * 100
                print(f"Progress: {progress:.1f}% ({processed}/{total_tokens} tokens)")

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds.")
    model.save_pretrained(save_dir)
    print("=" * 60)

if __name__ == "__main__":
    CORPUS_FILE = "data/processed/corpus.txt"
    SAVE_DIRECTORY = "data/models/snn_lm_pretrained"
    train_snn_language_model(corpus_path=CORPUS_FILE, save_dir=SAVE_DIRECTORY, chunk_size=512)