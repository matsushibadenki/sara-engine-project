# ディレクトリパス: scripts/train/train_snn_lm.py
# ファイルの日本語タイトル: SNN言語モデル 実データ事前学習スクリプト
# ファイルの目的や内容: コーパスの末尾への過学習（Recency Bias）を防ぐため、テキストをチャンクに分割しランダムにシャッフルして学習（Experience Replay）するように修正。

import os
import sys
import time
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig

def train_snn_language_model(corpus_path: str, save_dir: str, epochs: int = 3):
    print("=" * 60)
    print("SARA-Engine: SNN Language Model Pre-training (Character-Level)")
    print("=" * 60)
    
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found at {corpus_path}")
        return

    config = SNNTransformerConfig(vocab_size=1114112)
    model = SpikingTransformerModel(config)
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    encoded_tokens = [ord(c) for c in text]
    total_tokens = len(encoded_tokens)
    print(f"Total tokens to process: {total_tokens}")

    # 修正箇所: コーパスを一筆書きで学習するのではなく、チャンクに分割する
    # オーバーラップ（stride）させて文脈の繋がりを維持する
    chunk_size = 64
    stride = 32
    chunks = []
    for i in range(0, total_tokens - chunk_size, stride):
        chunks.append(encoded_tokens[i:i + chunk_size])
    
    print(f"Total chunks created: {len(chunks)}")

    start_time = time.time()
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        # 修正箇所: コーパス末尾への過学習を防ぐため、チャンクをランダムにシャッフルして学習
        random.shuffle(chunks)
        
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            # チャンクごとに文脈をリセットして学習
            model.learn_sequence(chunk)
            
            if (i + 1) % 100 == 0 or i == total_chunks - 1:
                progress = ((i + 1) / total_chunks) * 100
                print(f"Progress: {progress:.1f}% ({i + 1}/{total_chunks} chunks)")

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds.")
    model.save_pretrained(save_dir)
    print("=" * 60)

if __name__ == "__main__":
    CORPUS_FILE = "data/processed/corpus.txt"
    SAVE_DIRECTORY = "models/snn_lm_pretrained"
    train_snn_language_model(corpus_path=CORPUS_FILE, save_dir=SAVE_DIRECTORY, epochs=3)