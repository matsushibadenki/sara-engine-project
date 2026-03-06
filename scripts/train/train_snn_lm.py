# {
#     "//": "ディレクトリパス: scripts/train/train_snn_lm.py",
#     "//": "ファイルの日本語タイトル: SNN言語モデル 実データ事前学習スクリプト",
#     "//": "ファイルの目的や内容: SARA BPE トークナイザーを使用してサブワードレベルで学習。文脈を長く保つためにチャンクサイズを128に拡大。"
# }

import os
import sys
import time
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig
from src.sara_engine.utils.tokenizer import SaraTokenizer

def train_snn_language_model(corpus_path: str, save_dir: str, epochs: int = 1):
    print("=" * 60)
    print("SARA-Engine: SNN Language Model Pre-training (Subword-Level)")
    print("=" * 60)
    
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found at {corpus_path}")
        return

    tokenizer = SaraTokenizer(vocab_size=4096)
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print("Training tokenizer...")
    tokenizer.train([text])
    
    encoded_tokens = tokenizer.encode(text)
    total_tokens = len(encoded_tokens)
    print(f"Total tokens to process: {total_tokens}")

    config = SNNTransformerConfig(vocab_size=tokenizer.vocab_size)
    model = SpikingTransformerModel(config)

    # 修正箇所: チャンクサイズを広げて、文脈の繋がりを長く学習させる
    chunk_size = 128
    stride = 32
    chunks = []
    for i in range(0, total_tokens - chunk_size, stride):
        chunks.append(encoded_tokens[i:i + chunk_size])
    
    print(f"Total chunks created: {len(chunks)}")

    start_time = time.time()
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        random.shuffle(chunks)
        
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            model.learn_sequence(chunk)
            
            if (i + 1) % 100 == 0 or i == total_chunks - 1:
                progress = ((i + 1) / total_chunks) * 100
                print(f"Progress: {progress:.1f}% ({i + 1}/{total_chunks} chunks)")

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds.")
    
    model.save_pretrained(save_dir)
    tokenizer.model_path = os.path.join(save_dir, "sara_vocab.json")
    tokenizer.save()
    print("=" * 60)

if __name__ == "__main__":
    CORPUS_FILE = "data/processed/corpus.txt"
    SAVE_DIRECTORY = "models/snn_lm_pretrained"
    train_snn_language_model(corpus_path=CORPUS_FILE, save_dir=SAVE_DIRECTORY, epochs=1)