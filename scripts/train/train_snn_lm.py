from sara_engine.utils.tokenizer import SaraTokenizer
from sara_engine.utils.corpus import clean_corpus_lines
from sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig
from typing import List
import argparse
import random
import time
import sys
import os
# ディレクトリパス: scripts/train/train_snn_lm.py
# ファイルの日本語タイトル: SNN言語モデル 実データ事前学習スクリプト
# ファイルの目的や内容: SARA BPE トークナイザーを使用してサブワードレベルで学習。ノイズ（英語論文の断片など）を除外するため、日本語含有率のチェックを追加。


sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'src')))


def _build_chunks(
    token_sequences: List[List[int]],
    chunk_size: int,
    stride: int,
) -> List[List[int]]:
    chunks: List[List[int]] = []
    for seq in token_sequences:
        if len(seq) < 2:
            continue

        if len(seq) <= chunk_size:
            chunks.append(seq)
            continue

        last_start = len(seq) - chunk_size
        for start in range(0, last_start + 1, stride):
            chunks.append(seq[start:start + chunk_size])

        if last_start % stride != 0:
            chunks.append(seq[-chunk_size:])

    return chunks


def train_snn_language_model(
    corpus_path: str,
    save_dir: str,
    epochs: int = 3,
    chunk_size: int = 96,
    stride: int = 24,
    learn_epochs_per_chunk: int = 2,
):
    print("=" * 60)
    print("SARA-Engine: SNN Language Model Pre-training (Subword-Level)")
    print("=" * 60)

    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found at {corpus_path}")
        return

    tokenizer = SaraTokenizer(vocab_size=4096)
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_lines = f.read().splitlines()

    cleaned_lines = clean_corpus_lines(raw_lines, merge_wrapped=True)
    if not cleaned_lines:
        print("Error: No usable lines after cleaning.")
        return

    text = "\n".join(cleaned_lines)
    print(f"Usable lines: {len(cleaned_lines)}/{len(raw_lines)}")

    print("Training tokenizer...")
    tokenizer.train([text])

    eos_id = tokenizer.vocab.get("<eos>", 3)
    token_sequences: List[List[int]] = []
    total_tokens = 0
    for line in cleaned_lines:
        token_ids = tokenizer.encode(line)
        if not token_ids:
            continue
        sequence = token_ids + [eos_id]
        token_sequences.append(sequence)
        total_tokens += len(sequence)

    if not token_sequences:
        print("Error: No tokenized sequences created.")
        return

    print(f"Total tokens to process: {total_tokens}")

    config = SNNTransformerConfig(vocab_size=tokenizer.vocab_size)
    model = SpikingTransformerModel(config)

    chunks = _build_chunks(
        token_sequences, chunk_size=chunk_size, stride=stride)

    print(f"Total chunks created: {len(chunks)}")

    start_time = time.time()
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        random.shuffle(chunks)

        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            model.learn_sequence(chunk, epochs=learn_epochs_per_chunk)

            if (i + 1) % 100 == 0 or i == total_chunks - 1:
                progress = ((i + 1) / total_chunks) * 100
                print(
                    f"Progress: {progress:.1f}% ({i + 1}/{total_chunks} chunks)")

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds.")

    model.save_pretrained(save_dir)
    tokenizer.model_path = os.path.join(save_dir, "sara_vocab.json")
    tokenizer.save()
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SNN language model with cleaner corpus chunking.")
    parser.add_argument(
        "--corpus", default="data/processed/corpus.txt", help="Path to corpus text file.")
    parser.add_argument("--save-dir", default="models/snn_lm_pretrained",
                        help="Output directory for model/tokenizer.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of global training epochs.")
    parser.add_argument("--chunk-size", type=int,
                        default=96, help="Token chunk length.")
    parser.add_argument("--stride", type=int, default=24,
                        help="Stride for chunk sampling.")
    parser.add_argument("--learn-epochs", type=int, default=2,
                        help="Internal model.learn_sequence epochs per chunk.")
    args = parser.parse_args()

    train_snn_language_model(
        corpus_path=args.corpus,
        save_dir=args.save_dir,
        epochs=max(1, args.epochs),
        chunk_size=max(16, args.chunk_size),
        stride=max(4, args.stride),
        learn_epochs_per_chunk=max(1, args.learn_epochs),
    )
