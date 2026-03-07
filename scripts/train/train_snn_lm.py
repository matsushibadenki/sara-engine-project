from sara_engine.utils.tokenizer import SaraTokenizer
from sara_engine.utils.corpus import (
    clean_corpus_lines,
    load_chat_jsonl_pairs,
    clean_chat_pairs,
    build_definition_qa_pairs,
    is_low_quality_response,
)
from sara_engine.models.snn_transformer import SpikingTransformerModel, SNNTransformerConfig
from typing import List, Tuple
import argparse
import random
import time
import sys
import os
import json
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


def _load_document_lines(corpus_path: str) -> List[str]:
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_lines = f.read().splitlines()
    cleaned_lines = clean_corpus_lines(raw_lines, merge_wrapped=True)
    filtered_lines: List[str] = []
    seen: set[str] = set()
    for line in cleaned_lines:
        text = line.strip()
        if len(text) < 16 or len(text) > 180:
            continue
        if is_low_quality_response(text):
            continue
        if text in seen:
            continue
        seen.add(text)
        filtered_lines.append(text)
    return filtered_lines


def _load_chat_pairs(chat_path: str, document_lines: List[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    pairs.extend(_seed_chat_pairs())
    if os.path.exists(chat_path):
        pairs.extend(load_chat_jsonl_pairs(chat_path))
    pairs.extend(build_definition_qa_pairs(document_lines))
    return clean_chat_pairs(pairs)


def _seed_chat_pairs() -> List[Tuple[str, str]]:
    return [
        ("ニューラルネットワークとは何ですか？", "ニューラルネットワークは、重み付き結合でつながった計算ユニットからなる数理モデルです。学習では重みを調整し、分類、回帰、生成などの課題に対応します。"),
        ("ディープラーニングとは何ですか？", "ディープラーニングは、多層のニューラルネットワークを用いて特徴表現を段階的に学習する手法です。画像、音声、自然言語処理で広く使われます。"),
        ("パーセプトロンとは何ですか？", "パーセプトロンは、入力の重み付き和から出力を決める基本的なニューラルネットワークです。単純パーセプトロンは線形分離できる問題に適しています。"),
        ("排他的論理和とは何ですか？", "排他的論理和(XOR)は、入力が異なるときだけ真になる論理演算です。単純パーセプトロンでは線形分離できないため、多層ネットワークが必要です。"),
        ("ヒトの神経系について教えてください。", "ヒトの神経系は、ニューロンがシナプスでつながるネットワークです。樹状突起が入力を受け取り、細胞体で処理し、軸索を通じて他の細胞へ信号を伝えます。"),
        ("シナプスとは何ですか？", "シナプスは、ニューロン同士が情報を受け渡す接合部です。生物の学習では、シナプス結合の強さが変化することが重要な役割を持ちます。"),
        ("学習とは何ですか？", "ニューラルネットワークの学習とは、予測誤差が小さくなるように重みやしきい値を調整することです。"),
        ("推論とは何ですか？", "推論は、学習済みの重みを用いて新しい入力から出力を計算する処理です。"),
        ("バックプロパゲーションとは何ですか？", "バックプロパゲーションは、出力誤差を各層へ逆向きに伝えて重みを更新する学習法です。深層学習を実用化した重要な技術です。"),
        ("CNNとは何ですか？", "CNNは畳み込みニューラルネットワークのことで、局所受容野と重み共有を使って画像の特徴を効率よく抽出します。"),
        ("RNNとは何ですか？", "RNNはリカレントニューラルネットワークのことで、過去の状態を保持しながら時系列データを処理します。"),
        ("Transformerとは何ですか？", "Transformerは自己注意機構を用いて系列内の関係を並列に扱うモデルです。現在の大規模言語モデルの中心的な構造です。"),
        ("スパイキングニューラルネットワークとは何ですか？", "スパイキングニューラルネットワークは、ニューロンの発火タイミングを情報として扱う神経回路モデルです。通常のニューラルネットワークより生物学的な挙動に近い特徴があります。"),
        ("重みとは何ですか？", "重みは、ニューラルネットワークにおいて入力信号の強さを調整するパラメータです。学習によって更新され、モデルの振る舞いを決めます。"),
        ("活性化関数とは何ですか？", "活性化関数は、ニューロンへの入力を非線形に変換して出力を決める関数です。モデルに表現力を与えるために重要です。"),
    ]


def _build_training_texts(
    document_lines: List[str],
    chat_pairs: List[Tuple[str, str]],
) -> List[str]:
    texts: List[str] = []

    for line in document_lines:
        texts.append(line)

    for prompt, response in chat_pairs:
        texts.append(f"User: {prompt}\nSARA: {response}")
        texts.append(f"質問: {prompt}\n回答: {response}")
        if len(prompt) <= 32:
            texts.append(f"{prompt}\n{response}")

    return texts


def _build_token_sequences(
    tokenizer: SaraTokenizer,
    document_lines: List[str],
    chat_pairs: List[Tuple[str, str]],
    eos_id: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    doc_sequences: List[List[int]] = []
    chat_sequences: List[List[int]] = []

    for line in document_lines:
        token_ids = tokenizer.encode(line)
        if token_ids:
            doc_sequences.append(token_ids + [eos_id])

    for prompt, response in chat_pairs:
        variants = [
            f"User: {prompt}\nSARA: {response}",
            f"質問: {prompt}\n回答: {response}",
        ]
        if len(prompt) <= 32:
            variants.append(f"{prompt}\n{response}")
        for text in variants:
            token_ids = tokenizer.encode(text)
            if token_ids:
                chat_sequences.append(token_ids + [eos_id])

    return doc_sequences, chat_sequences


def _export_clean_chat_dataset(pairs: List[Tuple[str, str]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for prompt, response in pairs:
            f.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")


def train_snn_language_model(
    corpus_path: str,
    save_dir: str,
    chat_path: str = "data/raw/chat_data.jsonl",
    epochs: int = 3,
    chunk_size: int = 96,
    stride: int = 24,
    learn_epochs_per_chunk: int = 2,
    chat_weight: int = 3,
):
    print("=" * 60)
    print("SARA-Engine: SNN Language Model Pre-training (Subword-Level)")
    print("=" * 60)

    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found at {corpus_path}")
        return

    tokenizer = SaraTokenizer(
        vocab_size=4096,
        model_path=os.path.join(save_dir, "sara_vocab.json"),
    )
    document_lines = _load_document_lines(corpus_path)
    if not document_lines:
        print("Error: No usable lines after cleaning.")
        return

    chat_pairs = _load_chat_pairs(chat_path, document_lines)
    training_texts = _build_training_texts(document_lines, chat_pairs)
    print(f"Usable document lines: {len(document_lines)}")
    print(f"Usable chat pairs: {len(chat_pairs)}")

    print("Training tokenizer...")
    tokenizer.train(training_texts)

    eos_id = tokenizer.vocab.get("<eos>", 3)
    doc_sequences, chat_sequences = _build_token_sequences(
        tokenizer, document_lines, chat_pairs, eos_id
    )

    if not doc_sequences and not chat_sequences:
        print("Error: No tokenized sequences created.")
        return

    weighted_chat_sequences = chat_sequences * max(1, chat_weight)
    all_sequences = doc_sequences + weighted_chat_sequences
    total_tokens = sum(len(seq) for seq in all_sequences)
    print(f"Total tokens to process: {total_tokens}")
    print(f"Document sequences: {len(doc_sequences)}")
    print(f"Chat sequences (weighted): {len(weighted_chat_sequences)}")

    config = SNNTransformerConfig(vocab_size=tokenizer.vocab_size)
    model = SpikingTransformerModel(config)

    chunks = _build_chunks(all_sequences, chunk_size=chunk_size, stride=stride)
    if not chunks:
        print("Error: No training chunks created.")
        return

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
    _export_clean_chat_dataset(chat_pairs, os.path.join(save_dir, "clean_chat_data.jsonl"))
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SNN language model with cleaner corpus chunking.")
    parser.add_argument(
        "--corpus", default="data/processed/corpus.txt", help="Path to corpus text file.")
    parser.add_argument(
        "--chat-data", default="data/raw/chat_data.jsonl", help="Path to chat pair jsonl.")
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
    parser.add_argument("--chat-weight", type=int, default=3,
                        help="Oversampling factor for conversational sequences.")
    args = parser.parse_args()

    train_snn_language_model(
        corpus_path=args.corpus,
        save_dir=args.save_dir,
        chat_path=args.chat_data,
        epochs=max(1, args.epochs),
        chunk_size=max(16, args.chunk_size),
        stride=max(4, args.stride),
        learn_epochs_per_chunk=max(1, args.learn_epochs),
        chat_weight=max(1, args.chat_weight),
    )
