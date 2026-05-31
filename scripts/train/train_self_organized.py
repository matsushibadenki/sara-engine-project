# ファイルの日本語タイトル: 自己組織化学習スクリプト (SNN固有機能ベース)
# ファイルの目的や内容: 外部のLLM（教師モデル）を使わず、STDPや予測符号化といったSNN固有のダイナミクスを用いてテキストコーパスから自律的にオンライン学習する。
# {
#     "//": "古い形式のモデルファイルをロードする処理を整理し、fit()による事前学習を正しく実行するように修正"
# }

import argparse

from sara_engine.utils.project_paths import ensure_output_directory, model_path, processed_data_path, project_path, resolve_project_relative
from sara_engine.models.spiking_llm import SpikingLLM
import os
import sys
import time
import tqdm

# プロジェクトルートのsrcをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def train_self_organized(corpus_path, save_dir, vocab_size=65536, sdr_size=128, context_window=15):
    print("="*50)
    print("🚀 Starting Self-Organized SNN Training (No Backpropagation)")
    print("="*50)

    # 1. 保存ディレクトリの準備とモデルの初期化
    save_dir = ensure_output_directory(save_dir)
    print(
        f"[INFO] Initializing SpikingLLM (SDR={sdr_size}, Vocab={vocab_size})...")
    llm = SpikingLLM(sdr_size=sdr_size, vocab_size=vocab_size,
                     context_window=context_window)

    # 既存の記憶（モデル）があれば読み込む
    model_file_path = os.path.join(save_dir, "spiking_llm_weights.json")
    if os.path.exists(model_file_path):
        try:
            print(f"[INFO] Found existing memory at {save_dir}. Restoring...")
            if hasattr(SpikingLLM, 'from_pretrained'):
                llm = SpikingLLM.from_pretrained(save_dir)
                print("[INFO] Successfully restored previous state.")
        except Exception as e:
            print(
                f"[WARNING] Failed to load existing model: {e}. Starting fresh.")

    # 2. コーパスの読み込み
    if not os.path.exists(corpus_path):
        print(f"[ERROR] Corpus file not found at {corpus_path}")
        return

    print(f"[INFO] Reading corpus from {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        full_text = f.read()

    if not full_text.strip():
        print("[WARNING] The corpus file is empty.")
        return

    # 3. Direct Wiring (Rustコアによる高速な長期記憶の構築)
    print(f"[INFO] Building Direct Synaptic Wiring (Long-term Memory)...")
    start_time = time.time()
    if hasattr(llm, 'fit'):
        llm.fit(full_text)

    # 4. SDR/STDP によるオンライン動的記憶の学習
    lines = [line.strip() for line in full_text.split('\n') if line.strip()]
    print(
        f"[INFO] Learning {len(lines)} sequences via STDP / Local Plasticity...")

    checkpoint_interval = max(1, len(lines) // 10)

    try:
        for i, line in enumerate(tqdm.tqdm(lines, desc="Self-Organizing")):
            tokens = llm.encode_text(line)

            if len(tokens) > 1 and hasattr(llm, 'learn_sequence'):
                llm.learn_sequence(tokens)

            if (i + 1) % checkpoint_interval == 0:
                if hasattr(llm, 'save_pretrained'):
                    llm.save_pretrained(save_dir)

    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted by user. Saving current progress...")

    # 5. 最終保存とシナプス整理
    print("[INFO] Saving final memory state...")
    if hasattr(llm, 'save_pretrained'):
        llm.save_pretrained(save_dir)

    if hasattr(llm, 'prune_synapses'):
        print("[INFO] Pruning weak synapses for memory efficiency...")
        llm.prune_synapses(threshold=0.01)
        llm.save_pretrained(save_dir)

    elapsed = time.time() - start_time
    print(f"✨ Self-Organized Training completed in {elapsed:.2f} seconds.")


def _default_corpus_path() -> str:
    processed_corpus = processed_data_path("corpus.txt")
    if os.path.exists(processed_corpus):
        return processed_corpus
    return project_path("data", "corpus.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the self-organized SpikingLLM without backpropagation.")
    parser.add_argument("--corpus", default=_default_corpus_path(), help="Corpus file path.")
    parser.add_argument("--save-dir", default=model_path("self_organized_llm"), help="Managed model output directory.")
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--sdr-size", type=int, default=128)
    parser.add_argument("--context-window", type=int, default=15)
    args = parser.parse_args()

    corpus_file = resolve_project_relative(str(args.corpus))
    model_dir = resolve_project_relative(str(args.save_dir))

    train_self_organized(
        corpus_path=corpus_file,
        save_dir=model_dir,
        vocab_size=int(args.vocab_size),
        sdr_size=int(args.sdr_size),
        context_window=int(args.context_window)
    )
