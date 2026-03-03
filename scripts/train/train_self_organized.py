{
    "//": "ディレクトリパス: scripts/train/train_self_organized.py",
    "//": "ファイルの日本語タイトル: 自己組織化学習スクリプト (SNN固有機能ベース)",
    "//": "ファイルの目的や内容: 外部のLLM（教師モデル）を使わず、STDPや予測符号化といったSNN固有のダイナミクスを用いてテキストコーパスから自律的にオンライン学習する。"
}

import os
import sys
import time
import tqdm

# プロジェクトルートのsrcをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sara_engine.models.spiking_llm import SpikingLLM

def train_self_organized(corpus_path, save_dir, vocab_size=65536, sdr_size=128, context_window=15):
    print("="*50)
    print("🚀 Starting Self-Organized SNN Training (No Backpropagation)")
    print("="*50)
    
    # 1. 保存ディレクトリの準備とモデルの初期化
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Initializing SpikingLLM (SDR={sdr_size}, Vocab={vocab_size})...")
    llm = SpikingLLM(sdr_size=sdr_size, vocab_size=vocab_size, context_window=context_window)
    
    # 既存の記憶（モデル）があれば読み込む
    if os.path.exists(os.path.join(save_dir, "model.msgpack")) or os.path.exists(os.path.join(save_dir, "config.json")):
        try:
            print(f"[INFO] Found existing memory at {save_dir}. Restoring...")
            if hasattr(llm, 'load_pretrained'):
                llm.load_pretrained(save_dir)
                print("[INFO] Successfully restored previous state.")
            else:
                print("[WARNING] 'load_pretrained' method not found in SpikingLLM. Starting fresh.")
        except Exception as e:
            print(f"[WARNING] Failed to load existing model: {e}. Starting fresh.")

    # 2. コーパスの読み込み
    if not os.path.exists(corpus_path):
        print(f"[ERROR] Corpus file not found at {corpus_path}")
        return

    print(f"[INFO] Reading corpus from {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        # 大規模データでも処理しやすいように行単位で読み込む
        lines = f.readlines()

    if not lines:
        print("[WARNING] The corpus file is empty.")
        return

    # 3. 自律学習 (Predictive Coding / STDP Sequence Learning)
    print(f"[INFO] Learning {len(lines)} sequences via STDP / Local Plasticity...")
    start_time = time.time()
    
    checkpoint_interval = max(1, len(lines) // 10) # 10%ごとにチェックポイント保存
    
    try:
        for i, line in enumerate(tqdm.tqdm(lines, desc="Self-Organizing")):
            line = line.strip()
            if not line:
                continue
                
            # a. テキストをSNN用のトークンにエンコード
            tokens = llm.encode_text(line)
            
            # b. SNN独自の系列学習（時間差に基づくシナプス更新）
            # ここで誤差逆伝播法を使わず、ローカルな発火タイミングから因果関係を学習します
            if len(tokens) > 1:
                if hasattr(llm, 'learn_sequence'):
                    llm.learn_sequence(tokens)
                elif hasattr(llm, 'fit'):
                    # フォールバック: fitメソッドが単一文字列を受け取る場合
                    llm.fit(line)
                
            # c. 定期的なチェックポイント保存（メモリ揮発やクラッシュ対策）
            if (i + 1) % checkpoint_interval == 0:
                if hasattr(llm, 'save_pretrained'):
                    llm.save_pretrained(save_dir)
                    
    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted by user. Saving current progress...")
    
    # 4. 最終保存とシナプス整理
    print("[INFO] Saving final memory state...")
    if hasattr(llm, 'save_pretrained'):
        llm.save_pretrained(save_dir)
        
    # もし Synaptic Pruning（忘却によるメモリ最適化）メソッドが実装されていれば実行する
    if hasattr(llm, 'prune_synapses'):
        print("[INFO] Pruning weak synapses for memory efficiency...")
        llm.prune_synapses(threshold=0.01)
        llm.save_pretrained(save_dir)

    elapsed = time.time() - start_time
    print(f"✨ Self-Organized Training completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    # デフォルトのパス設定
    corpus_file = os.path.join(project_root, "data/processed/corpus.txt")
    model_dir = os.path.join(project_root, "workspace/models/self_organized_llm")
    
    train_self_organized(
        corpus_path=corpus_file, 
        save_dir=model_dir,
        vocab_size=65536,
        sdr_size=128,
        context_window=15
    )