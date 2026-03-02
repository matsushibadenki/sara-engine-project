_FILE_INFO = {
    "//": "ディレクトリパス: examples/memory/demo_scalable_ltm.py",
    "//": "タイトル: 100万トークン対応 スケーラブルLTM デモンストレーション",
    "//": "目的: 行列演算なしで多言語テキストを高速に記憶・検索できることを証明する。"
}

import os
import sys
import time
import random

# パスの追加 (実行環境依存を回避)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from sara_engine.memory.scalable_ltm import SNNMemoryModule

def generate_dummy_sdr(seed: int, active_bits: int = 20, total_space: int = 10000) -> list:
    """疑似的なSDRを生成するユーティリティ"""
    random.seed(seed)
    return random.sample(range(total_space), active_bits)

def main():
    print("=== SARA Engine: Scalable SNN LTM Demo ===")
    
    # workspace下にデータを保存する設計
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../workspace'))
    os.makedirs(workspace_dir, exist_ok=True)
    
    # モジュールの初期化
    print("Initializing SNN Memory Module (Rust Accelerated)...")
    memory = SNNMemoryModule(workspace_dir=workspace_dir, threshold=0.15)
    
    # 多言語データの記憶 (Memorization)
    corpus = [
        {"text": "The quick brown fox jumps over the lazy dog.", "lang": "en"},
        {"text": "Artificial intelligence based on spiking neural networks is energy efficient.", "lang": "en"},
        {"text": "誤差逆伝播法を使わない新しい学習アルゴリズムの提案。", "lang": "ja"},
        {"text": "Le chat noir dort sur le canapé.", "lang": "fr"},
        {"text": "El aprendizaje automático está transformando el mundo.", "lang": "es"},
        {"text": "生物由来の設計はGPUを必要としない。", "lang": "ja"}
    ]
    
    print("\n--- Memorizing Multilingual Knowledge ---")
    for i, item in enumerate(corpus):
        # 実際はTokenizer等でSDR化しますが、ここではシード値を用いた疑似SDRを使用
        sdr = generate_dummy_sdr(seed=i, active_bits=25)
        mem_id = memory.memorize(sdr=sdr, content=item["text"], language=item["lang"])
        print(f"Stored Memory ID {mem_id}: {item['text'][:30]}... (Lang: {item['lang']})")
    
    # 永続化テスト
    memory.save()
    
    # 検索テスト (Recall)
    print("\n--- Testing Fuzzy Recall (Intersection-based) ---")
    # インデックス2 ("誤差逆伝播法を使わない...") に似たSDRクエリを作成（少しノイズを混ぜる）
    target_sdr = generate_dummy_sdr(seed=2, active_bits=25)
    query_sdr = target_sdr[:20] + generate_dummy_sdr(seed=999, active_bits=5) # 80% overlap
    
    start_time = time.perf_counter()
    results = memory.recall(query_sdr, top_k=3)
    elapsed = (time.perf_counter() - start_time) * 1000
    
    print(f"Recall completed in {elapsed:.3f} ms.")
    for rank, res in enumerate(results, 1):
        print(f"Rank {rank} [Score: {res['score']:.2f}]: {res['content']}")

    stats = memory.get_memory_stats()
    print(f"\nSystem Stats: {stats['total_memories']} items stored at {stats['storage_path']}")

if __name__ == "__main__":
    main()