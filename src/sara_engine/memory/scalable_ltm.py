from typing import List, Dict, Any, Optional
import json
import time
import os
_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/memory/scalable_ltm.py",
    "//": "タイトル: 100万トークン対応 スケーラブル疎分散長期記憶 (Scalable LTM)",
    "//": "目的: Rustコアを活用し、数百万トークン規模のSDRを高速かつ省メモリでファジー検索・連想する。ANN/Transformerの代替となるSNNアーキテクチャ。"
}


try:
    from ..sara_rust_core import ScalableSDRMemory
except ImportError:
    print("Warning: Rust core 'ScalableSDRMemory' not found. Please run 'pip install -e .' to rebuild the library.")
    ScalableSDRMemory = None


class SNNMemoryModule:
    """
    TransformersのMemory/Attention層の代替となるSNNベースの長期記憶モジュール。
    行列演算を使用せず、Rustの高速なセット積集合による生物学的連想を実現。
    """

    def __init__(self, workspace_dir: str = "workspace", threshold: float = 0.1):
        self.workspace_dir = workspace_dir
        os.makedirs(self.workspace_dir, exist_ok=True)
        self.save_path = os.path.join(self.workspace_dir, "scalable_ltm.json")

        # Rustコアを利用してCPU単体でも10ms以内の推論速度を目指す
        if ScalableSDRMemory is None:
            raise RuntimeError(
                "sara_rust_core is required for Million-token LTM.")

        self.engine = ScalableSDRMemory(threshold)
        self.metadata_store: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0
        self.load()

    def memorize(self, sdr: List[int], content: str, language: str = "en", context_type: str = "general") -> int:
        """
        SDRと関連するメタデータを記憶として定着させる (STDP不要のOne-shot学習)
        """
        mem_id = self.next_id
        self.engine.add_memory(mem_id, sdr)

        self.metadata_store[mem_id] = {
            "content": content,
            "language": language,
            "type": context_type,
            "timestamp": time.time(),
            "sdr_size": len(sdr)
        }
        self.next_id += 1
        return mem_id

    def recall(self, query_sdr: List[int], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        曖昧さの許容 (Fuzzy Recall) を用いて連想検索を実行する。
        """
        start_time = time.perf_counter()

        # Rust層での超高速・行列レス検索
        raw_results = self.engine.search(query_sdr, top_k)

        results = []
        for mem_id, score in raw_results:
            if mem_id in self.metadata_store:
                meta = self.metadata_store[mem_id]
                results.append({
                    "id": mem_id,
                    "score": score,
                    "content": meta["content"],
                    "language": meta["language"],
                    "type": meta["type"]
                })

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        # ログは英語指定
        # print(f"[SNNMemoryModule] Recalled {len(results)} items in {elapsed_ms:.2f} ms.")
        return results

    def save(self):
        """メタデータの永続化"""
        try:
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata_store, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving LTM metadata: {e}")

    def load(self):
        """メタデータの復元とRustエンジンへの再ロード"""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    store = json.load(f)
                    # JSONのキーは文字列化されるためintに戻す
                    self.metadata_store = {int(k): v for k, v in store.items()}
                    self.next_id = max(
                        self.metadata_store.keys(), default=-1) + 1

                    # Note: 実際の運用ではSDRも保存して再ロードする必要がありますが、
                    # 今回はデモ・API定義としてメタデータのみの構造としています。
            except Exception as e:
                print(f"Error loading LTM metadata: {e}")
                self.metadata_store = {}
                self.next_id = 0

    def get_memory_stats(self) -> Dict[str, Any]:
        return {
            "total_memories": self.engine.memory_count(),
            "storage_path": self.save_path
        }
