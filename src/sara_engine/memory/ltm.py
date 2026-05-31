# ディレクトリパス: src/sara_engine/memory/ltm.py
# タイトル: 疎分散長期記憶 (Sparse Distributed Memory Store)
# 目的: SDRベースの長期記憶ストレージ。文脈内学習へフィードバックするために検索結果へSDRを含めるよう拡張。
import pickle
import os
import time
from typing import List, Dict, Any, Optional

class SparseMemoryStore:
    """
    SDRベースの長期記憶ストレージ (Sparse Distributed Memory)
    """
    def __init__(self, filepath: str = "sara_ltm.pkl"):
        self.filepath = filepath
        self.memories: List[Dict[str, Any]] = []
        self.load()

    def add(
        self,
        sdr: List[int],
        content: str,
        memory_type: str = "episodic",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """記憶を追加して即時保存"""
        entry = {
            'sdr': sdr,
            'content': content,
            'timestamp': time.time(),
            'type': memory_type,
            'metadata': dict(metadata) if isinstance(metadata, dict) else {},
        }
        self.memories.append(entry)
        self.save()

    @staticmethod
    def _coerce_lower_set(values: Any) -> set[str]:
        if not isinstance(values, list):
            return set()
        return {
            str(value).strip().lower()
            for value in values
            if str(value).strip()
        }

    def _metadata_focus(
        self,
        memory: Dict[str, Any],
        query_metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        if not isinstance(query_metadata, dict):
            return {
                "context_match": 1.0,
                "role_match": 1.0,
                "keyword_overlap": 0.0,
            }

        metadata = memory.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        preferred_contexts = self._coerce_lower_set(query_metadata.get("contexts", []))
        memory_context = str(metadata.get("context", memory.get("type", ""))).strip().lower()
        context_match = 1.0 if not preferred_contexts or memory_context in preferred_contexts else 0.0

        preferred_role = str(query_metadata.get("preferred_role", "hybrid")).strip().lower() or "hybrid"
        memory_role = str(metadata.get("memory_role", "hybrid")).strip().lower() or "hybrid"
        role_match = 1.0 if preferred_role == "hybrid" or memory_role in {preferred_role, "hybrid"} else 0.0

        query_keywords = self._coerce_lower_set(query_metadata.get("keywords", []))
        memory_keywords = self._coerce_lower_set(metadata.get("keywords", []))
        keyword_overlap = 0.0
        if query_keywords:
            keyword_overlap = len(query_keywords.intersection(memory_keywords)) / max(1, len(query_keywords))

        return {
            "context_match": context_match,
            "role_match": role_match,
            "keyword_overlap": keyword_overlap,
        }

    def search(
        self,
        query_sdr: List[int],
        top_k: int = 3,
        threshold: float = 0.1,
        query_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.memories:
            return []

        query_set = set(query_sdr)
        if not query_set:
            return []
            
        results = []
        for mem in self.memories:
            mem_sdr_set = set(mem['sdr'])
            if not mem_sdr_set:
                continue
            
            overlap = len(query_set.intersection(mem_sdr_set))
            denominator = len(query_set)
            base_score = overlap / denominator if denominator > 0 else 0.0
            metadata_focus = self._metadata_focus(mem, query_metadata)
            context_match = metadata_focus["context_match"]
            role_match = metadata_focus["role_match"]
            keyword_overlap = metadata_focus["keyword_overlap"]
            normalized_query_metadata = query_metadata if isinstance(query_metadata, dict) else {}
            query_contexts = self._coerce_lower_set(normalized_query_metadata.get("contexts", []))
            query_keywords = self._coerce_lower_set(normalized_query_metadata.get("keywords", []))

            if query_contexts and context_match <= 0.0 and keyword_overlap <= 0.0:
                continue
            if base_score < 0.25 and context_match <= 0.0 and keyword_overlap <= 0.0:
                continue
            if query_keywords and keyword_overlap <= 0.0 and context_match <= 0.0:
                continue
            if base_score < 0.20 and role_match <= 0.0 and keyword_overlap <= 0.0:
                continue

            score = base_score
            score *= 1.0 + (context_match * 0.08) + (role_match * 0.06) + min(0.12, keyword_overlap * 0.15)

            if score >= threshold:
                results.append({
                    'content': mem['content'],
                    'score': score,
                    'base_score': base_score,
                    'type': mem['type'],
                    'timestamp': mem['timestamp'],
                    'sdr': mem['sdr'],  # 皮質へのフィードバック(ICL)のためにSDRを含める
                    'metadata': dict(mem.get('metadata', {})),
                    'ltm_context_match': bool(context_match > 0.0),
                    'ltm_role_match': bool(role_match > 0.0),
                    'ltm_metadata_keyword_overlap': keyword_overlap,
                })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def save(self):
        try:
            with open(self.filepath, 'wb') as f:
                pickle.dump(self.memories, f)
        except Exception as e:
            print(f"Error saving LTM: {e}")

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'rb') as f:
                    self.memories = pickle.load(f)
            except Exception as e:
                print(f"Error loading LTM: {e}")
                self.memories = []
    
    def clear(self):
        self.memories = []
        if os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
            except OSError:
                pass
