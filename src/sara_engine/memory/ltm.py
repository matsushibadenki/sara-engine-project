import pickle
import os
import time
from typing import List, Dict, Any

class SparseMemoryStore:
    """
    SDRベースの長期記憶ストレージ (Sparse Distributed Memory)
    """
    def __init__(self, filepath: str = "sara_ltm.pkl"):
        self.filepath = filepath
        self.memories: List[Dict[str, Any]] = []
        self.load()

    def add(self, sdr: List[int], content: str, memory_type: str = "episodic"):
        """記憶を追加して即時保存"""
        entry = {
            'sdr': sdr,
            'content': content,
            'timestamp': time.time(),
            'type': memory_type
        }
        self.memories.append(entry)
        self.save()

    def search(self, query_sdr: List[int], top_k: int = 3, threshold: float = 0.1) -> List[Dict[str, Any]]:
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
            score = overlap / denominator if denominator > 0 else 0.0
            
            if score >= threshold:
                results.append({
                    'content': mem['content'],
                    'score': score,
                    'type': mem['type'],
                    'timestamp': mem['timestamp']
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