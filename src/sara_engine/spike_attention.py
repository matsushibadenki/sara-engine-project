# src/sara_engine/spike_attention.py
# title: Spike-based Attention Mechanism
# description: 行列演算を使わず、スパイクの一致度（Overlap）を用いて過去の記憶を参照する注意機構。

import numpy as np
from typing import List, Tuple, Dict

class SpikeAttention:
    """
    スパイクベースの注意機構 (Spike Attention)
    """
    
    def __init__(self, input_size: int, hidden_size: int, memory_size: int = 50):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size 
        
        self.memory_keys: List[List[int]] = []
        self.memory_values: List[List[int]] = []
        
        self.w_query = self._init_sparse_mapping(input_size, hidden_size, density=0.1)
        self.w_key = self._init_sparse_mapping(input_size, hidden_size, density=0.1)
        self.w_value = self._init_sparse_mapping(input_size, hidden_size, density=0.1)
        
        self.temperature = 5.0 
        self.decay = 0.95      

    def _init_sparse_mapping(self, n_in: int, n_out: int, density: float) -> List[List[int]]:
        mapping = []
        for _ in range(n_in):
            n_targets = int(n_out * density)
            if n_targets > 0:
                targets = np.random.choice(n_out, n_targets, replace=False).tolist()
                mapping.append(targets)
            else:
                mapping.append([])
        return mapping

    def _project(self, spikes: List[int], mapping: List[List[int]]) -> List[int]:
        projected = set()
        for idx in spikes:
            if idx < len(mapping):
                for target in mapping[idx]:
                    projected.add(target)
        return list(projected)

    def update_memory(self, spikes: List[int]):
        key = self._project(spikes, self.w_key)
        value = self._project(spikes, self.w_value)
        
        self.memory_keys.append(key)
        self.memory_values.append(value)
        
        if len(self.memory_keys) > self.memory_size:
            self.memory_keys.pop(0)
            self.memory_values.pop(0)

    def compute(self, current_spikes: List[int]) -> List[int]:
        if not self.memory_keys:
            return []
            
        query = self._project(current_spikes, self.w_query)
        query_set = set(query)
        
        if not query_set:
            return []

        scores = []
        for i, key in enumerate(self.memory_keys):
            time_factor = self.decay ** (len(self.memory_keys) - 1 - i)
            key_set = set(key)
            if not key_set:
                scores.append(0.0)
                continue
                
            intersection = len(query_set.intersection(key_set))
            union = len(query_set.union(key_set))
            jaccard = intersection / union if union > 0 else 0.0
            
            # 修正: 温度パラメータは冪乗で適用
            score = (jaccard ** self.temperature) * time_factor
            scores.append(score)
        
        attention_spikes = set()
        total_score = sum(scores)
        
        if total_score > 0:
            # 正規化
            normalized_scores = [s / total_score for s in scores]
            
            # Top-K 選択
            top_k = 3
            top_indices = np.argsort(normalized_scores)[-top_k:]
            
            for idx in top_indices:
                if normalized_scores[idx] > 0.1:  # 閾値
                    for neuron_idx in self.memory_values[idx]:
                        attention_spikes.add(neuron_idx)
        
        return list(attention_spikes)

    def reset(self):
        self.memory_keys.clear()
        self.memory_values.clear()