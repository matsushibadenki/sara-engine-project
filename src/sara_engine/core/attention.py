import numpy as np
from typing import List

class SpikeAttention:
    """
    Spike-based Attention Mechanism (Simplified)
    スパイクのタイミングと活性度に基づく簡易アテンション
    行列演算を行わず、スパイクの一致度（Overlap）で重要度を判定する。
    """
    def __init__(self, input_size: int, hidden_size: int, memory_size: int = 50):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        
        # Key-Value Memory (Spike Patterns)
        # 実際には行列ではなく、活性化したニューロンのインデックスリストを保持する
        self.memory_keys: List[List[int]] = []
        self.memory_values: List[List[int]] = []
        
        # Query Projection (Sparse Connectivity)
        # 入力スパイクをQuery空間へ射影するためのスパース結合
        self.w_query: List[List[int]] = [] # Adjacency list
        self._init_sparse_weights(self.w_query, input_size, hidden_size, density=0.05)
        
        # Key Projection
        self.w_key: List[List[int]] = []
        self._init_sparse_weights(self.w_key, input_size, hidden_size, density=0.05)
        
        # Value Projection
        self.w_value: List[List[int]] = []
        self._init_sparse_weights(self.w_value, input_size, hidden_size, density=0.05)

    def _init_sparse_weights(self, weight_list: List[List[int]], dim_in: int, dim_out: int, density: float):
        for _ in range(dim_in):
            targets = np.random.choice(dim_out, int(dim_out * density), replace=False).tolist()
            weight_list.append(targets)

    def _project(self, input_spikes: List[int], weights: List[List[int]]) -> List[int]:
        """スパイク入力を重みに従って投影し、活性化スパイクを返す (Winner-Take-All)"""
        if not input_spikes:
            return []
            
        # 投影先のニューロンの電位を積算
        potentials = {}
        for idx in input_spikes:
            if idx < len(weights):
                for target in weights[idx]:
                    potentials[target] = potentials.get(target, 0.0) + 1.0
        
        if not potentials:
            return []
            
        # Top-K (Sparsity constraint)
        k = max(1, int(self.hidden_size * 0.05))
        sorted_neurons = sorted(potentials.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_neurons[:k]]

    def compute(self, query_spikes: List[int]) -> List[int]:
        """
        Queryスパイクに基づいてMemoryから関連情報を取得し、
        コンテキスト信号（スパイク）として返す。
        """
        if not self.memory_keys:
            return []

        # 1. Project Query
        q_vec = self._project(query_spikes, self.w_query)
        q_set = set(q_vec)
        
        # 2. Calculate Attention Scores (Overlap with Keys)
        scores = []
        for i, k_vec in enumerate(self.memory_keys):
            k_set = set(k_vec)
            overlap = len(q_set.intersection(k_set))
            scores.append((i, overlap))
        
        # 3. Retrieve Values (Softmax-like selection)
        # スパイクベースなので、スコアが高い上位のメモリのみを採用する
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k_memory = 3
        
        context_spikes = set()
        for i, score in scores[:top_k_memory]:
            if score > 0:
                context_spikes.update(self.memory_values[i])
        
        return list(context_spikes)

    def update_memory(self, input_spikes: List[int]):
        """現在の入力をメモリに追加（古いものは忘却）"""
        k_vec = self._project(input_spikes, self.w_key)
        v_vec = self._project(input_spikes, self.w_value)
        
        self.memory_keys.append(k_vec)
        self.memory_values.append(v_vec)
        
        if len(self.memory_keys) > self.memory_size:
            self.memory_keys.pop(0)
            self.memory_values.pop(0)

    def reset(self):
        self.memory_keys.clear()
        self.memory_values.clear()