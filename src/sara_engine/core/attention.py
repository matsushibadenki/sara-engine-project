_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/attention.py",
    "//": "タイトル: Spike-based Multi-Head Attention",
    "//": "目的: TransformersのSelf-AttentionをSNNで再現する。"
}

import numpy as np
from typing import List

# Rust Core Import Check
try:
    from .. import sara_rust_core
    RUST_AVAILABLE = True
except ImportError:
    try:
        import sara_rust_core
        RUST_AVAILABLE = True
    except ImportError:
        RUST_AVAILABLE = False

class SpikeAttention:
    """
    Spike-based Multi-Head Attention Mechanism.
    行列演算を使わず、スパイクの一致度（Overlap）で重要度を判定する。
    """
    def __init__(self, input_size: int, hidden_size: int, memory_size: int = 50, num_heads: int = 4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_heads = num_heads
        
        self.use_rust = RUST_AVAILABLE
        
        if self.use_rust:
            self.core = sara_rust_core.RustSpikeAttention(input_size, hidden_size, num_heads, memory_size)
        else:
            # Python fallback implementation
            self.memory_keys: List[List[List[int]]] = [] # [time][head][indices]
            self.memory_values: List[List[List[int]]] = []
            
            self.w_query: List[List[int]] = [] 
            self._init_sparse_weights(self.w_query, input_size, hidden_size)
            
            self.w_key: List[List[int]] = []
            self._init_sparse_weights(self.w_key, input_size, hidden_size)
            
            self.w_value: List[List[int]] = []
            self._init_sparse_weights(self.w_value, input_size, hidden_size)

    def _init_sparse_weights(self, weight_list: List[List[int]], dim_in: int, dim_out: int, density: float = 0.05):
        for _ in range(dim_in):
            targets = np.random.choice(dim_out, max(1, int(dim_out * density)), replace=False).tolist()
            weight_list.append(targets)

    def _project(self, input_spikes: List[int], weights: List[List[int]]) -> List[int]:
        if not input_spikes: return []
        potentials = {}
        for idx in input_spikes:
            if idx < len(weights):
                for target in weights[idx]:
                    potentials[target] = potentials.get(target, 0) + 1
        
        if not potentials: return []
        
        # Winner-Take-All (Sort by potential)
        k = max(1, int(self.hidden_size * 0.1))
        sorted_neurons = sorted(potentials.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_neurons[:k]]

    def compute(self, query_spikes: List[int]) -> List[int]:
        """
        Queryスパイクに基づいてMemoryから関連情報を取得する。
        """
        if self.use_rust:
            return self.core.compute(query_spikes)
        else:
            return self._compute_python(query_spikes)

    def _compute_python(self, query_spikes: List[int]) -> List[int]:
        # 1. Project
        q_full = self._project(query_spikes, self.w_query)
        k_full = self._project(query_spikes, self.w_key)
        v_full = self._project(query_spikes, self.w_value)
        
        # 2. Split Heads
        q_heads = [[] for _ in range(self.num_heads)]
        k_heads = [[] for _ in range(self.num_heads)]
        v_heads = [[] for _ in range(self.num_heads)]
        
        for idx in q_full: q_heads[idx % self.num_heads].append(idx)
        for idx in k_full: k_heads[idx % self.num_heads].append(idx)
        for idx in v_full: v_heads[idx % self.num_heads].append(idx)
        
        # 3. Store in Memory
        self.memory_keys.append(k_heads)
        self.memory_values.append(v_heads)
        if len(self.memory_keys) > self.memory_size:
            self.memory_keys.pop(0)
            self.memory_values.pop(0)
            
        if len(self.memory_keys) < 2: return []

        # 4. Attention (Overlap)
        context_spikes = set()
        
        for h in range(self.num_heads):
            q_set = set(q_heads[h])
            if not q_set: continue
            
            scores = []
            for t, past_keys_heads in enumerate(self.memory_keys):
                k_set = set(past_keys_heads[h])
                overlap = len(q_set.intersection(k_set))
                if overlap > 0:
                    scores.append((t, overlap))
            
            # Top-K retrieval
            scores.sort(key=lambda x: x[1], reverse=True)
            for t, _ in scores[:3]:
                for v in self.memory_values[t][h]:
                    context_spikes.add(v)
                    
        return list(context_spikes)

    def update_memory(self, input_spikes: List[int]):
        """
        注意: compute() 内で自動的にupdateする設計にしたため、
        外部からの明示的な呼び出しは互換性のために残すが何もしない。
        """
        pass

    def reset(self):
        if self.use_rust:
            self.core.reset()
        else:
            self.memory_keys.clear()
            self.memory_values.clear()