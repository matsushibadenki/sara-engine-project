_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/attention.py",
    "//": "タイトル: Spike-based Multi-Head Attention",
    "//": "目的: 行列演算やBPを使わずに、スパイクの一致度（Overlap）と時間減衰を伴う加算（Membrane Potential）によってSelf-AttentionをSNNで実用的に再現する。"
}

import random
from typing import List, Dict, Set

try:
    from .. import sara_rust_core  # type: ignore
    RUST_AVAILABLE = True
except ImportError:
    try:
        import sara_rust_core  # type: ignore
        RUST_AVAILABLE = True
    except ImportError:
        RUST_AVAILABLE = False


class SpikeAttention:
    def __init__(self, input_size: int, hidden_size: int, memory_size: int = 50, num_heads: int = 4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_heads = num_heads
        
        self.use_rust = False 
        
        if self.use_rust:
            self.core = sara_rust_core.RustSpikeAttention(input_size, hidden_size, num_heads, memory_size)
        else:
            self.memory_keys: List[List[Set[int]]] = [] 
            self.memory_values: List[List[List[int]]] = [] 
            
            self.w_query: List[List[int]] = [] 
            self._init_sparse_weights(self.w_query, input_size, hidden_size)
            
            self.w_key: List[List[int]] = []
            self._init_sparse_weights(self.w_key, input_size, hidden_size)
            
            self.w_value: List[List[int]] = []
            self._init_sparse_weights(self.w_value, input_size, hidden_size)

    def _init_sparse_weights(self, weight_list: List[List[int]], dim_in: int, dim_out: int, density: float = 0.05):
        for _ in range(dim_in):
            num_targets = max(1, int(dim_out * density))
            targets = random.sample(range(dim_out), num_targets)
            weight_list.append(sorted(targets))

    def _project(self, input_spikes: List[int], weights: List[List[int]]) -> List[int]:
        if not input_spikes: return []
        
        potentials: Dict[int, int] = {}
        for idx in input_spikes:
            if idx < len(weights):
                for target in weights[idx]:
                    potentials[target] = potentials.get(target, 0) + 1
        
        if not potentials: return []
        
        mean_p = sum(potentials.values()) / len(potentials)
        threshold = max(2.0, mean_p) 
        
        out_spikes = [n for n, p in potentials.items() if p >= threshold]
        
        target_k = max(1, int(self.hidden_size * 0.1))
        if len(out_spikes) > target_k:
            sorted_neurons = sorted(potentials.items(), key=lambda x: x[1], reverse=True)
            out_spikes = [n for n, _ in sorted_neurons[:target_k]]
            
        return out_spikes

    def compute(self, query_spikes: List[int]) -> List[int]:
        if self.use_rust:
            return self.core.compute(query_spikes)
        else:
            return self._compute_python(query_spikes)

    def _compute_python(self, query_spikes: List[int]) -> List[int]:
        q_full = self._project(query_spikes, self.w_query)
        k_full = self._project(query_spikes, self.w_key)
        v_full = self._project(query_spikes, self.w_value)
        
        q_heads: List[Set[int]] = [set() for _ in range(self.num_heads)]
        k_heads: List[Set[int]] = [set() for _ in range(self.num_heads)]
        v_heads: List[List[int]] = [[] for _ in range(self.num_heads)]
        
        head_dim = max(1, self.hidden_size // self.num_heads)
        
        for idx in q_full: q_heads[min(idx // head_dim, self.num_heads - 1)].add(idx)
        for idx in k_full: k_heads[min(idx // head_dim, self.num_heads - 1)].add(idx)
        for idx in v_full: v_heads[min(idx // head_dim, self.num_heads - 1)].append(idx)
        
        self.memory_keys.append(k_heads)
        self.memory_values.append(v_heads)
        if len(self.memory_keys) > self.memory_size:
            self.memory_keys.pop(0)
            self.memory_values.pop(0)
            
        if len(self.memory_keys) < 2: return []

        v_potentials: Dict[int, float] = {}
        current_time = len(self.memory_keys) - 1
        
        for h in range(self.num_heads):
            q_set = q_heads[h]
            if not q_set: continue
            
            for t, past_keys_heads in enumerate(self.memory_keys):
                k_set = past_keys_heads[h]
                if not k_set: continue
                
                overlap = len(q_set.intersection(k_set))
                
                if overlap > 0:
                    age = current_time - t
                    decay = 0.98 ** age
                    score = overlap * decay
                    
                    for v_idx in self.memory_values[t][h]:
                        v_potentials[v_idx] = v_potentials.get(v_idx, 0.0) + score
                        
        if not v_potentials:
            return []

        avg_v = sum(v_potentials.values()) / len(v_potentials)
        v_threshold = max(1.0, avg_v * 1.1) 
        
        context_spikes = [v for v, p in v_potentials.items() if p >= v_threshold]
        
        target_density = max(1, int(self.hidden_size * 0.05))
        if len(context_spikes) > target_density:
            sorted_v = sorted(v_potentials.items(), key=lambda x: x[1], reverse=True)
            context_spikes = [v for v, _ in sorted_v[:target_density]]
            
        return sorted(context_spikes)

    def update_memory(self, input_spikes: List[int]):
        pass

    def reset(self):
        if self.use_rust:
            self.core.reset()
        else:
            self.memory_keys.clear()
            self.memory_values.clear()