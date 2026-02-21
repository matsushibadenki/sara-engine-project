_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/spike_attention.py",
    "//": "タイトル: スパイク自己注意機構 (Rustバックエンド対応)",
    "//": "目的: Rust拡張モジュールを正しいパスからインポートし、スパイク伝播処理をオフロードして高速化する。"
}

import random
from typing import List, Dict

try:
    # インポートパスを修正: sara_engineパッケージ内から呼び出す
    from sara_engine import sara_rust_core
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("Warning: sara_rust_core not found. Using slow Python backend for Attention.")

class SpikingSelfAttention:
    def __init__(self, sdr_size: int, num_heads: int = 4, decay_rate: float = 0.9):
        self.sdr_size = sdr_size
        self.num_heads = num_heads
        self.decay_rate = decay_rate
        
        self.W_q: List[Dict[int, float]] = [{} for _ in range(sdr_size)]
        self.W_k: List[Dict[int, float]] = [{} for _ in range(sdr_size)]
        self.W_v: List[Dict[int, float]] = [{} for _ in range(sdr_size)]
        
        self._init_sparse_weights(self.W_q)
        self._init_sparse_weights(self.W_k)
        self._init_sparse_weights(self.W_v)
        
        self.k_trace: List[float] = [0.0] * sdr_size
        self.v_trace: List[float] = [0.0] * sdr_size

    def _init_sparse_weights(self, weights: List[Dict[int, float]], density: float = 0.05):
        rng = random.Random(42)
        for i in range(self.sdr_size):
            num_connections = max(1, int(self.sdr_size * density))
            targets = rng.sample(range(self.sdr_size), num_connections)
            for t in targets:
                weights[i][t] = random.uniform(0.1, 0.5)

    def reset(self):
        self.k_trace = [0.0] * self.sdr_size
        self.v_trace = [0.0] * self.sdr_size

    def _propagate(self, active_inputs: List[int], weights: List[Dict[int, float]]) -> List[int]:
        if RUST_AVAILABLE:
            return sara_rust_core.sparse_propagate_threshold(active_inputs, weights, self.sdr_size, 1.0)
            
        potentials = [0.0] * self.sdr_size
        for pre_id in active_inputs:
            for post_id, w in weights[pre_id].items():
                potentials[post_id] += w
        
        return [i for i, p in enumerate(potentials) if p > 1.0]

    def forward(self, current_spikes: List[int]) -> List[int]:
        q_spikes = self._propagate(current_spikes, self.W_q)
        k_spikes = self._propagate(current_spikes, self.W_k)
        v_spikes = self._propagate(current_spikes, self.W_v)
        
        for i in range(self.sdr_size):
            self.k_trace[i] *= self.decay_rate
            self.v_trace[i] *= self.decay_rate
            
        for k in k_spikes:
            self.k_trace[k] += 1.0
        for v in v_spikes:
            self.v_trace[v] += 1.0
            
        overlap_score = sum(self.k_trace[q] for q in q_spikes)
        
        output_potentials = [0.0] * self.sdr_size
        if overlap_score > 0:
            signal_strength = min(1.0, overlap_score / max(1.0, len(q_spikes)))
            for i in range(self.sdr_size):
                output_potentials[i] = self.v_trace[i] * signal_strength
                
        target_active = max(1, int(self.sdr_size * 0.1))
        active_neurons = [(i, p) for i, p in enumerate(output_potentials) if p > 0.5]
        active_neurons.sort(key=lambda x: x[1], reverse=True)
        
        attended_spikes = sorted([i for i, p in active_neurons[:target_active]])
        return attended_spikes