_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/spike_attention.py",
    "//": "ファイルの日本語タイトル: 生体模倣型 スパイキング・アテンション",
    "//": "ファイルの目的や内容: mypy型エラーの修正。動的モジュール解決の回避と、weights変数への型アノテーション追加。"
}

import random
from typing import List, Dict

try:
    from sara_engine import sara_rust_core  # type: ignore
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

class SpikeSelfAttention:
    """
    Spike-driven Self-Attention without matrix multiplication or backpropagation.
    Uses STDP (Spike-Timing Dependent Plasticity) traces to associate Query and Key spikes,
    routing Value spikes accordingly.
    """
    def __init__(self, embed_dim: int, density: float = 0.1, use_rust: bool = True):
        self.embed_dim = embed_dim
        self.use_rust = use_rust and RUST_AVAILABLE
        
        # Sparse projection dictionaries: [input_neuron_idx] -> {output_neuron_idx: weight}
        self.q_weights = self._init_sparse_weights(embed_dim, embed_dim, density)
        self.k_weights = self._init_sparse_weights(embed_dim, embed_dim, density)
        self.v_weights = self._init_sparse_weights(embed_dim, embed_dim, density)
        self.o_weights = self._init_sparse_weights(embed_dim, embed_dim, density)
        
        # Biological short-term memory traces for Keys
        self.k_traces = [0.0] * embed_dim
        self.trace_decay = 0.9
        
        # Dynamic attention synapses (STDP learned routing: Query -> Value)
        self.attn_synapses: List[Dict[int, float]] = [{} for _ in range(embed_dim)]

    def _init_sparse_weights(self, in_dim: int, out_dim: int, density: float) -> List[Dict[int, float]]:
        weights: List[Dict[int, float]] = [{} for _ in range(in_dim)]
        for i in range(in_dim):
            num_connections = max(1, int(out_dim * density))
            targets = random.sample(range(out_dim), num_connections)
            for t in targets:
                # Initialize with random biological excitatory/inhibitory weights
                weights[i][t] = random.uniform(-1.0, 1.0)
        return weights

    def _sparse_propagate(self, active_spikes: List[int], weights: List[Dict[int, float]], out_size: int, threshold: float = 0.5) -> List[int]:
        if self.use_rust:
            # Call high-speed Rust core if available
            return sara_rust_core.sparse_propagate_threshold(active_spikes, weights, out_size, threshold)
            
        # Pure Python fallback
        potentials = [0.0] * out_size
        for s in active_spikes:
            if s < len(weights):
                for t, w in weights[s].items():
                    if t < out_size:
                        potentials[t] += w
                        
        return [i for i, p in enumerate(potentials) if p > threshold]

    def forward(self, x_spikes: List[int], learning: bool = True) -> List[int]:
        """Processes a single time-step of spikes."""
        # 1. Project to Q, K, V spaces
        q_spikes = self._sparse_propagate(x_spikes, self.q_weights, self.embed_dim, threshold=0.5)
        k_spikes = self._sparse_propagate(x_spikes, self.k_weights, self.embed_dim, threshold=0.5)
        v_spikes = self._sparse_propagate(x_spikes, self.v_weights, self.embed_dim, threshold=0.5)

        # 2. Decay previous Key traces
        for i in range(self.embed_dim):
            self.k_traces[i] *= self.trace_decay

        # 3. Update Key traces with new spikes
        for k in k_spikes:
            self.k_traces[k] = 1.0

        out_potentials = [0.0] * self.embed_dim

        # 4. Spike-driven Attention & STDP Learning
        for q in q_spikes:
            if learning:
                # STDP: Co-occurrence of Q spike and recent K trace strengthens their association
                for k in range(self.embed_dim):
                    if self.k_traces[k] > 0.1:
                        # Biological Hebbian update
                        self.attn_synapses[q][k] = self.attn_synapses[q].get(k, 0.0) + 0.1 * self.k_traces[k]
                        if self.attn_synapses[q][k] > 2.0:
                            self.attn_synapses[q][k] = 2.0
            
            # Route Value spikes modulated by learned Attention synapses
            for v in v_spikes:
                weight = self.attn_synapses[q].get(v, 0.0)
                if weight > 0.2:
                    out_potentials[q] += weight

        attn_out_spikes = [i for i, p in enumerate(out_potentials) if p > 1.0]
        
        # 5. Output projection
        y_spikes = self._sparse_propagate(attn_out_spikes, self.o_weights, self.embed_dim, threshold=0.5)
        return y_spikes