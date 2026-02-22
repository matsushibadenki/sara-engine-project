_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/spike_attention.py",
    "//": "ファイルの日本語タイトル: 生体模倣型 スパイキング・アテンション",
    "//": "ファイルの目的や内容: RustとPythonの境界で発生する辞書データのシリアライズ遅延(PyO3のオーバーヘッド)を排除し、純粋なPython処理に統合することで劇的な高速化を図る。"
}

import random
from typing import List, Dict, Set

class SpikeSelfAttention:
    """
    Spike-driven Self-Attention without matrix multiplication or backpropagation.
    Replaces Q*K^T and Softmax with biological Coincidence Detection (Set intersections).
    """
    def __init__(self, embed_dim: int, density: float = 0.05, context_size: int = 64, coincidence_threshold: int = 2):
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.base_coincidence_threshold = coincidence_threshold
        
        self.q_weights = self._init_sparse_weights(embed_dim, embed_dim, density)
        self.k_weights = self._init_sparse_weights(embed_dim, embed_dim, density)
        self.v_weights = self._init_sparse_weights(embed_dim, embed_dim, density)
        self.o_weights = self._init_sparse_weights(embed_dim, embed_dim, density)
        
        self.key_buffer: List[Set[int]] = []
        self.value_buffer: List[Set[int]] = []

    def _init_sparse_weights(self, in_dim: int, out_dim: int, density: float) -> List[Dict[int, float]]:
        weights: List[Dict[int, float]] = [{} for _ in range(in_dim)]
        for i in range(in_dim):
            num_connections = max(1, int(out_dim * density))
            targets = random.sample(range(out_dim), num_connections)
            for t in targets:
                weights[i][t] = random.uniform(-1.0, 1.0)
        return weights

    def reset_state(self):
        self.key_buffer.clear()
        self.value_buffer.clear()

    def _sparse_propagate(self, active_spikes: List[int], weights: List[Dict[int, float]], out_size: int, threshold: float = 0.5) -> List[int]:
        # Rust拡張の呼び出しを廃止し、Python内でのみ処理を完結させる
        potentials = [0.0] * out_size
        for s in active_spikes:
            if s < len(weights):
                for t, w in weights[s].items():
                    if t < out_size:
                        potentials[t] += w
        return [i for i, p in enumerate(potentials) if p > threshold]

    def forward(self, x_spikes: List[int], learning: bool = True) -> List[int]:
        q_list = self._sparse_propagate(x_spikes, self.q_weights, self.embed_dim, threshold=0.8)
        k_list = self._sparse_propagate(x_spikes, self.k_weights, self.embed_dim, threshold=0.8)
        v_list = self._sparse_propagate(x_spikes, self.v_weights, self.embed_dim, threshold=0.8)

        q_spikes = set(q_list)
        k_spikes = set(k_list)
        v_spikes = set(v_list)

        self.key_buffer.append(k_spikes)
        self.value_buffer.append(v_spikes)
        
        if len(self.key_buffer) > self.context_size:
            self.key_buffer.pop(0)
            self.value_buffer.pop(0)

        routed_v_spikes = set()
        dynamic_threshold = max(self.base_coincidence_threshold, int(len(q_spikes) * 0.3))
        
        for i, past_k in enumerate(self.key_buffer):
            coincidence = len(q_spikes.intersection(past_k))
            if coincidence >= dynamic_threshold:
                routed_v_spikes.update(self.value_buffer[i])

        y_spikes = self._sparse_propagate(list(routed_v_spikes), self.o_weights, self.embed_dim, threshold=1.0)
        return y_spikes