_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/nn/attention.py",
    "//": "ファイルの日本語タイトル: 高速化版スパイキング・アテンション",
    "//": "ファイルの目的や内容: sara_rust_core.SpikeEngine を統合し、大規模なスパイク伝播と学習を高速化したアテンション層。"
}

import random
from typing import List, Dict, Set, Optional
from .module import SNNModule

# Rustコアが利用可能な場合はインポート
try:
    from sara_engine import sara_rust_core
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

class SpikeSelfAttention(SNNModule):
    def __init__(self, embed_dim: int, density: float = 0.1, context_size: int = 64, use_rust: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.use_rust = use_rust and RUST_AVAILABLE
        
        # 重みの初期化
        self.q_weights = self._init_sparse_weights(embed_dim, embed_dim, density)
        self.k_weights = self._init_sparse_weights(embed_dim, embed_dim, density)
        self.v_weights = self._init_sparse_weights(embed_dim, embed_dim, density)
        self.o_weights = self._init_sparse_weights(embed_dim, embed_dim, density)
        
        # Rustエンジンを使用する場合のセットアップ
        if self.use_rust:
            self.q_engine = sara_rust_core.SpikeEngine()
            self.k_engine = sara_rust_core.SpikeEngine()
            self.v_engine = sara_rust_core.SpikeEngine()
            self.o_engine = sara_rust_core.SpikeEngine()
            self._sync_to_rust()

        self.register_state("q_weights")
        self.register_state("k_weights")
        self.register_state("v_weights")
        self.register_state("o_weights")
        
        self.key_buffer: List[Set[int]] = []
        self.value_buffer: List[Set[int]] = []

    def _sync_to_rust(self):
        """Python側の重みをRustコアへ同期する"""
        if self.use_rust:
            self.q_engine.set_weights(self.q_weights)
            self.k_engine.set_weights(self.k_weights)
            self.v_engine.set_weights(self.v_weights)
            self.o_engine.set_weights(self.o_weights)

    def _init_sparse_weights(self, in_dim: int, out_dim: int, density: float) -> List[Dict[int, float]]:
        weights = [{} for _ in range(in_dim)]
        for i in range(in_dim):
            num = max(1, int(out_dim * density))
            for t in random.sample(range(out_dim), num):
                weights[i][t] = random.uniform(0.1, 1.0)
        return weights

    def forward(self, x_spikes: List[int], learning: bool = False) -> List[int]:
        threshold = 1.0 if learning else 0.5
        max_out = max(1, int(self.embed_dim * 0.15))

        if self.use_rust:
            # Rustエンジンによる高速伝播
            q_list = self.q_engine.propagate(x_spikes, threshold, max_out)
            k_list = self.k_engine.propagate(x_spikes, threshold, max_out)
            v_list = self.v_engine.propagate(x_spikes, threshold, max_out)
        else:
            # 従来のPython実装
            q_list = self._sparse_propagate(x_spikes, self.q_weights, self.embed_dim, threshold, max_out)
            k_list = self._sparse_propagate(x_spikes, self.k_weights, self.embed_dim, threshold, max_out)
            v_list = self._sparse_propagate(x_spikes, self.v_weights, self.embed_dim, threshold, max_out)

        q_spikes = set(q_list)
        self.key_buffer.append(set(k_list))
        self.value_buffer.append(set(v_list))
        
        if len(self.key_buffer) > self.context_size:
            self.key_buffer.pop(0)
            self.value_buffer.pop(0)

        # 動的ルーティング (Coincidence Detection)
        routed_v = set()
        best_match_idx = -1
        max_coinc = 0
        dyn_thresh = max(1, int(len(q_spikes) * 0.2))
        
        for i, past_k in enumerate(self.key_buffer):
            coinc = len(q_spikes.intersection(past_k))
            if coinc >= dyn_thresh and coinc > max_coinc:
                max_coinc = coinc
                best_match_idx = i
        
        if best_match_idx != -1:
            routed_v = self.value_buffer[best_match_idx]

        if self.use_rust:
            y_spikes = self.o_engine.propagate(list(routed_v), threshold, max_out)
            if learning:
                self.q_engine.apply_stdp(x_spikes, list(q_spikes | set(x_spikes)), 0.05)
                self.k_engine.apply_stdp(x_spikes, k_list, 0.05)
                self.v_engine.apply_stdp(x_spikes, v_list, 0.05)
                self.o_engine.apply_stdp(list(routed_v), y_spikes, 0.05)
                # Python側の重みも同期 (Save/Loadのため)
                self.q_weights = self.q_engine.get_weights()
        else:
            y_spikes = self._sparse_propagate(list(routed_v), self.o_weights, self.embed_dim, threshold, max_out)
            if learning:
                self._apply_stdp(x_spikes, q_list, self.q_weights)
                # ... 他の重みの更新 ...

        return y_spikes

    def _sparse_propagate(self, active, weights, out_dim, threshold, max_out):
        # (既存のPython実装のフォールバック)
        potentials = [0.0] * out_dim
        for s in active:
            if s < len(weights):
                for t, w in weights[s].items(): potentials[t] += w
        active_sorted = sorted([(i, p) for i, p in enumerate(potentials) if p > threshold], key=lambda x: x[1], reverse=True)
        return [i for i, p in active_sorted[:max_out]]