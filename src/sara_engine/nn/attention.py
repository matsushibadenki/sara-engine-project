_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/nn/attention.py",
    "//": "ファイルの日本語タイトル: スパイキング・セルフアテンション (nn.Module版)",
    "//": "ファイルの目的や内容: TransformersのAttention層をSNNModuleに適合させ、state_dict対応やモジュールツリーへの組み込みを可能にしたクラス。行列演算を完全に排除。"
}

import random
from typing import List, Dict, Set
from .module import SNNModule

class SpikeSelfAttention(SNNModule):
    def __init__(self, embed_dim: int, density: float = 0.1, context_size: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_size = context_size
        
        self.q_weights = self._init_sparse_weights(embed_dim, embed_dim, density)
        self.k_weights = self._init_sparse_weights(embed_dim, embed_dim, density)
        self.v_weights = self._init_sparse_weights(embed_dim, embed_dim, density)
        self.o_weights = self._init_sparse_weights(embed_dim, embed_dim, density)
        
        # SNNModuleの機能を利用して状態を登録 (Save/Load可能にする)
        self.register_state("q_weights")
        self.register_state("k_weights")
        self.register_state("v_weights")
        self.register_state("o_weights")
        
        # 動的な文脈バッファ
        self.key_buffer: List[Set[int]] = []
        self.value_buffer: List[Set[int]] = []

    def _init_sparse_weights(self, in_dim: int, out_dim: int, density: float) -> List[Dict[int, float]]:
        weights: List[Dict[int, float]] = [{} for _ in range(in_dim)]
        for i in range(in_dim):
            num_connections = max(1, int(out_dim * density))
            targets = random.sample(range(out_dim), num_connections)
            for t in targets:
                weights[i][t] = random.uniform(0.1, 1.0)
        return weights

    def reset_state(self):
        super().reset_state()
        self.key_buffer.clear()
        self.value_buffer.clear()

    def _sparse_propagate(self, active_spikes: List[int], weights: List[Dict[int, float]], out_size: int, threshold: float = 0.5) -> List[int]:
        potentials = [0.0] * out_size
        for s in active_spikes:
            if s < len(weights):
                for t, w in weights[s].items():
                    if t < out_size:
                        potentials[t] += w
        
        active = [(i, p) for i, p in enumerate(potentials) if p > threshold]
        active.sort(key=lambda x: x[1], reverse=True)
        max_spikes = max(1, int(out_size * 0.15))
        return [i for i, p in active[:max_spikes]]

    def _apply_stdp(self, pre_spikes: List[int], post_spikes: List[int], weights: List[Dict[int, float]], lr: float = 0.05):
        post_set = set(post_spikes)
        for pre in pre_spikes:
            if pre < len(weights):
                to_remove = []
                for target in list(weights[pre].keys()):
                    if target in post_set:
                        weights[pre][target] = min(3.0, weights[pre][target] + lr)
                    else:
                        weights[pre][target] = max(0.0, weights[pre][target] - lr * 0.05)
                        if weights[pre][target] <= 0.01:
                            to_remove.append(target)
                for target in to_remove:
                    del weights[pre][target]
                
                for target in post_set:
                    if target not in weights[pre]:
                        if random.random() < 0.1:
                            weights[pre][target] = 0.2

    def forward(self, x_spikes: List[int], learning: bool = False) -> List[int]:
        threshold = 1.0 if learning else 0.5
        
        q_list = self._sparse_propagate(x_spikes, self.q_weights, self.embed_dim, threshold=threshold)
        k_list = self._sparse_propagate(x_spikes, self.k_weights, self.embed_dim, threshold=threshold)
        v_list = self._sparse_propagate(x_spikes, self.v_weights, self.embed_dim, threshold=threshold)

        q_spikes = set(q_list)
        k_spikes = set(k_list)
        v_spikes = set(v_list)

        self.key_buffer.append(k_spikes)
        self.value_buffer.append(v_spikes)
        
        if len(self.key_buffer) > self.context_size:
            self.key_buffer.pop(0)
            self.value_buffer.pop(0)

        routed_v_spikes = set()
        dynamic_threshold = max(1, int(len(q_spikes) * 0.2))
        
        best_match_idx = -1
        max_coincidence = 0
        
        for i, past_k in enumerate(self.key_buffer):
            coincidence = len(q_spikes.intersection(past_k))
            if coincidence >= dynamic_threshold and coincidence > max_coincidence:
                max_coincidence = coincidence
                best_match_idx = i

        if best_match_idx != -1:
            routed_v_spikes.update(self.value_buffer[best_match_idx])

        y_spikes = self._sparse_propagate(list(routed_v_spikes), self.o_weights, self.embed_dim, threshold=threshold)
        
        if learning:
            forced_q = list(set(q_list) | set(x_spikes))
            forced_k = list(set(k_list) | set(x_spikes))
            forced_v = list(set(v_list) | set(x_spikes))
            self._apply_stdp(x_spikes, forced_q, self.q_weights)
            self._apply_stdp(x_spikes, forced_k, self.k_weights)
            self._apply_stdp(x_spikes, forced_v, self.v_weights)
            
            forced_y = list(set(y_spikes) | set(x_spikes))
            pre_y = list(routed_v_spikes) if routed_v_spikes else x_spikes
            self._apply_stdp(pre_y, forced_y, self.o_weights)
            
        return y_spikes