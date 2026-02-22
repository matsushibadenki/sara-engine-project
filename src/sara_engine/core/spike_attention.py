_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/spike_attention.py",
    "//": "ファイルの日本語タイトル: 連合学習強化型 スパイキング・アテンション",
    "//": "ファイルの目的や内容: 構造的可塑性(シナプス新生)とTeacher Forcingを導入し、連合記憶の形成と想起を確実にする。"
}

import random
from typing import List, Dict, Set

class SpikeSelfAttention:
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

    def state_dict(self) -> Dict:
        return {
            "embed_dim": self.embed_dim,
            "context_size": self.context_size,
            "base_coincidence_threshold": self.base_coincidence_threshold,
            "q_weights": self.q_weights,
            "k_weights": self.k_weights,
            "v_weights": self.v_weights,
            "o_weights": self.o_weights
        }

    def load_state_dict(self, state: Dict):
        self.embed_dim = state["embed_dim"]
        self.context_size = state["context_size"]
        self.base_coincidence_threshold = state["base_coincidence_threshold"]
        self.q_weights = [{int(k): float(v) for k, v in layer.items()} for layer in state["q_weights"]]
        self.k_weights = [{int(k): float(v) for k, v in layer.items()} for layer in state["k_weights"]]
        self.v_weights = [{int(k): float(v) for k, v in layer.items()} for layer in state["v_weights"]]
        self.o_weights = [{int(k): float(v) for k, v in layer.items()} for layer in state["o_weights"]]

    def _sparse_propagate(self, active_spikes: List[int], weights: List[Dict[int, float]], out_size: int, threshold: float = 0.5, gain: float = 1.0) -> List[int]:
        potentials = [0.0] * out_size
        for s in active_spikes:
            if s < len(weights):
                for t, w in weights[s].items():
                    if t < out_size:
                        potentials[t] += w * gain
        return [i for i, p in enumerate(potentials) if p > threshold]

    def _apply_stdp(self, pre_spikes: List[int], post_spikes: List[int], weights: List[Dict[int, float]], lr: float = 0.05):
        """Modified STDP: Minimize LTD and enable structural plasticity."""
        post_set = set(post_spikes)
        for pre in pre_spikes:
            if pre < len(weights):
                for target in list(weights[pre].keys()):
                    if target in post_set:
                        weights[pre][target] = min(3.0, weights[pre][target] + lr)
                    else:
                        weights[pre][target] = max(0.0, weights[pre][target] - lr * 0.01)
                
                # シナプス新生 (Structural Plasticity)
                for target in post_set:
                    if target not in weights[pre]:
                        if random.random() < 0.2:
                            weights[pre][target] = 0.5

    def forward(self, x_spikes: List[int], learning: bool = True) -> List[int]:
        gain = 1.0 if learning else 12.0
        
        q_list = self._sparse_propagate(x_spikes, self.q_weights, self.embed_dim, threshold=0.5, gain=gain)
        k_list = self._sparse_propagate(x_spikes, self.k_weights, self.embed_dim, threshold=0.5, gain=gain)
        v_list = self._sparse_propagate(x_spikes, self.v_weights, self.embed_dim, threshold=0.5, gain=gain)

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
        
        for i, past_k in enumerate(self.key_buffer):
            coincidence = len(q_spikes.intersection(past_k))
            if coincidence >= dynamic_threshold:
                routed_v_spikes.update(self.value_buffer[i])

        y_spikes = self._sparse_propagate(list(routed_v_spikes), self.o_weights, self.embed_dim, threshold=0.5, gain=gain)
        
        if learning:
            # Teacher Forcing: Force attention routing to map input patterns to themselves
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