_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/spike_attention.py",
    "//": "ファイルの日本語タイトル: 連合学習強化型 スパイキング・アテンション",
    "//": "ファイルの目的や内容: スパイクの爆発およびクロストークを防ぐためのスパースルーティング。Rustエンジンによる高速化と、TransformersのMulti-Headに相当するMulti-Pathway Attentionを実装。"
}

import random
from typing import List, Dict, Set

try:
    import sara_rust_core
    HAS_RUST_CORE = True
except ImportError:
    HAS_RUST_CORE = False

class SpikeSelfAttention:
    def __init__(self, embed_dim: int, density: float = 0.05, context_size: int = 64, coincidence_threshold: int = 2):
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.base_coincidence_threshold = coincidence_threshold
        self.use_rust = HAS_RUST_CORE
        
        initial_q = self._init_sparse_weights(embed_dim, embed_dim, density)
        initial_k = self._init_sparse_weights(embed_dim, embed_dim, density)
        initial_v = self._init_sparse_weights(embed_dim, embed_dim, density)
        initial_o = self._init_sparse_weights(embed_dim, embed_dim, density)
        
        if self.use_rust:
            self.q_engine = sara_rust_core.SpikeEngine()
            self.k_engine = sara_rust_core.SpikeEngine()
            self.v_engine = sara_rust_core.SpikeEngine()
            self.o_engine = sara_rust_core.SpikeEngine()
            
            self.q_engine.set_weights(initial_q)
            self.k_engine.set_weights(initial_k)
            self.v_engine.set_weights(initial_v)
            self.o_engine.set_weights(initial_o)
        else:
            self.q_weights = initial_q
            self.k_weights = initial_k
            self.v_weights = initial_v
            self.o_weights = initial_o
            
        self.key_buffer: List[Set[int]] = []
        self.value_buffer: List[Set[int]] = []
        self.step_counter = 0

    def _init_sparse_weights(self, in_dim: int, out_dim: int, density: float) -> List[Dict[int, float]]:
        weights: List[Dict[int, float]] = [{} for _ in range(in_dim)]
        for i in range(in_dim):
            num_connections = max(1, int(out_dim * density))
            targets = random.sample(range(out_dim), num_connections)
            for t in targets:
                weights[i][t] = random.uniform(0.0, 1.0)
        return weights

    def reset_state(self):
        self.key_buffer.clear()
        self.value_buffer.clear()
        self.step_counter = 0

    def state_dict(self) -> Dict:
        if self.use_rust:
            q_w = self.q_engine.get_weights()
            k_w = self.k_engine.get_weights()
            v_w = self.v_engine.get_weights()
            o_w = self.o_engine.get_weights()
        else:
            q_w = self.q_weights
            k_w = self.k_weights
            v_w = self.v_weights
            o_w = self.o_weights
            
        return {
            "embed_dim": self.embed_dim,
            "context_size": self.context_size,
            "base_coincidence_threshold": self.base_coincidence_threshold,
            "q_weights": q_w,
            "k_weights": k_w,
            "v_weights": v_w,
            "o_weights": o_w
        }

    def load_state_dict(self, state: Dict):
        self.embed_dim = state["embed_dim"]
        self.context_size = state["context_size"]
        self.base_coincidence_threshold = state["base_coincidence_threshold"]
        
        q_w = [{int(k): float(v) for k, v in layer.items()} for layer in state["q_weights"]]
        k_w = [{int(k): float(v) for k, v in layer.items()} for layer in state["k_weights"]]
        v_w = [{int(k): float(v) for k, v in layer.items()} for layer in state["v_weights"]]
        o_w = [{int(k): float(v) for k, v in layer.items()} for layer in state["o_weights"]]
        
        if self.use_rust:
            self.q_engine.set_weights(q_w)
            self.k_engine.set_weights(k_w)
            self.v_engine.set_weights(v_w)
            self.o_engine.set_weights(o_w)
        else:
            self.q_weights = q_w
            self.k_weights = k_w
            self.v_weights = v_w
            self.o_weights = o_w

    def _sparse_propagate(self, active_spikes: List[int], engine_or_weights, out_size: int, threshold: float = 0.5, gain: float = 1.0) -> List[int]:
        max_spikes = max(1, int(out_size * 0.1))
        if self.use_rust:
            return engine_or_weights.propagate(active_spikes, threshold / gain, max_spikes)
        else:
            potentials = [0.0] * out_size
            for s in active_spikes:
                if s < len(engine_or_weights):
                    for t, w in engine_or_weights[s].items():
                        if t < out_size:
                            potentials[t] += w * gain
            
            active = [(i, p) for i, p in enumerate(potentials) if p > threshold]
            active.sort(key=lambda x: x[1], reverse=True)
            return [i for i, p in active[:max_spikes]]

    def _apply_stdp(self, pre_spikes: List[int], post_spikes: List[int], engine_or_weights, lr: float = 0.05):
        if self.use_rust:
            engine_or_weights.apply_stdp(pre_spikes, post_spikes, lr)
        else:
            post_set = set(post_spikes)
            for pre in pre_spikes:
                if pre < len(engine_or_weights):
                    for target in list(engine_or_weights[pre].keys()):
                        if target in post_set:
                            engine_or_weights[pre][target] = min(3.0, engine_or_weights[pre][target] + lr)
                        else:
                            engine_or_weights[pre][target] = max(0.0, engine_or_weights[pre][target] - lr * 0.05)
                    
                    for target in post_set:
                        if target not in engine_or_weights[pre]:
                            if random.random() < 0.1:
                                engine_or_weights[pre][target] = 0.2

    def _decay_weights(self, engine_or_weights, decay_rate: float = 0.99):
        if self.use_rust:
            weights = engine_or_weights.get_weights()
            for w_dict in weights:
                for target in list(w_dict.keys()):
                    w_dict[target] *= decay_rate
                    if w_dict[target] < 0.05:
                        del w_dict[target]
            engine_or_weights.set_weights(weights)

    def forward(self, x_spikes: List[int], learning: bool = True) -> List[int]:
        gain = 1.0 if learning else 1.2
        
        q_src = self.q_engine if self.use_rust else self.q_weights
        k_src = self.k_engine if self.use_rust else self.k_weights
        v_src = self.v_engine if self.use_rust else self.v_weights
        o_src = self.o_engine if self.use_rust else self.o_weights
        
        q_list = self._sparse_propagate(x_spikes, q_src, self.embed_dim, threshold=0.5, gain=gain)
        k_list = self._sparse_propagate(x_spikes, k_src, self.embed_dim, threshold=0.5, gain=gain)
        v_list = self._sparse_propagate(x_spikes, v_src, self.embed_dim, threshold=0.5, gain=gain)

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

        y_spikes = self._sparse_propagate(list(routed_v_spikes), o_src, self.embed_dim, threshold=0.5, gain=gain)
        
        if learning:
            self.step_counter += 1
            forced_q = list(set(q_list) | set(x_spikes))
            forced_k = list(set(k_list) | set(x_spikes))
            forced_v = list(set(v_list) | set(x_spikes))
            
            self._apply_stdp(x_spikes, forced_q, q_src)
            self._apply_stdp(x_spikes, forced_k, k_src)
            self._apply_stdp(x_spikes, forced_v, v_src)
            
            forced_y = list(set(y_spikes) | set(x_spikes))
            pre_y = list(routed_v_spikes) if routed_v_spikes else x_spikes
            self._apply_stdp(pre_y, forced_y, o_src)
            
            if self.use_rust and self.step_counter % 100 == 0:
                self._decay_weights(q_src)
                self._decay_weights(k_src)
                self._decay_weights(v_src)
                self._decay_weights(o_src)
            
        return y_spikes


class SpikeMultiPathwayAttention:
    """
    Biological alternative to Transformers' Multi-Head Attention.
    Utilizes multiple distinct pathways with varying receptive field densities and thresholds
    to extract diverse features without matrix multiplications.
    """
    def __init__(self, embed_dim: int, num_pathways: int = 4, context_size: int = 64):
        self.embed_dim = embed_dim
        self.num_pathways = num_pathways
        self.pathways = []
        
        for i in range(num_pathways):
            density = 0.02 + (i * 0.03) 
            threshold = max(1, 3 - (i % 3)) 
            self.pathways.append(SpikeSelfAttention(
                embed_dim=embed_dim, 
                density=density, 
                context_size=context_size,
                coincidence_threshold=threshold
            ))
            
    def reset_state(self):
        for pathway in self.pathways:
            pathway.reset_state()
            
    def state_dict(self) -> Dict:
        return {
            "embed_dim": self.embed_dim,
            "num_pathways": self.num_pathways,
            "pathways": [p.state_dict() for p in self.pathways]
        }
        
    def load_state_dict(self, state: Dict):
        self.embed_dim = state["embed_dim"]
        self.num_pathways = state["num_pathways"]
        for p, s in zip(self.pathways, state["pathways"]):
            p.load_state_dict(s)
            
    def forward(self, x_spikes: List[int], learning: bool = True) -> List[int]:
        out_spikes = set()
        for pathway in self.pathways:
            pathway_out = pathway.forward(x_spikes, learning=learning)
            out_spikes.update(pathway_out)
            
        out_list = list(out_spikes)
        max_spikes = max(1, int(self.embed_dim * 0.3))
        if len(out_list) > max_spikes:
            out_list = random.sample(out_list, max_spikes)
            
        return out_list