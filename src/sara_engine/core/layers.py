_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/layers.py",
    "//": "ファイルの日本語タイトル: スパイキング・ニューラル・レイヤー",
    "//": "ファイルの目的や内容: FFNへのTeacher Forcingとシナプス新生の導入、および推論時のNormalization緩和。"
}

import random
from typing import List, Optional, Tuple, Dict, Set

try:
    from .. import sara_rust_core  # type: ignore
    RUST_AVAILABLE = True
except ImportError:
    try:
        import sara_rust_core  # type: ignore
        RUST_AVAILABLE = True
    except ImportError:
        RUST_AVAILABLE = False


class DynamicLiquidLayer:
    def __init__(self, input_size: int, hidden_size: int, decay: float, 
                 density: float = 0.05, input_scale: float = 1.0, 
                 rec_scale: float = 0.8, feedback_scale: float = 0.5,
                 use_rust: Optional[bool] = None,
                 target_rate: float = 0.05):
        self.size = hidden_size
        self.input_size = input_size
        self.decay = decay
        self.density = density
        self.input_scale = input_scale
        self.rec_scale = rec_scale
        self.feedback_scale = feedback_scale
        self.target_rate = target_rate
        
        if use_rust is None:
            self.use_rust = RUST_AVAILABLE
        else:
            self.use_rust = use_rust and RUST_AVAILABLE

        if self.use_rust:
            self.core = sara_rust_core.RustLiquidLayer(input_size, hidden_size, decay, density, feedback_scale)
        else:
            self.v = [0.0] * hidden_size
            self.refractory = [0.0] * hidden_size
            self.dynamic_thresh = [1.0] * hidden_size
            self.trace = [0.0] * hidden_size
            self.in_weights: List[Dict[int, float]] = [{} for _ in range(input_size)]
            for i in range(input_size):
                n_connect = int(hidden_size * density)
                if n_connect > 0:
                    targets = random.sample(range(hidden_size), n_connect)
                    for t in targets:
                        self.in_weights[i][t] = random.uniform(-input_scale, input_scale)
            
            self.rec_weights: List[Dict[int, float]] = [{} for _ in range(hidden_size)]
            rec_density = 0.1
            for i in range(hidden_size):
                n_connect = int(hidden_size * rec_density)
                if n_connect > 0:
                    candidates = [x for x in range(hidden_size) if x != i]
                    if len(candidates) >= n_connect:
                        targets = random.sample(candidates, n_connect)
                        for t in targets:
                            self.rec_weights[i][t] = random.uniform(-rec_scale, rec_scale)
            self.feedback_map: List[List[int]] = [random.sample(range(hidden_size), int(hidden_size * 0.05)) for _ in range(hidden_size)]

    def state_dict(self) -> Dict:
        if self.use_rust: return {}
        return {
            "in_weights": self.in_weights,
            "rec_weights": self.rec_weights,
            "dynamic_thresh": self.dynamic_thresh,
            "v": self.v,
            "refractory": self.refractory
        }

    def load_state_dict(self, state: Dict):
        if self.use_rust or not state: return
        self.in_weights = [{int(k): float(v) for k, v in l.items()} for l in state["in_weights"]]
        self.rec_weights = [{int(k): float(v) for k, v in l.items()} for l in state["rec_weights"]]
        self.dynamic_thresh = state["dynamic_thresh"]
        self.v = state.get("v", [0.0] * self.size)
        self.refractory = state.get("refractory", [0.0] * self.size)

    def forward(self, active_inputs: List[int], prev_active_hidden: List[int], 
                feedback_active: List[int] = [], attention_signal: List[int] = [],
                learning: bool = False, reward: float = 1.0) -> List[int]:
        if self.use_rust:
            return self.core.forward(active_inputs, prev_active_hidden, feedback_active, attention_signal, learning, float(reward))
        
        for i in range(self.size):
            self.v[i] *= self.decay
            if self.refractory[i] > 0.0: self.refractory[i] -= 1.0
            self.trace[i] *= 0.95

        for inp_idx in active_inputs:
            if inp_idx < len(self.in_weights):
                for target, weight in self.in_weights[inp_idx].items(): self.v[target] += weight

        for hid_idx in prev_active_hidden:
            if hid_idx < len(self.rec_weights):
                for target, weight in self.rec_weights[hid_idx].items(): self.v[target] += weight
        
        for fb_idx in feedback_active:
            if fb_idx < len(self.feedback_map):
                for target in self.feedback_map[fb_idx]: self.v[target] += self.feedback_scale
        
        for att_idx in attention_signal:
            if att_idx < self.size: self.v[att_idx] += 1.5

        fired_indices = []
        candidates = [i for i in range(self.size) if self.v[i] >= self.dynamic_thresh[i] and self.refractory[i] <= 0]
        
        max_spikes = int(self.size * 0.15)
        if len(candidates) > max_spikes:
            candidates.sort(key=lambda x: self.v[x], reverse=True)
            fired_indices = candidates[:max_spikes]
        else:
            fired_indices = candidates
            
        fired_set = set(fired_indices)
        for i in range(self.size):
            if i in fired_set:
                self.v[i] = 0.0
                self.refractory[i] = random.uniform(2.0, 5.0) if learning else 3.0
                self.trace[i] += 1.0
                self.dynamic_thresh[i] += 0.05
            else:
                self.dynamic_thresh[i] += (self.target_rate - 0.05) * 0.01
            
            self.dynamic_thresh[i] = max(0.5, min(5.0, self.dynamic_thresh[i]))

        if learning and prev_active_hidden:
            for pre_id in prev_active_hidden:
                if pre_id < len(self.rec_weights):
                    for target_id in self.rec_weights[pre_id].keys():
                        if target_id in fired_set:
                            self.rec_weights[pre_id][target_id] = min(2.0, self.rec_weights[pre_id][target_id] + 0.02 * reward)
                        else:
                            self.rec_weights[pre_id][target_id] = max(-2.0, self.rec_weights[pre_id][target_id] - 0.005 * reward)
        return fired_indices

    def reset(self):
        if self.use_rust: self.core.reset()
        else:
            self.v = [0.0] * self.size
            self.refractory = [0.0] * self.size
            self.dynamic_thresh = [1.0] * self.size

class SpikeNormalization:
    def __init__(self, target_rate: float = 0.1, adaptation_rate: float = 0.01):
        self.target_rate = target_rate
        self.adaptation_rate = adaptation_rate
        self.threshold_offsets: Dict[int, float] = {}

    def state_dict(self) -> Dict:
        return {"threshold_offsets": self.threshold_offsets}

    def load_state_dict(self, state: Dict):
        self.threshold_offsets = {int(k): float(v) for k, v in state["threshold_offsets"].items()}

    def forward(self, spikes: List[int], dim: int, learning: bool = True) -> List[int]:
        normalized_spikes = []
        for s in spikes:
            offset = self.threshold_offsets.get(s, 0.0)
            if learning:
                if random.random() > offset: normalized_spikes.append(s)
            else:
                # 推論時は決定論的に。オフセットの閾値を緩和し、抑制されすぎないようにする（0.5 -> 0.99）
                if offset < 0.99: normalized_spikes.append(s)
        
        if learning:
            current_rate = len(spikes) / max(1, dim)
            error = current_rate - self.target_rate
            for s in spikes:
                curr = self.threshold_offsets.get(s, 0.0)
                self.threshold_offsets[s] = max(0.0, min(0.9, curr + self.adaptation_rate * error))
        return normalized_spikes

class SpikeFeedForward:
    def __init__(self, embed_dim: int, hidden_dim: int, density: float = 0.1):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.w1 = self._init_sparse_weights(embed_dim, hidden_dim, density)
        self.w2 = self._init_sparse_weights(hidden_dim, embed_dim, density)

    def state_dict(self) -> Dict:
        return {"w1": self.w1, "w2": self.w2}

    def load_state_dict(self, state: Dict):
        self.w1 = [{int(k): float(v) for k, v in l.items()} for l in state["w1"]]
        self.w2 = [{int(k): float(v) for k, v in l.items()} for l in state["w2"]]

    def _init_sparse_weights(self, in_dim: int, out_dim: int, density: float) -> List[Dict[int, float]]:
        weights = [{} for _ in range(in_dim)]
        for i in range(in_dim):
            num = max(1, int(out_dim * density))
            for t in random.sample(range(out_dim), num): weights[i][t] = random.uniform(-1.0, 1.0)
        return weights

    def _sparse_propagate(self, active_spikes: List[int], weights: List[Dict[int, float]], out_size: int, threshold: float = 0.5, gain: float = 1.0) -> List[int]:
        potentials = [0.0] * out_size
        for s in active_spikes:
            if s < len(weights):
                for t, w in weights[s].items(): potentials[t] += w * gain
        return [i for i, p in enumerate(potentials) if p > threshold]

    def _apply_stdp(self, pre_spikes: List[int], post_spikes: List[int], weights: List[Dict[int, float]], lr: float = 0.05):
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

    def forward(self, spikes: List[int], learning: bool = True) -> List[int]:
        gain = 1.0 if learning else 12.0
        h = self._sparse_propagate(spikes, self.w1, self.hidden_dim, threshold=0.5, gain=gain)
        out = self._sparse_propagate(h, self.w2, self.embed_dim, threshold=0.5, gain=gain)
        if learning:
            forced_h = list(set(h))
            if len(forced_h) < max(1, len(spikes) // 2):
                forced_h.extend([(s * 7) % self.hidden_dim for s in spikes])
                forced_h = list(set(forced_h))
                
            forced_out = list(set(out) | set(spikes))
            
            self._apply_stdp(spikes, forced_h, self.w1)
            self._apply_stdp(forced_h, forced_out, self.w2)
        return out