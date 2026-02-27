_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/transformer.py",
    "//": "ファイルの日本語タイトル: スパイキング・トランスフォーマー",
    "//": "ファイルの目的や内容: LIFモデルによるアテンション層を追加し、長文理解とRustコアの連携をサポートするTransformer代替モデル。"
}

import json
from typing import List, Dict, Optional
from sara_engine.core.spike_attention import SpikeSelfAttention
from sara_engine.core.layers import SpikeFeedForward, SpikeNormalization

try:
    from sara_engine.sara_rust_core import SpikeEngine
    HAS_RUST_CORE = True
except ImportError:
    HAS_RUST_CORE = False

class LIFSpikeAttention:
    """
    Phase 3: LIF (Leaky Integrate-and-Fire) based Attention.
    Utilizes membrane potentials to maintain context over longer sequences.
    Supports Rust core for acceleration.
    """
    def __init__(self, embed_dim: int, density: float = 0.05, decay_rate: float = 0.9, use_rust: bool = True):
        self.embed_dim = embed_dim
        self.density = density
        self.decay_rate = decay_rate
        self.use_rust = use_rust and HAS_RUST_CORE
        
        if self.use_rust:
            self.engine = SpikeEngine(decay_rate=self.decay_rate)
        else:
            self.weights: Dict[int, Dict[int, float]] = {}
            self.potentials: Dict[int, float] = {}
            
    def state_dict(self) -> Dict:
        if self.use_rust:
            w_list = self.engine.get_weights()
            weights_dict = {str(i): {str(k): v for k, v in layer.items()} for i, layer in enumerate(w_list) if layer}
            return {"weights": weights_dict, "decay_rate": self.decay_rate}
        else:
            return {"weights": self.weights, "decay_rate": self.decay_rate}

    def load_state_dict(self, state: Dict):
        self.decay_rate = state.get("decay_rate", 0.9)
        weights_data = state.get("weights", {})
        
        if self.use_rust:
            self.engine = SpikeEngine(decay_rate=self.decay_rate)
            if weights_data:
                max_idx = max(int(k) for k in weights_data.keys())
                w_list = [{} for _ in range(max_idx + 1)]
                for k, v in weights_data.items():
                    w_list[int(k)] = {int(post): float(val) for post, val in v.items()}
                self.engine.set_weights(w_list)
        else:
            self.weights = {int(k): {int(post): float(val) for post, val in v.items()} for k, v in weights_data.items()}

    def reset_state(self):
        if self.use_rust:
            self.engine.reset_potentials()
        else:
            self.potentials.clear()

    def forward(self, x_spikes: List[int], learning: bool = True, threshold: float = 0.5, max_out: int = 64) -> List[int]:
        if self.use_rust:
            # Multi-core Rust LIF Engine
            out_spikes = self.engine.propagate(x_spikes, threshold, max_out)
            if learning and out_spikes:
                self.engine.apply_stdp(x_spikes, out_spikes, 0.05)
                self.engine.normalize_weights(1.0)
            return out_spikes
        else:
            # Python fallback LIF
            for k in list(self.potentials.keys()):
                self.potentials[k] *= self.decay_rate
                if self.potentials[k] < 0.01:
                    del self.potentials[k]
                    
            for pre in x_spikes:
                targets = self.weights.get(pre, {})
                for post, w in targets.items():
                    self.potentials[post] = self.potentials.get(post, 0.0) + w
                    
            active = [(k, v) for k, v in self.potentials.items() if v > threshold]
            active.sort(key=lambda x: x[1], reverse=True)
            out_spikes = [k for k, v in active[:max_out]]
            
            for spike in out_spikes:
                self.potentials[spike] = 0.0
                
            if learning and out_spikes:
                for pre in x_spikes:
                    if pre not in self.weights:
                        self.weights[pre] = {}
                    for post in out_spikes:
                        self.weights[pre][post] = min(self.weights[pre].get(post, 0.2) + 0.05, 3.0)
                        
            return out_spikes

class SpikeTransformerBlock:
    """
    A full SNN block replacing the standard Transformer block.
    Uses SpikeSelfAttention or LIFSpikeAttention.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, density: float = 0.05, context_size: int = 64, use_lif: bool = False):
        self.embed_dim = embed_dim
        self.use_lif = use_lif
        if use_lif:
            self.attention = LIFSpikeAttention(embed_dim, density)
        else:
            self.attention = SpikeSelfAttention(embed_dim, density, context_size)
        self.norm1 = SpikeNormalization(target_rate=0.1)
        self.ffn = SpikeFeedForward(embed_dim, hidden_dim, density)
        self.norm2 = SpikeNormalization(target_rate=0.1)

    def state_dict(self) -> Dict:
        return {
            "embed_dim": self.embed_dim,
            "use_lif": self.use_lif,
            "attention": self.attention.state_dict(),
            "norm1": self.norm1.state_dict(),
            "ffn": self.ffn.state_dict(),
            "norm2": self.norm2.state_dict(),
        }

    def load_state_dict(self, state: Dict):
        self.embed_dim = state["embed_dim"]
        self.use_lif = state.get("use_lif", False)
        
        if self.use_lif and not isinstance(self.attention, LIFSpikeAttention):
             self.attention = LIFSpikeAttention(self.embed_dim)
        elif not self.use_lif and not isinstance(self.attention, SpikeSelfAttention):
             self.attention = SpikeSelfAttention(self.embed_dim, 0.05, 64)
             
        self.attention.load_state_dict(state["attention"])
        self.norm1.load_state_dict(state["norm1"])
        self.ffn.load_state_dict(state["ffn"])
        self.norm2.load_state_dict(state["norm2"])

    def reset_state(self):
        self.attention.reset_state()

    def forward(self, x_spikes: List[int], learning: bool = True) -> List[int]:
        attn_out = self.attention.forward(x_spikes, learning=learning)
        res1_spikes = list(set(x_spikes) | set(attn_out))
        norm1_out = self.norm1.forward(res1_spikes, self.embed_dim, learning=learning)
        ffn_out = self.ffn.forward(norm1_out, learning=learning)
        res2_spikes = list(set(norm1_out) | set(ffn_out))
        out_spikes = self.norm2.forward(res2_spikes, self.embed_dim, learning=learning)
        return out_spikes

class SpikeTransformerModel:
    """
    HuggingFace-like wrapper for stacking multiple SpikeTransformerBlocks.
    """
    def __init__(self, num_layers: int, embed_dim: int, hidden_dim: int, density: float = 0.05, context_size: int = 64, use_lif: bool = False):
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.density = density
        self.context_size = context_size
        self.use_lif = use_lif
        self.layers = [SpikeTransformerBlock(embed_dim, hidden_dim, density, context_size, use_lif) for _ in range(num_layers)]

    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()

    def forward(self, x_spikes: List[int], learning: bool = True) -> List[int]:
        current_spikes = x_spikes
        for layer in self.layers:
            current_spikes = layer.forward(current_spikes, learning=learning)
        return current_spikes

    def state_dict(self) -> Dict:
        return {
            "num_layers": self.num_layers,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "density": self.density,
            "context_size": self.context_size,
            "use_lif": self.use_lif,
            "layers": [layer.state_dict() for layer in self.layers]
        }

    def load_state_dict(self, state: Dict):
        self.num_layers = state["num_layers"]
        self.embed_dim = state["embed_dim"]
        self.hidden_dim = state["hidden_dim"]
        self.density = state["density"]
        self.context_size = state["context_size"]
        self.use_lif = state.get("use_lif", False)
        
        if len(self.layers) != self.num_layers or self.layers[0].use_lif != self.use_lif:
            self.layers = [SpikeTransformerBlock(self.embed_dim, self.hidden_dim, self.density, self.context_size, self.use_lif) for _ in range(self.num_layers)]

        for i, layer_state in enumerate(state["layers"]):
            self.layers[i].load_state_dict(layer_state)

    def save_pretrained(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, indent=2)

    def load_pretrained(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.load_state_dict(state)