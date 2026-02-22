_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/transformer.py",
    "//": "ファイルの日本語タイトル: スパイキング・トランスフォーマー",
    "//": "ファイルの目的や内容: 複数層のスタックを可能にするModelクラスを追加し、JSONベースのsave_pretrained/load_pretrainedを実装。"
}

import json
from typing import List, Dict
from sara_engine.core.spike_attention import SpikeSelfAttention
from sara_engine.core.layers import SpikeFeedForward, SpikeNormalization

class SpikeTransformerBlock:
    """
    A full SNN block replacing the standard Transformer block.
    Uses SpikeSelfAttention, SpikeFeedForward, and SpikeNormalization.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, density: float = 0.05, context_size: int = 64):
        self.embed_dim = embed_dim
        self.attention = SpikeSelfAttention(embed_dim, density, context_size)
        self.norm1 = SpikeNormalization(target_rate=0.1)
        self.ffn = SpikeFeedForward(embed_dim, hidden_dim, density)
        self.norm2 = SpikeNormalization(target_rate=0.1)

    def state_dict(self) -> Dict:
        return {
            "embed_dim": self.embed_dim,
            "attention": self.attention.state_dict(),
            "norm1": self.norm1.state_dict(),
            "ffn": self.ffn.state_dict(),
            "norm2": self.norm2.state_dict(),
        }

    def load_state_dict(self, state: Dict):
        self.embed_dim = state["embed_dim"]
        self.attention.load_state_dict(state["attention"])
        self.norm1.load_state_dict(state["norm1"])
        self.ffn.load_state_dict(state["ffn"])
        self.norm2.load_state_dict(state["norm2"])

    def reset_state(self):
        self.attention.reset_state()

    def forward(self, x_spikes: List[int], learning: bool = True) -> List[int]:
        # 1. Multi-Head Spike Attention (Spike routing based on coincidence)
        attn_out = self.attention.forward(x_spikes, learning=learning)
        
        # 2. Residual connection 1 (Spike set union)
        res1_spikes = list(set(x_spikes) | set(attn_out))
        
        # 3. Normalization 1
        norm1_out = self.norm1.forward(res1_spikes, self.embed_dim, learning=learning)
        
        # 4. Feed Forward Network
        ffn_out = self.ffn.forward(norm1_out, learning=learning)
        
        # 5. Residual connection 2
        res2_spikes = list(set(norm1_out) | set(ffn_out))
        
        # 6. Normalization 2
        out_spikes = self.norm2.forward(res2_spikes, self.embed_dim, learning=learning)
        
        return out_spikes


class SpikeTransformerModel:
    """
    HuggingFace-like wrapper for stacking multiple SpikeTransformerBlocks.
    """
    def __init__(self, num_layers: int, embed_dim: int, hidden_dim: int, density: float = 0.05, context_size: int = 64):
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.density = density
        self.context_size = context_size
        self.layers = [SpikeTransformerBlock(embed_dim, hidden_dim, density, context_size) for _ in range(num_layers)]

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
            "layers": [layer.state_dict() for layer in self.layers]
        }

    def load_state_dict(self, state: Dict):
        self.num_layers = state["num_layers"]
        self.embed_dim = state["embed_dim"]
        self.hidden_dim = state["hidden_dim"]
        self.density = state["density"]
        self.context_size = state["context_size"]
        for i, layer_state in enumerate(state["layers"]):
            self.layers[i].load_state_dict(layer_state)

    def save_pretrained(self, filepath: str):
        """Save the SNN model state directly to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, indent=2)

    def load_pretrained(self, filepath: str):
        """Load the SNN model state from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.load_state_dict(state)