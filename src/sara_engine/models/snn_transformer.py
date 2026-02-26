_FILE_INFO = {
    "path": "src/sara_engine/models/snn_transformer.py",
    "title": "スパイキング・トランスフォーマーモデル",
    "purpose": "SNN版LayerNorm(恒常性)とDropoutを統合し、TransformerブロックのPre-Norm+Residual構造を生物学的に再現。",
}

from sara_engine.core.spike_attention import SpikeMultiPathwayAttention
from sara_engine import nn
from typing import List, Dict, Optional
import operator
import pickle
import random
import os
import json

class SNNTransformerConfig:
    def __init__(self, vocab_size: int = 65536, embed_dim: int = 128, num_layers: int = 2, ffn_dim: int = 256, num_pathways: int = 4, dropout_p: float = 0.1, target_spikes_ratio: float = 0.25):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.num_pathways = num_pathways
        self.dropout_p = dropout_p
        self.target_spikes_ratio = target_spikes_ratio

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "ffn_dim": self.ffn_dim,
            "num_pathways": self.num_pathways,
            "dropout_p": self.dropout_p,
            "target_spikes_ratio": self.target_spikes_ratio
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

class SNNTransformerBlock(nn.SNNModule):
    def __init__(self, config: SNNTransformerConfig):
        super().__init__()
        self.config = config

        # 目標発火数をエンベディング次元に対する割合で動的に決定
        target_spikes = max(1, int(config.embed_dim * config.target_spikes_ratio))

        # Pre-Norm & Dropout 1
        self.norm1 = nn.SpikeLayerNorm(target_spikes=target_spikes)
        self.dropout1 = nn.SpikeDropout(p=config.dropout_p)

        self.attention = SpikeMultiPathwayAttention(
            embed_dim=config.embed_dim,
            num_pathways=config.num_pathways,
            context_size=128
        )
        
        # Pre-Norm & Dropout 2
        self.norm2 = nn.SpikeLayerNorm(target_spikes=target_spikes)
        self.dropout2 = nn.SpikeDropout(p=config.dropout_p)

        self.ffn = nn.Sequential(
            nn.LinearSpike(in_features=config.embed_dim,
                           out_features=config.ffn_dim, density=0.2),
            nn.LinearSpike(in_features=config.ffn_dim,
                           out_features=config.embed_dim, density=0.2)
        )
        self.max_block_spikes = max(1, config.embed_dim // 2)

    def forward(self, spikes: List[int], learning: bool = True) -> List[int]:
        # Sublayer 1: Pre-Norm -> Attention -> Dropout -> Residual Add
        norm_spikes1 = self.norm1(spikes, learning=learning)
        attn_out = self.attention.forward(norm_spikes1, learning=learning)
        drop_attn = self.dropout1(attn_out, learning=learning)
        res1_spikes = list(set(spikes + drop_attn))

        # Sublayer 2: Pre-Norm -> FFN -> Dropout -> Residual Add
        norm_spikes2 = self.norm2(res1_spikes, learning=learning)
        ffn_out = self.ffn(norm_spikes2, learning=learning)
        drop_ffn = self.dropout2(ffn_out, learning=learning)
        res2_spikes = list(set(res1_spikes + drop_ffn))

        # ネットワーク全体でのスパイク爆発を抑える安全装置
        if len(res2_spikes) > self.max_block_spikes:
            res2_spikes = random.sample(res2_spikes, self.max_block_spikes)

        return res2_spikes

class SpikingTransformerModel(nn.SNNModule):
    def __init__(self, config: SNNTransformerConfig):
        super().__init__()
        self.config = config
        self.context_length = 64
        self.reservoir_size = 8192
        self.total_readout_size = self.reservoir_size + config.embed_dim

        layers = [SNNTransformerBlock(config)
                  for _ in range(config.num_layers)]
        self.transformer_layers = nn.Sequential(*layers)

        self.delay_buffer: List[int] = []
        self.readout_synapses: List[Dict[int, float]] = [
            {} for _ in range(self.total_readout_size)]
        self.register_state("readout_synapses")

    def reset_state(self):
        super().reset_state()
        self.delay_buffer.clear()
        for layer in getattr(self.transformer_layers, 'modules', []):
            if hasattr(layer, 'attention'):
                layer.attention.reset_state()

    def _get_sdr(self, delay: int, tok: int) -> List[int]:
        """動的ハッシュによるSDR生成。メモリを事前消費せず、無制限の語彙（Unicode全体）に対応。
        スパイク数を増大（10->20）することでコンテキスト識別能を向上。
        """
        seed_val = (delay * 73856093) ^ (tok * 19349663) ^ 42
        random.seed(seed_val)
        spikes = random.sample(range(self.reservoir_size), 20)
        random.seed()
        return spikes

    def _get_reservoir_spikes(self, token_id: int) -> List[int]:
        self.delay_buffer.insert(0, token_id)
        if len(self.delay_buffer) > self.context_length:
            self.delay_buffer.pop()

        spikes = set()
        for delay, tok in enumerate(self.delay_buffer):
            spikes.update(self._get_sdr(delay, tok))
        return list(spikes)

    def forward_step(self, token_id: int, learning: bool = True, target_id: Optional[int] = None, refractory_tokens: Optional[List[int]] = None) -> int:
        res_spikes = self._get_reservoir_spikes(token_id)
        block_spikes = list(
            set([s % self.config.embed_dim for s in res_spikes]))

        block_spikes = self.transformer_layers(block_spikes, learning=learning)

        readout_spikes = res_spikes

        out_potentials: Dict[int, float] = {}
        for s in readout_spikes:
            if s < self.total_readout_size:
                for v_idx, w in self.readout_synapses[s].items():
                    out_potentials[v_idx] = out_potentials.get(v_idx, 0.0) + w

        if not learning and refractory_tokens:
            decay_factor = 0.4
            for r_tok in reversed(refractory_tokens):
                if r_tok in out_potentials:
                    out_potentials[r_tok] *= decay_factor
                decay_factor += 0.15
                if decay_factor > 1.0:
                    decay_factor = 1.0

        if out_potentials:
            max_val = max(out_potentials.values())
            if max_val > 0.1:
                predicted_id = max(out_potentials.items(),
                                   key=operator.itemgetter(1))[0]
            else:
                predicted_id = 32
        else:
            predicted_id = 32

        if learning and target_id is not None:
            is_correct = (predicted_id == target_id)
            reward_factor = 4.0 if is_correct else 1.5
            punish_factor = 0.5 if is_correct else 2.5

            active_subset = readout_spikes

            for s in active_subset:
                if s < self.total_readout_size:
                    current_w = self.readout_synapses[s].get(target_id, 0.0)

                    self.readout_synapses[s][target_id] = min(
                        20.0, current_w + (1.5 * reward_factor))

                    if not is_correct and predicted_id in self.readout_synapses[s]:
                        self.readout_synapses[s][predicted_id] -= (
                            2.0 * punish_factor)
                        if self.readout_synapses[s][predicted_id] <= 0:
                            del self.readout_synapses[s][predicted_id]

                    for vocab_id in list(self.readout_synapses[s].keys()):
                        if vocab_id != target_id and vocab_id != predicted_id:
                            self.readout_synapses[s][vocab_id] -= 0.05
                            if self.readout_synapses[s][vocab_id] <= 0:
                                del self.readout_synapses[s][vocab_id]

        return predicted_id

    def learn_sequence(self, text: str):
        input_ids = [ord(c) for c in text] + [0]
        for _replay in range(2):
            self.reset_state()
            for i in range(len(input_ids) - 1):
                self.forward_step(
                    input_ids[i], learning=True, target_id=input_ids[i + 1])

    def generate(self, text: str, max_length: int = 150) -> str:
        input_ids = [ord(c) for c in text]
        self.reset_state()

        first_pred = 32
        for token_id in input_ids:
            first_pred = self.forward_step(token_id, learning=False)

        generated_chars = []
        if first_pred == 0 or first_pred == 32:
            current_token = first_pred
        else:
            current_token = first_pred
        refractory_buffer = []

        for _ in range(max_length):
            if current_token == 0:
                break

            try:
                char = chr(current_token)
                if current_token < 32 and current_token != 0:
                    char = ""
            except ValueError:
                char = ""

            generated_chars.append(char)

            refractory_buffer.append(current_token)
            if len(refractory_buffer) > 6:
                refractory_buffer.pop(0)

            current_token = self.forward_step(
                current_token, learning=False, refractory_tokens=refractory_buffer)

        return text + "".join(generated_chars)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=4)

        state_path = os.path.join(save_directory, "model_state.pkl")
        with open(state_path, "wb") as f:
            pickle.dump(self.state_dict(), f)

    @classmethod
    def from_pretrained(cls, save_directory: str):
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = SNNTransformerConfig.from_dict(json.load(f))

        model = cls(config)
        state_path = os.path.join(save_directory, "model_state.pkl")

        if os.path.exists(state_path):
            with open(state_path, "rb") as f:
                state = pickle.load(f)
            model.load_state_dict(state)

        return model