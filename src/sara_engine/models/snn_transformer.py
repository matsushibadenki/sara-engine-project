_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/snn_transformer.py",
    "//": "ファイルの日本語タイトル: スパイキング・トランスフォーマーモデル",
    "//": "ファイルの目的や内容: Rust拡張を活用した超高速なリードアウト層学習と、完全な自己回帰生成を実現。"
}

import json
import os
import random
from typing import List, Dict, Optional

from sara_engine.core.spike_attention import SpikeSelfAttention
from sara_engine.learning.stdp import STDPLayer

class SNNTransformerConfig:
    def __init__(self, vocab_size: int = 256, embed_dim: int = 128, num_layers: int = 2, ffn_dim: int = 256):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "ffn_dim": self.ffn_dim
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class SNNTransformerBlock:
    def __init__(self, config: SNNTransformerConfig):
        self.config = config
        self.attention = SpikeSelfAttention(config.embed_dim)
        self.ffn_in = STDPLayer(config.embed_dim, config.ffn_dim)
        self.ffn_out = STDPLayer(config.ffn_dim, config.embed_dim)

    def reset_state(self):
        self.attention.k_traces = [0.0] * self.config.embed_dim

    def forward(self, spikes: List[int], learning: bool = True) -> List[int]:
        attn_out = self.attention.forward(spikes, learning)
        res1_spikes = list(set(spikes + attn_out))
        
        ffn_hidden, _ = self.ffn_in.process_step(self._to_dense(res1_spikes, self.ffn_in.num_inputs), reward=1.0 if learning else 0.0)
        ffn_hidden_idx = [i for i, s in enumerate(ffn_hidden) if s == 1]
        
        ffn_out, _ = self.ffn_out.process_step(self._to_dense(ffn_hidden_idx, self.ffn_out.num_inputs), reward=1.0 if learning else 0.0)
        ffn_out_idx = [i for i, s in enumerate(ffn_out) if s == 1]
        
        res2_spikes = list(set(res1_spikes + ffn_out_idx))
        return res2_spikes

    def _to_dense(self, sparse_indices: List[int], size: int) -> List[int]:
        dense = [0] * size
        for idx in sparse_indices:
            if idx < size:
                dense[idx] = 1
        return dense


class SpikingTransformerModel:
    def __init__(self, config: SNNTransformerConfig):
        self.config = config
        self.context_length = 32
        self.reservoir_size = 4096
        self.total_readout_size = self.reservoir_size + config.embed_dim
        
        # SDRマッピング
        self.sdr_map = {}
        random.seed(42)
        for delay in range(self.context_length):
            for tok in range(config.vocab_size):
                self.sdr_map[(delay, tok)] = random.sample(range(self.reservoir_size), 3)
        random.seed()
        
        self.layers = [SNNTransformerBlock(config) for _ in range(config.num_layers)]
        self.delay_buffer: List[int] = []

        # Rust拡張を利用した高速なReadout Layerの初期化
        try:
            from sara_engine import sara_rust_core
            self.rust_readout = sara_rust_core.RustReadoutLayer(self.total_readout_size, config.vocab_size)
            self.use_rust = True
            print("Successfully initialized RustReadoutLayer for ultra-fast learning.")
        except ImportError:
            self.rust_readout = None
            self.use_rust = False
            self.readout_synapses: List[Dict[int, float]] = [{} for _ in range(self.total_readout_size)]
            print("Warning: Rust extension not found. Using slow Python fallback.")

        print(f"Initialized SpikingTransformerModel with config: {config.to_dict()}")

    def reset_state(self):
        self.delay_buffer = []
        for layer in self.layers:
            layer.reset_state()

    def _get_reservoir_spikes(self, token_id: int) -> List[int]:
        self.delay_buffer.insert(0, token_id)
        if len(self.delay_buffer) > self.context_length:
            self.delay_buffer.pop()
            
        spikes = set()
        for delay, tok in enumerate(self.delay_buffer):
            spikes.update(self.sdr_map[(delay, tok)])
        return list(spikes)

    def forward_step(self, token_id: int, learning: bool = True, target_id: Optional[int] = None) -> int:
        res_spikes = self._get_reservoir_spikes(token_id)
        block_spikes = list(set([s % self.config.embed_dim for s in res_spikes]))
        
        for layer in self.layers:
            block_spikes = layer.forward(block_spikes, learning)
            
        combined_spikes = res_spikes + [s + self.reservoir_size for s in block_spikes]
        
        # 学習と推論をRustへ移譲（高速化の要）
        if self.use_rust:
            predicted_id = self.rust_readout.forward_and_learn(combined_spikes, learning, target_id)
        else:
            # Pythonフォールバック
            out_potentials = [0.0] * self.config.vocab_size
            for s in combined_spikes:
                if s < self.total_readout_size:
                    for v_idx, w in self.readout_synapses[s].items():
                        out_potentials[v_idx] += w
                        
            if max(out_potentials) > 0.1:
                predicted_id = out_potentials.index(max(out_potentials))
            else:
                predicted_id = 32

            if learning and target_id is not None:
                for s in combined_spikes:
                    if s < self.total_readout_size:
                        current_w = self.readout_synapses[s].get(target_id, 0.0)
                        self.readout_synapses[s][target_id] = min(15.0, current_w + 1.5)
                        
                        if predicted_id != target_id and predicted_id in self.readout_synapses[s]:
                            self.readout_synapses[s][predicted_id] -= 0.1
                            if self.readout_synapses[s][predicted_id] <= 0:
                                del self.readout_synapses[s][predicted_id]

        return predicted_id

    def learn_sequence(self, text: str):
        input_bytes = list(text.encode('utf-8')) + [0]
        self.reset_state()
        for i in range(len(input_bytes) - 1):
            self.forward_step(input_bytes[i], learning=True, target_id=input_bytes[i+1])

    def generate(self, text: str, max_length: int = 100) -> str:
        input_bytes = list(text.encode('utf-8'))
        self.reset_state()
        
        last_pred = 0
        for byte_val in input_bytes:
            last_pred = self.forward_step(byte_val, learning=False)
            
        generated_bytes = []
        current_token = last_pred
        
        for _ in range(max_length):
            if current_token == 0:
                break
                
            generated_bytes.append(current_token)
            current_token = self.forward_step(current_token, learning=False)
            
        generated_text = bytes(generated_bytes).decode('utf-8', errors='ignore')
        return text + generated_text

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=4)
            
        weights_path = os.path.join(save_directory, "readout_synapses.json")
        with open(weights_path, "w", encoding="utf-8") as f:
            if self.use_rust:
                synapses = self.rust_readout.get_synapses()
            else:
                synapses = self.readout_synapses
            json.dump(synapses, f)
            
        print(f"Model config and synapses successfully saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, save_directory: str):
        config_path = os.path.join(save_directory, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")
            
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
            
        config = SNNTransformerConfig.from_dict(config_dict)
        model = cls(config)
        
        weights_path = os.path.join(save_directory, "readout_synapses.json")
        if os.path.exists(weights_path):
            with open(weights_path, "r", encoding="utf-8") as f:
                loaded_synapses = json.load(f)
                converted_synapses = [
                    {int(k): float(v) for k, v in neuron_dict.items()} 
                    for neuron_dict in loaded_synapses
                ]
                if model.use_rust:
                    model.rust_readout.set_synapses(converted_synapses)
                else:
                    model.readout_synapses = converted_synapses
            print(f"Loaded trained synapses from {weights_path}")
        else:
            print("Warning: Synapses file not found. Initialized with empty synapses.")
            
        return model