_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/snn_transformer.py",
    "//": "ファイルの日本語タイトル: スパイキング・トランスフォーマーモデル",
    "//": "ファイルの目的や内容: sara_engine.nnモジュールを用いてリファクタリング。自前のFFNをnn.LinearSpikeに、Attentionをnn.SpikeSelfAttentionに置き換え、state_dictベースの保存/復元に統一。"
}

import json
import os
import random
import pickle
from typing import List, Dict, Optional

from sara_engine import nn

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


class SNNTransformerBlock(nn.SNNModule):
    def __init__(self, config: SNNTransformerConfig):
        super().__init__()
        self.config = config
        # nn.SNNModule の機能を利用してAttentionとFFNを構築
        self.attention = nn.SpikeSelfAttention(embed_dim=config.embed_dim, density=0.1, context_size=64)
        self.ffn = nn.Sequential(
            nn.LinearSpike(in_features=config.embed_dim, out_features=config.ffn_dim, density=0.2),
            nn.LinearSpike(in_features=config.ffn_dim, out_features=config.embed_dim, density=0.2)
        )
        self.max_block_spikes = max(1, config.embed_dim // 2)

    def forward(self, spikes: List[int], learning: bool = True) -> List[int]:
        attn_out = self.attention(spikes, learning=learning)
        
        # Residual connection 1
        res1_spikes = list(set(spikes + attn_out))
        
        ffn_out = self.ffn(res1_spikes, learning=learning)
        
        # Residual connection 2
        res2_spikes = list(set(res1_spikes + ffn_out))
        
        if len(res2_spikes) > self.max_block_spikes:
            res2_spikes = random.sample(res2_spikes, self.max_block_spikes)
            
        return res2_spikes


class SpikingTransformerModel(nn.SNNModule):
    def __init__(self, config: SNNTransformerConfig):
        super().__init__()
        self.config = config
        self.context_length = 32
        self.reservoir_size = 4096
        self.total_readout_size = self.reservoir_size + config.embed_dim
        
        # Fixed SDR Map (シード固定で再現するため保存不要)
        self.sdr_map = {}
        random.seed(42)
        for delay in range(self.context_length):
            for tok in range(config.vocab_size):
                self.sdr_map[(delay, tok)] = random.sample(range(self.reservoir_size), 3)
        random.seed()
        
        # SNNTransformerBlock を nn.Sequential で直列に繋ぐ
        layers = [SNNTransformerBlock(config) for _ in range(config.num_layers)]
        self.transformer_layers = nn.Sequential(*layers)
        
        # 動的状態バッファ
        self.delay_buffer: List[int] = []
        
        # 出力層のシナプス重みを状態として登録
        self.readout_synapses: List[Dict[int, float]] = [{} for _ in range(self.total_readout_size)]
        self.register_state("readout_synapses")

    def reset_state(self):
        super().reset_state()
        self.delay_buffer.clear()

    def _get_reservoir_spikes(self, token_id: int) -> List[int]:
        self.delay_buffer.insert(0, token_id)
        if len(self.delay_buffer) > self.context_length:
            self.delay_buffer.pop()
            
        spikes = set()
        for delay, tok in enumerate(self.delay_buffer):
            spikes.update(self.sdr_map.get((delay, tok), []))
        return list(spikes)

    def forward_step(self, token_id: int, learning: bool = True, target_id: Optional[int] = None) -> int:
        res_spikes = self._get_reservoir_spikes(token_id)
        block_spikes = list(set([s % self.config.embed_dim for s in res_spikes]))
        
        # nn.Sequential の forward を呼び出し
        block_spikes = self.transformer_layers(block_spikes, learning=learning)
            
        combined_spikes = res_spikes + [s + self.reservoir_size for s in block_spikes]
        
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
            
        return text + bytes(generated_bytes).decode('utf-8', errors='ignore')

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
        else:
            # 旧バージョン (readout_synapses.json) からの互換フォールバック
            old_weights_path = os.path.join(save_directory, "readout_synapses.json")
            if os.path.exists(old_weights_path):
                with open(old_weights_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    model.readout_synapses = [{int(k): float(v) for k, v in n.items()} for n in loaded]
                    
        return model