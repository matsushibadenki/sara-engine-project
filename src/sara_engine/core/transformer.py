_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/transformer.py",
    "//": "タイトル: スパイクトランスフォーマーモデル (安定化・LTD導入版)",
    "//": "目的: カオスなランダム間引きを廃止して位置情報を安定させ、FFNにLTD(長期抑圧)を追加してシナプス飽和を防ぐ。"
}

import random
from typing import List, Dict

try:
    from sara_engine import sara_rust_core
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("Warning: sara_rust_core not found. Using slow Python backend for FFN.")

from sara_engine.core.spike_attention import SpikingSelfAttention

class SpikeHomeostaticNorm:
    def __init__(self, size: int, target_sparsity: float = 0.1):
        self.size = size
        self.target_active = max(1, int(size * target_sparsity))
        # ニューロン固有の優先度を固定で割り当て、決定論的かつ安定した間引きを行う
        rng = random.Random(42)
        self.priority = [rng.random() for _ in range(size)]

    def normalize(self, spikes: List[int]) -> List[int]:
        if len(spikes) > self.target_active:
            # 優先度が高い順にソートして抽出（バタフライ効果を防ぐ）
            spikes_with_priority = [(s, self.priority[s]) for s in spikes]
            spikes_with_priority.sort(key=lambda x: x[1], reverse=True)
            return sorted([s for s, p in spikes_with_priority[:self.target_active]])
        return spikes

class SpikePositionalEncoding:
    def __init__(self, d_model: int, density: float = 0.1):
        self.d_model = d_model
        self.density = density
        self.pos_spikes: Dict[int, List[int]] = {}
        
    def get_pe(self, pos: int) -> List[int]:
        if pos in self.pos_spikes:
            return self.pos_spikes[pos]
        
        num_active = max(1, int(self.d_model * self.density))
        local_rng = random.Random(42 + pos)
        spikes = sorted(local_rng.sample(range(self.d_model), num_active))
        
        self.pos_spikes[pos] = spikes
        return spikes

class SpikePositionwiseFFN:
    def __init__(self, d_model: int, d_ff: int, target_sparsity: float = 0.1):
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = [{} for _ in range(d_model)]
        self.W2 = [{} for _ in range(d_ff)]
        self._init_sparse_weights(self.W1, d_model, d_ff, 0.1)
        self._init_sparse_weights(self.W2, d_ff, d_model, 0.1)
        
        self.hidden_active = max(1, int(d_ff * target_sparsity))
        self.out_active = max(1, int(d_model * target_sparsity))

    def _init_sparse_weights(self, weights: List[Dict[int, float]], in_size: int, out_size: int, density: float):
        rng = random.Random(42)
        for i in range(in_size):
            num_connections = max(1, int(out_size * density))
            targets = rng.sample(range(out_size), num_connections)
            for t in targets:
                weights[i][t] = random.uniform(0.05, 0.2)

    def _wta(self, potentials: List[float], k: int) -> List[int]:
        active_neurons = [(i, p) for i, p in enumerate(potentials) if p > 0.0]
        active_neurons.sort(key=lambda x: x[1], reverse=True)
        return sorted([i for i, p in active_neurons[:k]])

    def forward(self, spikes: List[int], learning: bool = True) -> List[int]:
        if RUST_AVAILABLE:
            hidden_spikes = sara_rust_core.sparse_propagate_and_wta(
                spikes, self.W1, self.d_ff, self.hidden_active
            )
            out_spikes = sara_rust_core.sparse_propagate_and_wta(
                hidden_spikes, self.W2, self.d_model, self.out_active
            )
        else:
            hidden_potentials = [0.0] * self.d_ff
            for s in spikes:
                for t, w in self.W1[s].items():
                    hidden_potentials[t] += w
            hidden_spikes = self._wta(hidden_potentials, self.hidden_active)

            out_potentials = [0.0] * self.d_model
            for s in hidden_spikes:
                for t, w in self.W2[s].items():
                    out_potentials[t] += w
            out_spikes = self._wta(out_potentials, self.out_active)

        if learning:
            # W1: 同時発火したシナプスは強化(LTP)、それ以外は減衰(LTD)
            for s in spikes:
                for t, w in self.W1[s].items():
                    if t in hidden_spikes:
                        self.W1[s][t] = min(1.0, w + 0.01)
                    else:
                        self.W1[s][t] = max(0.0, w - 0.002)
            # W2: 同時発火したシナプスは強化(LTP)、それ以外は減衰(LTD)
            for s in hidden_spikes:
                for t, w in self.W2[s].items():
                    if t in out_spikes:
                        self.W2[s][t] = min(1.0, w + 0.01)
                    else:
                        self.W2[s][t] = max(0.0, w - 0.002)
                        
        return out_spikes

class SpikeTransformerBlock:
    def __init__(self, d_model: int, num_heads: int, ffn_hidden: int):
        self.attention = SpikingSelfAttention(sdr_size=d_model, num_heads=num_heads, decay_rate=0.9)
        self.norm1 = SpikeHomeostaticNorm(size=d_model, target_sparsity=0.1)
        self.ffn = SpikePositionwiseFFN(d_model, ffn_hidden, target_sparsity=0.1)
        self.norm2 = SpikeHomeostaticNorm(size=d_model, target_sparsity=0.1)
        
    def forward(self, x_spikes: List[int], learning: bool = True) -> List[int]:
        attn_out = self.attention.forward(x_spikes)
        x_post_attn = sorted(list(set(x_spikes).union(set(attn_out))))
        x_post_attn = self.norm1.normalize(x_post_attn)
        
        ffn_out = self.ffn.forward(x_post_attn, learning=learning)
        output = sorted(list(set(x_post_attn).union(set(ffn_out))))
        output = self.norm2.normalize(output)
        
        return output

class SpikeTransformer:
    def __init__(self, vocab_size: int, d_model: int = 256, num_layers: int = 2, num_heads: int = 4):
        self.d_model = d_model
        self.pe = SpikePositionalEncoding(d_model, density=0.1)
        self.blocks = [SpikeTransformerBlock(d_model, num_heads, d_model * 2) for _ in range(num_layers)]
        self.embedding_table: Dict[int, List[int]] = {}
        
    def _get_embedding(self, token_id: int) -> List[int]:
        if token_id not in self.embedding_table:
            num_active = max(1, int(self.d_model * 0.05))
            local_rng = random.Random(token_id * 777)
            self.embedding_table[token_id] = sorted(local_rng.sample(range(self.d_model), num_active))
        return self.embedding_table[token_id]

    def reset_context(self):
        for block in self.blocks:
            block.attention.reset()

    def compute(self, token_ids: List[int], learning: bool = True) -> List[List[int]]:
        self.reset_context()
        outputs = []
        
        for pos, token in enumerate(token_ids):
            emb = self._get_embedding(token)
            pe = self.pe.get_pe(pos)
            x = sorted(list(set(emb).union(set(pe))))
            
            for block in self.blocks:
                x = block.forward(x, learning=learning)
            
            outputs.append(x)
            
        return outputs