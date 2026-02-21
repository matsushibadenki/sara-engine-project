# ディレクトリパス: src/sara_engine/core/transformer.py
# タイトル: スパイクトランスフォーマーモデル (バックグラウンド活性・恒常性強化版)
# 目的: 信号消失時に微弱なノイズを混入させて沈黙を打破し、シナプス・スケーリングで活性を維持する。

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
        self.target_active = max(4, int(size * target_sparsity))
        rng = random.Random(42)
        self.priority = [rng.random() for _ in range(size)]

    def normalize(self, spikes: List[int]) -> List[int]:
        if not spikes:
            return []
        if len(spikes) > self.target_active:
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
        
        num_active = max(2, int(self.d_model * self.density))
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
        self._init_sparse_weights(self.W1, d_model, d_ff, 0.2)
        self._init_sparse_weights(self.W2, d_ff, d_model, 0.2)
        
        self.hidden_active = max(4, int(d_ff * target_sparsity))
        self.out_active = max(4, int(d_model * target_sparsity))
        self.h_thresholds = [1.0] * d_ff
        self.o_thresholds = [1.0] * d_model
        
        self.target_weight_sum = 2.0 # シナプス・スケーリングの目標値

    def _init_sparse_weights(self, weights: List[Dict[int, float]], in_size: int, out_size: int, density: float):
        rng = random.Random(42)
        for i in range(in_size):
            num_connections = max(1, int(out_size * density))
            targets = rng.sample(range(out_size), num_connections)
            for t in targets:
                weights[i][t] = random.uniform(0.1, 0.5)

    def _apply_scaling(self, weights: List[Dict[int, float]], active_inputs: List[int]):
        """発火した入力に対してシナプス重みの総和を調整する（恒常性）"""
        for s in active_inputs:
            if not weights[s]: continue
            current_sum = sum(weights[s].values())
            if current_sum > 0:
                scale = self.target_weight_sum / current_sum
                for t in weights[s]:
                    weights[s][t] *= scale

    def _wta_adaptive(self, potentials: List[float], thresholds: List[float], k: int) -> List[int]:
        active_candidates = []
        for i, p in enumerate(potentials):
            if p >= thresholds[i]:
                active_candidates.append((i, p))
        
        if not active_candidates:
            # 信号が途絶えた場合、閾値を大幅に下げて次回の感度を確保
            for i in range(len(thresholds)):
                thresholds[i] = max(0.05, thresholds[i] * 0.8)
            return []

        active_candidates.sort(key=lambda x: x[1], reverse=True)
        firing = sorted([i for i, p in active_candidates[:k]])

        for i in range(len(thresholds)):
            if i in firing:
                thresholds[i] = min(5.0, thresholds[i] + 0.1)
            else:
                thresholds[i] = max(0.05, thresholds[i] - 0.01)
        
        return firing

    def forward(self, spikes: List[int], learning: bool = True) -> List[int]:
        # 沈黙打破: 入力が空でも微弱なランダムスパイクを注入して活性を維持する
        if not spikes and learning:
            spikes = random.sample(range(self.d_model), 2)

        hidden_potentials = [0.0] * self.d_ff
        for s in spikes:
            for t, w in self.W1[s].items():
                hidden_potentials[t] += w
        hidden_spikes = self._wta_adaptive(hidden_potentials, self.h_thresholds, self.hidden_active)

        out_potentials = [0.0] * self.d_model
        for s in hidden_spikes:
            for t, w in self.W2[s].items():
                out_potentials[t] += w
        out_spikes = self._wta_adaptive(out_potentials, self.o_thresholds, self.out_active)

        if learning:
            ltd_rate = 0.0002
            ltp_rate = 0.015
            min_weight = 0.05

            for s in spikes:
                for t, w in self.W1[s].items():
                    if t in hidden_spikes:
                        self.W1[s][t] = min(1.0, w + ltp_rate)
                    else:
                        self.W1[s][t] = max(min_weight, w - ltd_rate)
            self._apply_scaling(self.W1, spikes)

            for s in hidden_spikes:
                for t, w in self.W2[s].items():
                    if t in out_spikes:
                        self.W2[s][t] = min(1.0, w + ltp_rate)
                    else:
                        self.W2[s][t] = max(min_weight, w - ltd_rate)
            self._apply_scaling(self.W2, hidden_spikes)
                        
        return out_spikes

class SpikeTransformerBlock:
    def __init__(self, d_model: int, num_heads: int, ffn_hidden: int):
        self.attention = SpikingSelfAttention(sdr_size=d_model, num_heads=num_heads, decay_rate=0.8)
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
        self.pe = SpikePositionalEncoding(d_model, density=0.15)
        self.blocks = [SpikeTransformerBlock(d_model, num_heads, d_model * 2) for _ in range(num_layers)]
        self.embedding_table: Dict[int, List[int]] = {}
        
    def _get_embedding(self, token_id: int) -> List[int]:
        if token_id not in self.embedding_table:
            num_active = max(5, int(self.d_model * 0.1))
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