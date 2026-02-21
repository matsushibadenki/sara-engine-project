# src/sara_engine/models/spiking_transformer_stdp.py
# スパイキング・トランスフォーマー（Error-Driven Hebbian & STDP）
# 誤差逆伝播法と行列演算を一切排除し、STDPによる自己組織化と、出力層における局所的な誤差駆動型ヘブ則（小脳型学習）を用いて、次トークン予測を学習する生物学的SNNモデル。

import math
import random
from typing import List

class HomeostaticLIFNeuron:
    def __init__(self, base_threshold: float = 0.8, decay: float = 0.85, rest: float = 0.0):
        self.base_threshold = base_threshold
        self.threshold = base_threshold
        self.decay = decay
        self.rest = rest
        self.v = rest
        self.activity_trace = 0.0
        self.last_spike_time = -1

    def step(self, current: float, current_time: int) -> bool:
        self.v = (self.v - self.rest) * self.decay + self.rest + current
        self.activity_trace *= 0.95
        self.threshold = self.base_threshold + self.activity_trace * 0.2

        if self.v >= self.threshold:
            self.v = self.rest
            self.activity_trace += 1.0
            self.last_spike_time = current_time
            return True
        return False

class STDPAttention:
    def __init__(self, seq_len: int, d_model: int, lr_ltp: float = 0.02, lr_ltd: float = 0.01, window: int = 4):
        self.seq_len = seq_len
        self.d_model = d_model
        self.lr_ltp = lr_ltp
        self.lr_ltd = lr_ltd
        self.window = window
        
        self.attn_weights = [[random.uniform(0.1, 0.5) for _ in range(seq_len)] for _ in range(seq_len)]
        
        self.q_neurons = [[HomeostaticLIFNeuron() for _ in range(d_model)] for _ in range(seq_len)]
        self.k_neurons = [[HomeostaticLIFNeuron() for _ in range(d_model)] for _ in range(seq_len)]
        self.v_neurons = [[HomeostaticLIFNeuron() for _ in range(d_model)] for _ in range(seq_len)]
        self.out_neurons = [[HomeostaticLIFNeuron() for _ in range(d_model)] for _ in range(seq_len)]

    def apply_stdp(self, i: int, j: int, t_q: int, t_k: int):
        if t_q == -1 or t_k == -1:
            return
        delta_t = t_q - t_k
        if 0 < delta_t <= self.window:
            self.attn_weights[i][j] += self.lr_ltp * math.exp(-delta_t / self.window)
        elif -self.window <= delta_t <= 0:
            self.attn_weights[i][j] -= self.lr_ltd * math.exp(delta_t / self.window)
        
        if self.attn_weights[i][j] > 1.0:
            self.attn_weights[i][j] = 1.0
        elif self.attn_weights[i][j] < 0.0:
            self.attn_weights[i][j] = 0.0

    def forward_step(self, x: List[List[float]], current_time: int) -> List[List[bool]]:
        q_spikes = [[False] * self.d_model for _ in range(self.seq_len)]
        k_spikes = [[False] * self.d_model for _ in range(self.seq_len)]
        v_spikes = [[False] * self.d_model for _ in range(self.seq_len)]
        out_spikes = [[False] * self.d_model for _ in range(self.seq_len)]

        for i in range(self.seq_len):
            for d in range(self.d_model):
                current = x[i][d]
                q_spikes[i][d] = self.q_neurons[i][d].step(current, current_time)
                k_spikes[i][d] = self.k_neurons[i][d].step(current, current_time)
                v_spikes[i][d] = self.v_neurons[i][d].step(current, current_time)

        for i in range(self.seq_len):
            for j in range(self.seq_len):
                q_spike_time_avg = 0
                k_spike_time_avg = 0
                q_spike_count = 0
                k_spike_count = 0
                
                for d in range(self.d_model):
                    if q_spikes[i][d]:
                        q_spike_time_avg += current_time
                        q_spike_count += 1
                    if k_spikes[j][d]:
                        k_spike_time_avg += current_time
                        k_spike_count += 1
                        
                if q_spike_count > 0 and k_spike_count > 0:
                    t_q_avg = int(q_spike_time_avg / q_spike_count)
                    t_k_avg = int(k_spike_time_avg / k_spike_count)
                    self.apply_stdp(i, j, t_q_avg, t_k_avg)

                weight = self.attn_weights[i][j]
                for d in range(self.d_model):
                    if v_spikes[j][d]:
                        out_spikes[i][d] = self.out_neurons[i][d].step(weight, current_time)
                        
        return out_spikes

class FFNLayer:
    def __init__(self, seq_len: int, d_model: int, d_ff: int):
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.w1 = [[random.uniform(0.1, 0.5) for _ in range(d_model)] for _ in range(d_ff)]
        self.w2 = [[random.uniform(0.1, 0.5) for _ in range(d_ff)] for _ in range(d_model)]
        
        self.hidden_neurons = [[HomeostaticLIFNeuron() for _ in range(d_ff)] for _ in range(seq_len)]
        self.out_neurons = [[HomeostaticLIFNeuron() for _ in range(d_model)] for _ in range(seq_len)]

    def forward_step(self, in_spikes: List[List[bool]], current_time: int) -> List[List[bool]]:
        hidden_spikes = [[False] * self.d_ff for _ in range(self.seq_len)]
        out_spikes = [[False] * self.d_model for _ in range(self.seq_len)]

        for i in range(self.seq_len):
            for h in range(self.d_ff):
                current = 0.0
                for d in range(self.d_model):
                    if in_spikes[i][d]:
                        current += self.w1[h][d]
                hidden_spikes[i][h] = self.hidden_neurons[i][h].step(current, current_time)
                
        for i in range(self.seq_len):
            for d in range(self.d_model):
                current = 0.0
                for h in range(self.d_ff):
                    if hidden_spikes[i][h]:
                        current += self.w2[d][h]
                out_spikes[i][d] = self.out_neurons[i][d].step(current, current_time)
                
        return out_spikes

class SpikingTransformerBlock:
    def __init__(self, seq_len: int, d_model: int, d_ff: int):
        self.attention = STDPAttention(seq_len, d_model)
        self.ffn = FFNLayer(seq_len, d_model, d_ff)

    def forward_step(self, x: List[List[float]], current_time: int) -> List[List[bool]]:
        attn_out_spikes = self.attention.forward_step(x, current_time)
        ffn_out_spikes = self.ffn.forward_step(attn_out_spikes, current_time)
        return ffn_out_spikes

class SpikingTransformer:
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, d_ff: int, num_layers: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.embedding = [[random.uniform(0.1, 0.9) for _ in range(d_model)] for _ in range(vocab_size)]
        self.decoder_w = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(vocab_size)]
        
        self.layers = [SpikingTransformerBlock(seq_len, d_model, d_ff) for _ in range(num_layers)]

    def __call__(self, tokens: List[int], target_tokens: List[int] = None, simulation_steps: int = 15) -> List[int]:
        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
        elif len(tokens) < self.seq_len:
            tokens = tokens + [0] * (self.seq_len - len(tokens))

        if target_tokens is not None:
            if len(target_tokens) > self.seq_len:
                target_tokens = target_tokens[:self.seq_len]
            elif len(target_tokens) < self.seq_len:
                target_tokens = target_tokens + [0] * (self.seq_len - len(target_tokens))

        output_spike_counts = [[0.0] * self.vocab_size for _ in range(self.seq_len)]
        accumulated_final_spikes = [[0] * self.d_model for _ in range(self.seq_len)]

        # SNN Simulation Loop
        for t in range(simulation_steps):
            current_x = [[0.0] * self.d_model for _ in range(self.seq_len)]
            for i, token_id in enumerate(tokens):
                for d in range(self.d_model):
                    if random.random() < self.embedding[token_id][d]:
                        current_x[i][d] = 1.0

            layer_input = current_x
            for layer in self.layers:
                out_spikes = layer.forward_step(layer_input, t)
                layer_input = [[1.5 if spike else 0.0 for spike in row] for row in out_spikes]

            for i in range(self.seq_len):
                for d in range(self.d_model):
                    if layer_input[i][d] > 0:
                        accumulated_final_spikes[i][d] += 1
                        for v in range(self.vocab_size):
                            output_spike_counts[i][v] += self.decoder_w[v][d]

        # Get Predictions
        predicted_tokens = []
        for i in range(self.seq_len):
            max_score = float('-inf')
            best_token = 0
            # Ignore null byte (0) if there are other valid predictions
            for v in range(1, self.vocab_size):
                if output_spike_counts[i][v] > max_score:
                    max_score = output_spike_counts[i][v]
                    best_token = v
            # If nothing spiked properly, keep it 0 or fallback
            if max_score <= 0.0:
                best_token = 0
            predicted_tokens.append(best_token)

        # 誤差逆伝播を使わない Error-Driven Local Hebbian Learning
        if target_tokens is not None:
            lr = 0.1
            for i in range(self.seq_len):
                target_v = target_tokens[i]
                pred_v = predicted_tokens[i]
                in_token = tokens[i]
                
                # If prediction is wrong, locally adjust weights
                if target_v != 0 and target_v != pred_v:
                    for d in range(self.d_model):
                        if accumulated_final_spikes[i][d] > 0:
                            # 1. 出力層（Decoder）の修正: 正解シナプスを強化し、誤予測シナプスを抑制
                            self.decoder_w[target_v][d] += lr
                            self.decoder_w[pred_v][d] -= lr
                            
                            # 2. 入力層（Embedding）の連想学習: 正解予測に寄与する特徴ベクトルを強化
                            if self.decoder_w[target_v][d] > 0:
                                self.embedding[in_token][d] += lr * 0.1
                                if self.embedding[in_token][d] > 1.0: self.embedding[in_token][d] = 1.0
                            else:
                                self.embedding[in_token][d] -= lr * 0.1
                                if self.embedding[in_token][d] < 0.0: self.embedding[in_token][d] = 0.0

        return predicted_tokens