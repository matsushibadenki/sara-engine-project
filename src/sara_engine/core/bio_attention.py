# // src/sara_engine/core/bio_attention.py
# // 生物学的スパイキング自己注意機構
# // 目的や内容: 誤差逆伝播法と行列演算を一切使用せず、スパイクタイミング依存シナプス可塑性（STDP）を用いて、ニューロンの同期発火に基づくAttention機構を構築します。

import math
import random

class BioSpikingSelfAttention:
    def __init__(self, seq_len: int, d_model: int, tau_stdp: float = 5.0, learning_rate: float = 0.005, threshold: float = 1.0):
        self.seq_len = seq_len
        self.d_model = d_model
        self.tau_stdp = tau_stdp
        self.learning_rate = learning_rate
        self.threshold = threshold

        self.W_q = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]
        self.W_k = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]
        self.W_v = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_model)]

        self.mem_q = [[0.0 for _ in range(d_model)] for _ in range(seq_len)]
        self.mem_k = [[0.0 for _ in range(d_model)] for _ in range(seq_len)]
        self.mem_v = [[0.0 for _ in range(d_model)] for _ in range(seq_len)]
        self.mem_out = [[0.0 for _ in range(d_model)] for _ in range(seq_len)]

        self.last_spike_q = [[-1 for _ in range(d_model)] for _ in range(seq_len)]
        self.last_spike_k = [[-1 for _ in range(d_model)] for _ in range(seq_len)]

        self.attention_synapses = [[0.01 for _ in range(seq_len)] for _ in range(seq_len)]

    def _apply_weights(self, inputs: list[list[int]], weights: list[list[float]], mem: list[list[float]]) -> list[list[int]]:
        spikes = [[0 for _ in range(self.d_model)] for _ in range(self.seq_len)]
        for i in range(self.seq_len):
            for out_idx in range(self.d_model):
                current = 0.0
                for in_idx in range(self.d_model):
                    if inputs[i][in_idx] > 0:
                        current += weights[in_idx][out_idx]
                mem[i][out_idx] = mem[i][out_idx] * 0.9 + current
                if mem[i][out_idx] >= self.threshold:
                    spikes[i][out_idx] = 1
                    mem[i][out_idx] = 0.0
        return spikes

    def forward(self, x_spikes: list[list[int]], timestep: int) -> list[list[int]]:
        q_spikes = self._apply_weights(x_spikes, self.W_q, self.mem_q)
        k_spikes = self._apply_weights(x_spikes, self.W_k, self.mem_k)
        v_spikes = self._apply_weights(x_spikes, self.W_v, self.mem_v)

        for i in range(self.seq_len):
            for d in range(self.d_model):
                if q_spikes[i][d] > 0:
                    self.last_spike_q[i][d] = timestep
                if k_spikes[i][d] > 0:
                    self.last_spike_k[i][d] = timestep

        for q_idx in range(self.seq_len):
            for k_idx in range(self.seq_len):
                weight_update = 0.0
                for d in range(self.d_model):
                    t_q = self.last_spike_q[q_idx][d]
                    t_k = self.last_spike_k[k_idx][d]
                    if t_q == timestep and t_k >= 0:
                        delta_t = t_q - t_k
                        weight_update += math.exp(-delta_t / self.tau_stdp)
                self.attention_synapses[q_idx][k_idx] += self.learning_rate * weight_update
                self.attention_synapses[q_idx][k_idx] = min(1.0, self.attention_synapses[q_idx][k_idx])

        out_spikes = [[0 for _ in range(self.d_model)] for _ in range(self.seq_len)]
        for i in range(self.seq_len):
            for d in range(self.d_model):
                attended_v = 0.0
                for j in range(self.seq_len):
                    if v_spikes[j][d] > 0:
                        attended_v += self.attention_synapses[i][j]
                
                self.mem_out[i][d] = self.mem_out[i][d] * 0.9 + attended_v
                if self.mem_out[i][d] >= self.threshold:
                    out_spikes[i][d] = 1
                    self.mem_out[i][d] = 0.0
        
        return out_spikes