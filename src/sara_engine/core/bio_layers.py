# // src/sara_engine/core/bio_layers.py
# // 生物学的フィードフォワードおよびホメオスタシス層
# // 目的や内容: 逆伝播や行列演算を排除した、純粋なループとSTDPベースの多層ネットワークおよび閾値調整層（LayerNormの代替）を提供します。

import random
import math

class BioHomeostasis:
    def __init__(self, seq_len: int, d_model: int, target_rate: float = 0.1, adaptation_rate: float = 0.01):
        self.seq_len = seq_len
        self.d_model = d_model
        self.target_rate = target_rate
        self.adaptation_rate = adaptation_rate
        self.thresholds = [[1.0 for _ in range(d_model)] for _ in range(seq_len)]
        self.firing_rates = [[0.0 for _ in range(d_model)] for _ in range(seq_len)]

    def forward(self, spikes: list[list[int]]) -> list[list[int]]:
        out_spikes = [[0 for _ in range(self.d_model)] for _ in range(self.seq_len)]
        for i in range(self.seq_len):
            for d in range(self.d_model):
                self.firing_rates[i][d] = self.firing_rates[i][d] * 0.95 + spikes[i][d] * 0.05
                rate_error = self.firing_rates[i][d] - self.target_rate
                self.thresholds[i][d] += self.adaptation_rate * rate_error
                self.thresholds[i][d] = max(0.1, self.thresholds[i][d])
                
                if spikes[i][d] > 0 and self.thresholds[i][d] <= 1.5:
                    out_spikes[i][d] = 1
        return out_spikes

class BioSpikingFFN:
    def __init__(self, seq_len: int, d_model: int, d_ff: int, threshold: float = 1.0):
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.threshold = threshold

        self.W_1 = [[random.uniform(-0.1, 0.1) for _ in range(d_ff)] for _ in range(d_model)]
        self.W_2 = [[random.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(d_ff)]

        self.mem_hidden = [[0.0 for _ in range(d_ff)] for _ in range(seq_len)]
        self.mem_out = [[0.0 for _ in range(d_model)] for _ in range(seq_len)]
        
        self.last_spike_in = [[-1 for _ in range(d_model)] for _ in range(seq_len)]
        self.last_spike_hidden = [[-1 for _ in range(d_ff)] for _ in range(seq_len)]

    def forward(self, x_spikes: list[list[int]], timestep: int) -> list[list[int]]:
        hidden_spikes = [[0 for _ in range(self.d_ff)] for _ in range(self.seq_len)]
        out_spikes = [[0 for _ in range(self.d_model)] for _ in range(self.seq_len)]

        for i in range(self.seq_len):
            for in_idx in range(self.d_model):
                if x_spikes[i][in_idx] > 0:
                    self.last_spike_in[i][in_idx] = timestep

            for h_idx in range(self.d_ff):
                current = 0.0
                for in_idx in range(self.d_model):
                    if x_spikes[i][in_idx] > 0:
                        current += self.W_1[in_idx][h_idx]
                
                self.mem_hidden[i][h_idx] = self.mem_hidden[i][h_idx] * 0.9 + current
                if self.mem_hidden[i][h_idx] >= self.threshold:
                    hidden_spikes[i][h_idx] = 1
                    self.mem_hidden[i][h_idx] = 0.0
                    self.last_spike_hidden[i][h_idx] = timestep
                    
                    for in_idx in range(self.d_model):
                        t_in = self.last_spike_in[i][in_idx]
                        if t_in >= 0:
                            delta = timestep - t_in
                            self.W_1[in_idx][h_idx] += 0.001 * math.exp(-delta / 5.0)

            for out_idx in range(self.d_model):
                current = 0.0
                for h_idx in range(self.d_ff):
                    if hidden_spikes[i][h_idx] > 0:
                        current += self.W_2[h_idx][out_idx]
                
                self.mem_out[i][out_idx] = self.mem_out[i][out_idx] * 0.9 + current
                if self.mem_out[i][out_idx] >= self.threshold:
                    out_spikes[i][out_idx] = 1
                    self.mem_out[i][out_idx] = 0.0
                    
                    for h_idx in range(self.d_ff):
                        t_h = self.last_spike_hidden[i][h_idx]
                        if t_h >= 0:
                            delta = timestep - t_h
                            self.W_2[h_idx][out_idx] += 0.001 * math.exp(-delta / 5.0)

        return out_spikes