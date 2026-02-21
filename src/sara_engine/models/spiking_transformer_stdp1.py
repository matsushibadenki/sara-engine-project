# ディレクトリパス: src/sara_engine/models/spiking_transformer_stdp.py
# ファイルタイトル: スパイキング・トランスフォーマー（連合学習強化版）
# ファイルの目的: クラス定義順序の修正およびモード崩壊を抑制する連合学習の実装

import math
import random
from typing import List

class HomeostaticLIFNeuron:
    """
    ホメオスタシス機能付きLIFニューロン。
    活動トレースに基づき閾値を動的に調整し、過剰発火と沈黙を防止する。
    """
    def __init__(self, base_threshold: float = 0.3, decay: float = 0.9, rest: float = 0.0):
        self.base_threshold = base_threshold
        self.threshold = base_threshold
        self.decay = decay
        self.rest = rest
        self.v = rest
        self.activity_trace = 0.0
        self.threshold_max = 1.2
        self.last_spike_time = -10 # 不応期判定用

    def step(self, current: float, current_time: int) -> bool:
        # 不応期（発火直後の連続発火を抑制）
        if current_time - self.last_spike_time < 3:
            self.v = self.rest
            return False

        self.v = (self.v - self.rest) * self.decay + self.rest + current
        self.activity_trace *= 0.85
        self.threshold = min(self.threshold_max, self.base_threshold + self.activity_trace * 0.4)

        if self.v >= self.threshold:
            self.v = self.rest
            self.activity_trace += 1.0
            self.last_spike_time = current_time
            return True
        return False

class STDPAttention:
    """
    スパイクタイミング依存可塑性（STDP）を用いた注意機構。
    """
    def __init__(self, seq_len: int, d_model: int, lr_ltp: float = 0.02, lr_ltd: float = 0.01, window: int = 5):
        self.seq_len = seq_len
        self.d_model = d_model
        self.lr_ltp = lr_ltp
        self.lr_ltd = lr_ltd
        self.window = window
        self.attn_weights = [[random.uniform(0.2, 0.5) for _ in range(seq_len)] for _ in range(seq_len)]
        self.q_neurons = [[HomeostaticLIFNeuron() for _ in range(d_model)] for _ in range(seq_len)]
        self.k_neurons = [[HomeostaticLIFNeuron() for _ in range(d_model)] for _ in range(seq_len)]
        self.v_neurons = [[HomeostaticLIFNeuron() for _ in range(d_model)] for _ in range(seq_len)]
        self.out_neurons = [[HomeostaticLIFNeuron() for _ in range(d_model)] for _ in range(seq_len)]

    def apply_stdp(self, i: int, j: int):
        t_q = max(n.last_spike_time for n in self.q_neurons[i])
        t_k = max(n.last_spike_time for n in self.k_neurons[j])
        if t_q <= 0 or t_k <= 0: return
        delta_t = t_q - t_k
        if 0 < delta_t <= self.window:
            self.attn_weights[i][j] = min(1.0, self.attn_weights[i][j] + self.lr_ltp * math.exp(-delta_t / self.window))
        elif -self.window <= delta_t <= 0:
            self.attn_weights[i][j] = max(0.0, self.attn_weights[i][j] - self.lr_ltd * math.exp(delta_t / self.window))

    def forward_step(self, x: List[List[float]], t: int) -> List[List[bool]]:
        q_s = [[self.q_neurons[i][d].step(x[i][d], t) for d in range(self.d_model)] for i in range(self.seq_len)]
        k_s = [[self.k_neurons[i][d].step(x[i][d], t) for d in range(self.d_model)] for i in range(self.seq_len)]
        v_s = [[self.v_neurons[i][d].step(x[i][d], t) for d in range(self.d_model)] for i in range(self.seq_len)]
        out = [[False] * self.d_model for _ in range(self.seq_len)]
        for i in range(self.seq_len):
            for j in range(self.seq_len):
                if any(q_s[i]) or any(k_s[j]): self.apply_stdp(i, j)
                w = self.attn_weights[i][j]
                for d in range(self.d_model):
                    if v_s[j][d]: out[i][d] = self.out_neurons[i][d].step(w, t)
        return out

class FFNLayer:
    """
    行列演算を使用しない、スパイクベースのFeed Forward Network層。
    """
    def __init__(self, seq_len: int, d_model: int, d_ff: int):
        self.seq_len, self.d_model, self.d_ff = seq_len, d_model, d_ff
        self.w1 = [[random.uniform(0.1, 0.4) for _ in range(d_model)] for _ in range(d_ff)]
        self.w2 = [[random.uniform(0.1, 0.4) for _ in range(d_ff)] for _ in range(d_model)]
        self.h_neurons = [[HomeostaticLIFNeuron() for _ in range(d_ff)] for _ in range(seq_len)]
        self.o_neurons = [[HomeostaticLIFNeuron() for _ in range(d_model)] for _ in range(seq_len)]

    def forward_step(self, in_s: List[List[bool]], t: int) -> List[List[bool]]:
        out = [[False] * self.d_model for _ in range(self.seq_len)]
        for i in range(self.seq_len):
            for h in range(self.d_ff):
                current = sum(self.w1[h][d] for d in range(self.d_model) if in_s[i][d])
                if self.h_neurons[i][h].step(current, t):
                    for d in range(self.d_model):
                        if self.o_neurons[i][d].step(self.w2[d][h], t): out[i][d] = True
        return out

class SpikingTransformerBlock:
    """
    AttentionとFFNを組み合わせたトランスフォーマーブロック。
    """
    def __init__(self, seq_len: int, d_model: int, d_ff: int):
        self.attention = STDPAttention(seq_len, d_model)
        self.ffn = FFNLayer(seq_len, d_model, d_ff)

    def forward_step(self, x: List[List[float]], t: int) -> List[List[bool]]:
        return self.ffn.forward_step(self.attention.forward_step(x, t), t)

class SpikingTransformer:
    """
    Error-Driven Hebbian学習を用いたスパイキング・トランスフォーマー。
    """
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, d_ff: int, num_layers: int):
        self.vocab_size, self.seq_len, self.d_model = vocab_size, seq_len, d_model
        self.embedding = [[random.uniform(0.3, 0.6) for _ in range(d_model)] for _ in range(vocab_size)]
        self.decoder_w = [[random.uniform(0.05, 0.15) for _ in range(d_model)] for _ in range(vocab_size)]
        # クラス定義後にレイヤーを初期化
        self.layers = [SpikingTransformerBlock(seq_len, d_model, d_ff) for _ in range(num_layers)]

    def __call__(self, tokens: List[int], target_tokens: List[int] = None, simulation_steps: int = 30) -> List[int]:
        padded_tokens = (tokens + [0] * self.seq_len)[:self.seq_len]
        counts = [[0.0] * self.vocab_size for _ in range(self.seq_len)]
        acc_spikes = [[0] * self.d_model for _ in range(self.seq_len)]

        for t in range(simulation_steps):
            cur_x = [[1.0 if random.random() < self.embedding[tk][d] else 0.0 for d in range(self.d_model)] for tk in padded_tokens]
            layer_in = cur_x
            for layer in self.layers:
                out_s = layer.forward_step(layer_in, t)
                layer_in = [[1.5 if s else 0.0 for s in row] for row in out_s]
            for i in range(self.seq_len):
                for d in range(self.d_model):
                    if layer_in[i][d] > 0:
                        acc_spikes[i][d] += 1
                        for v in range(1, self.vocab_size):
                            counts[i][v] += self.decoder_w[v][d]

        preds = []
        for i in range(self.seq_len):
            if sum(counts[i]) > 0:
                preds.append(max(range(1, self.vocab_size), key=lambda v: counts[i][v]))
            else:
                preds.append(max(range(1, self.vocab_size), key=lambda v: sum(self.decoder_w[v])))

        if target_tokens:
            padded_targets = (target_tokens + [0] * self.seq_len)[:self.seq_len]
            lr = 0.1
            for i in range(self.seq_len):
                tar, pre, src = padded_targets[i], preds[i], padded_tokens[i]
                if tar == 0: continue
                for d in range(self.d_model):
                    if acc_spikes[i][d] > 0:
                        # 入力 src との相関に基づき tar を強化し、モード崩壊を抑制
                        assoc = self.embedding[src][d]
                        self.decoder_w[tar][d] = min(3.0, self.decoder_w[tar][d] + lr * assoc)
                        if pre != tar: 
                            self.decoder_w[pre][d] = max(0.0, self.decoder_w[pre][d] - lr * 0.5)
                        
                        # 重みの正規化により特定の文字の独走を防止
                        w_sum = sum(self.decoder_w[v][d] for v in range(self.vocab_size))
                        if w_sum > 5.0:
                            for v in range(self.vocab_size): self.decoder_w[v][d] *= (5.0 / w_sum)
        return preds