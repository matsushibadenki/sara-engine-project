# src/sara_engine/models/spiking_transformer_stdp.py
# 日本語タイトル: スパイキング・トランスフォーマー（対照学習・完全収束版）
# ファイルの目的や内容: 32次元の極小空間で膠着した学習を打破するため、背景ノイズ相殺と対照的エラー駆動学習を導入し、正確な再生を実現する。

import math
import random
from typing import List, Optional

class LIFNeuron:
    def __init__(self, threshold: float = 1.0, decay: float = 0.5):
        self.v = 0.0
        self.threshold = threshold
        self.decay = decay
        self.refractory = 0

    def step(self, current: float) -> bool:
        if self.refractory > 0:
            self.refractory -= 1
            return False
        self.v = self.v * self.decay + current
        if self.v >= self.threshold:
            self.v = 0.0
            self.refractory = 1
            return True
        if self.v < 0: self.v = 0
        return False

class STDPAttention:
    def __init__(self, seq_len: int, d_model: int):
        self.seq_len = seq_len
        self.d_model = d_model
        self.attn_weights = [[0.0 for _ in range(seq_len)] for _ in range(seq_len)]
        for i in range(seq_len):
            for j in range(i + 1):
                # 順序情報を強く保持
                self.attn_weights[i][j] = math.exp(-(i - j) * 1.0)
        self.neurons = [[LIFNeuron(threshold=0.8, decay=0.4) for _ in range(d_model)] for _ in range(seq_len)]

    def reset(self):
        for row in self.neurons:
            for n in row: n.v = 0.0

    def step(self, x_spikes: List[List[bool]], t: int) -> List[List[bool]]:
        out = [[False] * self.d_model for _ in range(self.seq_len)]
        for i in range(self.seq_len):
            for d in range(self.d_model):
                curr = sum(self.attn_weights[i][j] for j in range(i + 1) if x_spikes[j][d])
                if self.neurons[i][d].step(curr / 1.1):
                    out[i][d] = True
        return out

class SpikingFFN:
    def __init__(self, seq_len: int, d_model: int, d_ff: int):
        self.seq_len, self.d_model, self.d_ff = seq_len, d_model, d_ff
        rng = random.Random(42)
        self.w1 = [[rng.uniform(0.0, 0.5) for _ in range(d_model)] for _ in range(d_ff)]
        self.w2 = [[rng.uniform(0.0, 0.5) for _ in range(d_ff)] for _ in range(d_model)]
        self.h_neurons = [[LIFNeuron(threshold=1.1, decay=0.4) for _ in range(d_ff)] for _ in range(seq_len)]
        self.o_neurons = [[LIFNeuron(threshold=1.1, decay=0.4) for _ in range(d_model)] for _ in range(seq_len)]
        
    def reset(self):
        for row in self.h_neurons:
            for n in row: n.v = 0.0
        for row in self.o_neurons:
            for n in row: n.v = 0.0

    def step(self, x_spikes: List[List[bool]], t: int) -> List[List[bool]]:
        out = [[False] * self.d_model for _ in range(self.seq_len)]
        for i in range(self.seq_len):
            active_d = [d for d in range(self.d_model) if x_spikes[i][d]]
            if not (cnt := len(active_d)): continue
            h_fired = [h for h in range(self.d_ff) if self.h_neurons[i][h].step(sum(self.w1[h][d] for d in active_d) / math.sqrt(cnt))]
            if not h_fired: continue
            for d in range(self.d_model):
                if self.o_neurons[i][d].step(sum(self.w2[d][h] for h in h_fired) / math.sqrt(len(h_fired))):
                    out[i][d] = True
        return out

class SpikingTransformer:
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, d_ff: int, num_layers: int):
        self.vocab_size, self.seq_len, self.d_model = vocab_size, seq_len, d_model
        rng = random.Random(42)
        # 32次元空間を最大限活用するための疎なシグネチャ (5ビット/32次元)
        self.token_signatures = [[0.0] * d_model for _ in range(vocab_size)]
        for v in range(vocab_size):
            for idx in rng.sample(range(d_model), 5):
                self.token_signatures[v][idx] = 1.0
        self.pos_signatures = [[0.0] * d_model for _ in range(seq_len)]
        for p in range(seq_len):
            for idx in rng.sample(range(d_model), 3):
                self.pos_signatures[p][idx] = 0.3

        self.layers = [SpikingTransformerBlock(seq_len, d_model, d_ff) for _ in range(num_layers)]
        self.decoder_w = [[rng.uniform(-0.1, 0.1) for _ in range(d_model)] for _ in range(vocab_size)]
        self._normalize_decoder()

    def _normalize_decoder(self, indices: Optional[List[int]] = None):
        targets = indices if indices else range(self.vocab_size)
        for v in targets:
            norm = math.sqrt(sum(w*w for w in self.decoder_w[v])) + 1e-9
            self.decoder_w[v] = [w / norm for w in self.decoder_w[v]]

    def __call__(self, tokens: List[int], target_tokens: Optional[List[int]] = None, simulation_steps: int = 60) -> List[int]:
        padded_tokens = (tokens + [0] * self.seq_len)[: self.seq_len]
        for layer in self.layers: layer.reset()
        acc_spikes = [[0.0] * self.d_model for _ in range(self.seq_len)]
        input_v = [[0.0] * self.d_model for _ in range(self.seq_len)]

        for t in range(simulation_steps):
            layer_in = []
            for i, tk in enumerate(padded_tokens):
                spks = []
                for d in range(self.d_model):
                    input_v[i][d] += (self.token_signatures[tk][d] + self.pos_signatures[i][d])
                    if input_v[i][d] >= 1.0:
                        input_v[i][d] -= 1.0
                        spks.append(True)
                    else: spks.append(False)
                layer_in.append(spks)
            for layer in self.layers:
                layer_in = layer.step(layer_in, t)
            for i in range(self.seq_len):
                for d in range(self.d_model):
                    if layer_in[i][d]: acc_spikes[i][d] += 1.0

        preds = []
        for i in range(self.seq_len):
            # 背景ノイズの相殺: 平均的な発火を差し引いて固有パターンを抽出
            avg_s = sum(acc_spikes[i]) / self.d_model
            diff_x = [max(0, s - avg_s) for s in acc_spikes[i]]
            norm_x = math.sqrt(sum(s*s for s in diff_x)) + 1e-9
            
            best_v, best_score = 0, -2.0
            if norm_x > 0.5:
                for v in range(self.vocab_size):
                    dot = sum(diff_x[d] * self.decoder_w[v][d] for d in range(self.d_model))
                    if (score := dot / norm_x) > best_score:
                        best_score, best_v = score, v
            preds.append(best_v)

        if target_tokens:
            padded_targets = (target_tokens + [0] * self.seq_len)[: self.seq_len]
            # 対照学習: 正解を引き寄せ、誤回答を強く押し出す
            lr_pos, lr_neg = 0.8, 0.4
            for i, (tar, pre) in enumerate(zip(padded_targets, preds)):
                if tar == 0: continue
                avg_s = sum(acc_spikes[i]) / self.d_model
                diff_x = [max(0, s - avg_s) for s in acc_spikes[i]]
                norm_x = math.sqrt(sum(s*s for s in diff_x)) + 1e-9
                if norm_x > 0.5:
                    vec = [s / norm_x for s in diff_x]
                    for d in range(self.d_model):
                        self.decoder_w[tar][d] += lr_pos * vec[d]
                        if pre != 0 and pre != tar:
                            self.decoder_w[pre][d] -= lr_neg * vec[d]
                    self._normalize_decoder([tar, pre] if (pre != 0 and pre != tar) else [tar])
        return preds

class SpikingTransformerBlock:
    def __init__(self, seq_len: int, d_model: int, d_ff: int):
        self.attention = STDPAttention(seq_len, d_model)
        self.ffn = SpikingFFN(seq_len, d_model, d_ff)

    def reset(self):
        self.attention.reset()
        self.ffn.reset()

    def step(self, x: List[List[bool]], t: int) -> List[List[bool]]:
        attn_out = self.attention.step(x, t)
        m1 = [[x[i][d] or attn_out[i][d] for d in range(len(x[i]))] for i in range(len(x))]
        ffn_out = self.ffn.step(m1, t)
        return [[m1[i][d] or ffn_out[i][d] for d in range(len(x[i]))] for i in range(len(x))]