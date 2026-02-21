# src/sara_engine/models/spiking_transformer_stdp.py
# 日本語タイトル: スパイキング・トランスフォーマー（球面パーセプトロン学習・完全収束版）
# ファイルの目的や内容: 32次元の小さな空間でも表現が混線しないよう、抑制性シナプスと厳格なエラー駆動学習（球面パーセプトロン）を導入。エポックごとに正解文字列へ確実に収束させる。

import math
import random
from typing import List, Optional

class LIFNeuron:
    """
    ソフトリセット型 Leaky Integrate-and-Fire ニューロン
    不応期を無くし、発火後に閾値分だけ電位を下げることで情報ロスを防ぐ。
    """
    def __init__(self, threshold: float = 1.0, decay: float = 0.7):
        self.v = 0.0
        self.threshold = threshold
        self.decay = decay

    def step(self, current: float) -> bool:
        self.v = self.v * self.decay + current
        if self.v >= self.threshold:
            self.v -= self.threshold  # ソフトリセット（残余電位を保持）
            return True
        # 抑制性入力で電位が下がりすぎるのを防ぐ下限
        if self.v < -1.0:
            self.v = -1.0
        return False

class STDPAttention:
    """
    自己回帰（因果律）アテンション
    """
    def __init__(self, seq_len: int, d_model: int):
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 過去のトークンに対して指数関数的に減衰する初期注目度（文脈の形成）
        self.attn_weights = [[0.0 for _ in range(seq_len)] for _ in range(seq_len)]
        for i in range(seq_len):
            for j in range(i + 1):
                self.attn_weights[i][j] = math.exp(-(i - j) * 0.5)
                
        self.neurons = [[LIFNeuron(threshold=1.0, decay=0.6) for _ in range(d_model)] for _ in range(seq_len)]
        self.last_fire_q = [-1] * seq_len
        self.last_fire_k = [-1] * seq_len

    def reset(self):
        for row in self.neurons:
            for n in row:
                n.v = 0.0
        self.last_fire_q = [-1] * self.seq_len
        self.last_fire_k = [-1] * self.seq_len

    def step(self, x_spikes: List[List[bool]], t: int) -> List[List[bool]]:
        out = [[False] * self.d_model for _ in range(self.seq_len)]
        
        for i in range(self.seq_len):
            if any(x_spikes[i]):
                self.last_fire_q[i] = t
                self.last_fire_k[i] = t

        for i in range(self.seq_len):
            active_js = [j for j in range(i + 1) if any(x_spikes[j])]
            if not active_js:
                continue
                
            for d in range(self.d_model):
                curr = sum(self.attn_weights[i][j] for j in active_js if x_spikes[j][d])
                curr /= math.sqrt(len(active_js)) + 1e-5
                
                if self.neurons[i][d].step(curr):
                    out[i][d] = True
        return out

    def apply_stdp(self, lr_ltp: float = 0.05, lr_ltd: float = 0.01) -> None:
        for i in range(self.seq_len):
            for j in range(i + 1):
                if self.last_fire_q[i] >= 0 and self.last_fire_k[j] >= 0:
                    delta_t = self.last_fire_q[i] - self.last_fire_k[j]
                    if 0 <= delta_t <= 3:
                        self.attn_weights[i][j] = min(2.0, self.attn_weights[i][j] + lr_ltp)
                    else:
                        self.attn_weights[i][j] = max(0.0, self.attn_weights[i][j] - lr_ltd)

class SpikingFFN:
    """
    抑制性シナプスを含むFFN
    """
    def __init__(self, seq_len: int, d_model: int, d_ff: int):
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_ff = d_ff
        
        # 興奮性と抑制性（マイナス）の重みを混在させ、パターン分離能力を高める
        self.w1 = [[random.uniform(-0.5, 1.0) for _ in range(d_model)] for _ in range(d_ff)]
        self.w2 = [[random.uniform(-0.5, 1.0) for _ in range(d_ff)] for _ in range(d_model)]
        
        self.h_neurons = [[LIFNeuron(threshold=1.0, decay=0.6) for _ in range(d_ff)] for _ in range(seq_len)]
        self.o_neurons = [[LIFNeuron(threshold=1.0, decay=0.6) for _ in range(d_model)] for _ in range(seq_len)]
        
    def reset(self):
        for row in self.h_neurons:
            for n in row: n.v = 0.0
        for row in self.o_neurons:
            for n in row: n.v = 0.0

    def step(self, x_spikes: List[List[bool]], t: int) -> List[List[bool]]:
        out = [[False] * self.d_model for _ in range(self.seq_len)]
        for i in range(self.seq_len):
            active_d = sum(1 for d in range(self.d_model) if x_spikes[i][d])
            if active_d == 0:
                continue
                
            for h in range(self.d_ff):
                curr = sum(self.w1[h][d] for d in range(self.d_model) if x_spikes[i][d])
                curr /= math.sqrt(active_d)
                
                if self.h_neurons[i][h].step(curr):
                    for d in range(self.d_model):
                        # FFNからの出力は常に興奮性として次の層へ
                        if self.w2[d][h] > 0.0:
                            if self.o_neurons[i][d].step(self.w2[d][h]):
                                out[i][d] = True
        return out

class SpikingTransformerBlock:
    def __init__(self, seq_len: int, d_model: int, d_ff: int):
        self.attention = STDPAttention(seq_len, d_model)
        self.ffn = SpikingFFN(seq_len, d_model, d_ff)

    def reset(self):
        self.attention.reset()
        self.ffn.reset()

    def step(self, x: List[List[bool]], t: int) -> List[List[bool]]:
        attn_out = self.attention.step(x, t)
        merged1 = [[x[i][d] or attn_out[i][d] for d in range(len(x[i]))] for i in range(len(x))]
        
        ffn_out = self.ffn.step(merged1, t)
        merged2 = [[merged1[i][d] or ffn_out[i][d] for d in range(len(x[i]))] for i in range(len(x))]
        
        return merged2

class SpikingTransformer:
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, d_ff: int, num_layers: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 密な確率シグネチャ（32次元でも十分な分離を確保する）
        random.seed(42)
        self.token_signatures = [[random.uniform(0.1, 0.9) for _ in range(d_model)] for _ in range(vocab_size)]
        self.pos_signatures = [[random.uniform(0.1, 0.9) for _ in range(d_model)] for _ in range(seq_len)]
        random.seed()

        self.layers = [SpikingTransformerBlock(seq_len, d_model, d_ff) for _ in range(num_layers)]
        
        # デコーダー重み（ランダムな単位ベクトルで初期化）
        self.decoder_w = [[random.uniform(-1.0, 1.0) for _ in range(d_model)] for _ in range(vocab_size)]
        for v in range(vocab_size):
            norm = math.sqrt(sum(w*w for w in self.decoder_w[v])) + 1e-9
            self.decoder_w[v] = [w / norm for w in self.decoder_w[v]]

    def __call__(self, tokens: List[int], target_tokens: Optional[List[int]] = None, simulation_steps: int = 20) -> List[int]:
        padded_tokens = (tokens + [0] * self.seq_len)[: self.seq_len]
        
        for layer in self.layers:
            layer.reset()
            
        acc_spikes = [[0.0] * self.d_model for _ in range(self.seq_len)]

        for t in range(simulation_steps):
            layer_in = []
            for i, tk in enumerate(padded_tokens):
                tk_sig = self.token_signatures[tk]
                pos_sig = self.pos_signatures[i]
                
                spikes = []
                for d in range(self.d_model):
                    # トークンと位置をブレンドして固有の確率的入力を生成
                    prob = (tk_sig[d] + pos_sig[d]) * 0.5
                    spikes.append(random.random() < prob)
                layer_in.append(spikes)

            for layer in self.layers:
                layer_in = layer.step(layer_in, t)

            for i in range(self.seq_len):
                for d in range(self.d_model):
                    if layer_in[i][d]:
                        acc_spikes[i][d] += 1.0

        # コサイン類似度による推論
        preds = []
        for i in range(self.seq_len):
            norm_x = math.sqrt(sum(s*s for s in acc_spikes[i])) + 1e-9
            
            best_score = -2.0
            best_v = 0
            
            for v in range(self.vocab_size):
                dot = sum(acc_spikes[i][d] * self.decoder_w[v][d] for d in range(self.d_model))
                score = dot / norm_x  # decoder_w は既に正規化済み
                
                if score > best_score:
                    best_score = score
                    best_v = v
            
            if norm_x < 1.0:
                preds.append(0)
            else:
                preds.append(best_v)

        # 球面パーセプトロン学習（間違えた場合のみ強力に補正）
        if target_tokens:
            padded_targets = (target_tokens + [0] * self.seq_len)[: self.seq_len]
            lr = 0.5  # 高い学習率で素早く収束させる
            
            for i in range(self.seq_len):
                tar = padded_targets[i]
                if tar == 0: continue
                pre = preds[i]

                # 予測が外れた場合のみ学習（エラー駆動）
                if pre != tar:
                    norm_x = math.sqrt(sum(s*s for s in acc_spikes[i])) + 1e-9
                    
                    for d in range(self.d_model):
                        x_val = acc_spikes[i][d] / norm_x
                        
                        # 正解ベクトルを現在のスパイクパターンに引き寄せる
                        self.decoder_w[tar][d] += lr * x_val
                        
                        # 間違えたベクトルを現在のスパイクパターンから遠ざける
                        if pre != 0:
                            self.decoder_w[pre][d] -= lr * x_val
                            
                    # 重みベクトルの再正規化（球面を維持）
                    for v in [tar, pre]:
                        if v == 0: continue
                        norm_w = math.sqrt(sum(w*w for w in self.decoder_w[v])) + 1e-9
                        for d in range(self.d_model):
                            self.decoder_w[v][d] /= norm_w

            # STDPの更新
            for layer in self.layers:
                layer.attention.apply_stdp()

        return preds