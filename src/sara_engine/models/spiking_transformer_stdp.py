# src/sara_engine/models/spiking_transformer_stdp.py
# スパイキング・トランスフォーマー（Error-Driven Hebbian & STDP）修正版
#
# 修正した問題:
#   [Bug1] HomeostaticLIFNeuron: activity_trace が蓄積し閾値が無限上昇 → エポック2以降完全沈黙
#   [Bug2] STDPAttention.forward_step: t_q == t_k 常に成立 → apply_stdp が delta_t=0 で LTD のみ適用
#          → attn_weights が全て 0 に収束 → 信号が通らない
#   [Bug3] FFNLayer: w1/w2 の形状と in_s の型が噛み合っておらず入力電流がほぼ0
#   [Bug4] SpikingTransformer.__call__: acc_spikes が0の場合 counts が全ゼロ
#          → max(range(vocab)) が常に index=0 (=\x00) → bytes_to_text でフィルタされ空文字列
#   [Bug5] embedding の更新ロジックが逆方向に働くケースがある

import math
import random
from typing import List, Optional


class HomeostaticLIFNeuron:
    """
    Leaky Integrate-and-Fire ニューロン（ホメオスタシス付き）
    修正: activity_trace の減衰を強化し、閾値上昇の上限を設ける
    """
    def __init__(
        self,
        base_threshold: float = 0.3,
        decay: float = 0.9,
        rest: float = 0.0,
        homeostasis_strength: float = 0.01,
        threshold_max: float = 1.0,
    ):
        self.base_threshold = base_threshold
        self.threshold = base_threshold
        self.threshold_max = threshold_max
        self.homeostasis_strength = homeostasis_strength
        self.decay = decay
        self.rest = rest
        self.v = rest
        # [Fix1] activity_trace の減衰率を 0.95 → 0.80 に強化してリセットを速くする
        self.activity_trace = 0.0
        self.trace_decay = 0.80

    def step(self, current: float, current_time: int) -> bool:
        self.v = (self.v - self.rest) * self.decay + self.rest + current
        self.activity_trace *= self.trace_decay

        # [Fix1] 閾値上昇を base+activity_trace*strength に制限し、上限を設ける
        self.threshold = min(
            self.threshold_max,
            self.base_threshold + self.activity_trace * self.homeostasis_strength
        )

        if self.v >= self.threshold:
            self.v = self.rest
            self.activity_trace += 1.0
            return True
        return False


class STDPAttention:
    """
    STDP ベースの Attention
    修正: delta_t の計算に発火タイムスタンプを使用する（同時刻比較バグを修正）
    """
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        lr_ltp: float = 0.02,
        lr_ltd: float = 0.008,
        window: int = 5,
    ):
        self.seq_len = seq_len
        self.d_model = d_model
        self.lr_ltp = lr_ltp
        self.lr_ltd = lr_ltd
        self.window = window

        # attn_weights を初期値 0.3 付近で均一に持つ
        self.attn_weights = [
            [0.3 + random.uniform(-0.05, 0.05) for _ in range(seq_len)]
            for _ in range(seq_len)
        ]

        self.q_neurons = [
            [HomeostaticLIFNeuron() for _ in range(d_model)] for _ in range(seq_len)
        ]
        self.k_neurons = [
            [HomeostaticLIFNeuron() for _ in range(d_model)] for _ in range(seq_len)
        ]
        self.v_neurons = [
            [HomeostaticLIFNeuron() for _ in range(d_model)] for _ in range(seq_len)
        ]
        self.out_neurons = [
            [HomeostaticLIFNeuron() for _ in range(d_model)] for _ in range(seq_len)
        ]

        # [Fix2] 各位置の最終発火時刻を記録してSTDPのδtを正確に計算する
        self.last_q_fire: List[int] = [-1] * seq_len
        self.last_k_fire: List[int] = [-1] * seq_len

    def apply_stdp(self, i: int, j: int, t_q: int, t_k: int) -> None:
        if t_q < 0 or t_k < 0:
            return
        delta_t = t_q - t_k
        if 0 < delta_t <= self.window:
            dw = self.lr_ltp * math.exp(-delta_t / self.window)
            self.attn_weights[i][j] = min(1.0, self.attn_weights[i][j] + dw)
        elif -self.window <= delta_t < 0:
            dw = self.lr_ltd * math.exp(delta_t / self.window)
            self.attn_weights[i][j] = max(0.0, self.attn_weights[i][j] - dw)
        # delta_t == 0 の場合は更新しない（因果関係が不明）

    def forward_step(self, x: List[List[float]], current_time: int) -> List[List[bool]]:
        # Q / K / V ニューロンを更新し発火パターンを得る
        q_s = [
            [self.q_neurons[i][d].step(x[i][d], current_time) for d in range(self.d_model)]
            for i in range(self.seq_len)
        ]
        k_s = [
            [self.k_neurons[i][d].step(x[i][d], current_time) for d in range(self.d_model)]
            for i in range(self.seq_len)
        ]
        v_s = [
            [self.v_neurons[i][d].step(x[i][d], current_time) for d in range(self.d_model)]
            for i in range(self.seq_len)
        ]

        # [Fix2] 最終発火時刻を更新
        for i in range(self.seq_len):
            if any(q_s[i]):
                self.last_q_fire[i] = current_time
            if any(k_s[i]):
                self.last_k_fire[i] = current_time

        out = [[False] * self.d_model for _ in range(self.seq_len)]
        for i in range(self.seq_len):
            for j in range(self.seq_len):
                # [Fix2] 記録済みタイムスタンプを使って STDP を適用
                self.apply_stdp(i, j, self.last_q_fire[i], self.last_k_fire[j])
                w = self.attn_weights[i][j]
                for d in range(self.d_model):
                    if v_s[j][d]:
                        out[i][d] = self.out_neurons[i][d].step(w, current_time)
        return out


class FFNLayer:
    """
    FFN（位置ごとフィードフォワード）
    修正: w1/w2 の形状と in_s の型の不整合を修正し、入力電流の計算を正しくする
    """
    def __init__(self, seq_len: int, d_model: int, d_ff: int):
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_ff = d_ff

        # [Fix3] w1: shape = [d_ff][d_model]  w2: shape = [d_model][d_ff]
        # 初期重みを 0.2〜0.6 の範囲で高めに設定して初期電流を確保する
        self.w1 = [
            [random.uniform(0.2, 0.6) for _ in range(d_model)] for _ in range(d_ff)
        ]
        self.w2 = [
            [random.uniform(0.2, 0.6) for _ in range(d_ff)] for _ in range(d_model)
        ]

        self.h_neurons = [
            [HomeostaticLIFNeuron() for _ in range(d_ff)] for _ in range(seq_len)
        ]
        self.o_neurons = [
            [HomeostaticLIFNeuron() for _ in range(d_model)] for _ in range(seq_len)
        ]

    def forward_step(self, in_s: List[List[bool]], t: int) -> List[List[bool]]:
        out = [[False] * self.d_model for _ in range(self.seq_len)]
        for i in range(self.seq_len):
            # [Fix3] in_s[i][d] は bool → 発火した次元の重みを加算する（正しい形状でアクセス）
            h_inputs = [
                sum(self.w1[h][d] for d in range(self.d_model) if in_s[i][d])
                for h in range(self.d_ff)
            ]
            h_s = [
                self.h_neurons[i][h].step(h_inputs[h], t) for h in range(self.d_ff)
            ]
            o_inputs = [
                sum(self.w2[d][h] for h in range(self.d_ff) if h_s[h])
                for d in range(self.d_model)
            ]
            for d in range(self.d_model):
                out[i][d] = self.o_neurons[i][d].step(o_inputs[d], t)
        return out


class SpikingTransformerBlock:
    def __init__(self, seq_len: int, d_model: int, d_ff: int):
        self.attention = STDPAttention(seq_len, d_model)
        self.ffn = FFNLayer(seq_len, d_model, d_ff)

    def forward_step(self, x: List[List[float]], t: int) -> List[List[bool]]:
        attn_out = self.attention.forward_step(x, t)
        # Residual: 入力 x の確率的発火を attn_out に重ねてFFNへ渡す
        merged = [
            [attn_out[i][d] or (x[i][d] > 0.5) for d in range(len(x[i]))]
            for i in range(len(x))
        ]
        return self.ffn.forward_step(merged, t)


class SpikingTransformer:
    """
    メインモデル
    修正:
      [Fix4] acc_spikes が全0の場合でも counts を 0 のままにせず、
             decoder_w の期待値で補完する（フォールバック）
      [Fix5] embedding の更新方向を正解トークンに向けて一貫させる
    """
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        d_ff: int,
        num_layers: int,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model

        # embedding: 発火確率マップ [vocab_size][d_model]
        # 0.4〜0.7 の範囲で初期化し、確率的発火を安定させる
        self.embedding = [
            [random.uniform(0.4, 0.7) for _ in range(d_model)]
            for _ in range(vocab_size)
        ]

        # decoder_w: 各語彙への重み [vocab_size][d_model]
        # 初期値を 0.1〜0.3 に抑えて特定トークンへの偏りを防ぐ
        self.decoder_w = [
            [random.uniform(0.1, 0.3) for _ in range(d_model)]
            for _ in range(vocab_size)
        ]

        self.layers = [
            SpikingTransformerBlock(seq_len, d_model, d_ff)
            for _ in range(num_layers)
        ]

    def __call__(
        self,
        tokens: List[int],
        target_tokens: Optional[List[int]] = None,
        simulation_steps: int = 30,
    ) -> List[int]:
        # パディング
        padded_tokens = (tokens + [0] * self.seq_len)[: self.seq_len]

        counts = [[0.0] * self.vocab_size for _ in range(self.seq_len)]
        acc_spikes = [[0] * self.d_model for _ in range(self.seq_len)]

        for t in range(simulation_steps):
            # 確率的スパイク入力を生成
            cur_x = [
                [
                    1.0 if random.random() < self.embedding[tk][d] else 0.0
                    for d in range(self.d_model)
                ]
                for tk in padded_tokens
            ]

            layer_in: List[List[float]] = cur_x
            for layer in self.layers:
                out_s = layer.forward_step(layer_in, t)
                # bool → float に変換して次のレイヤーへ渡す
                layer_in = [[1.0 if s else 0.0 for s in row] for row in out_s]

            for i in range(self.seq_len):
                for d in range(self.d_model):
                    if layer_in[i][d] > 0.5:
                        acc_spikes[i][d] += 1
                        for v in range(self.vocab_size):
                            counts[i][v] += self.decoder_w[v][d]

        # [Fix4] acc_spikes が全0の位置は decoder_w の列和でフォールバック
        for i in range(self.seq_len):
            total_spikes = sum(acc_spikes[i])
            if total_spikes == 0:
                for v in range(self.vocab_size):
                    counts[i][v] = sum(self.decoder_w[v])

        preds = [
            max(range(self.vocab_size), key=lambda v: counts[i][v])
            for i in range(self.seq_len)
        ]

        # 学習フェーズ
        if target_tokens:
            padded_targets = (target_tokens + [0] * self.seq_len)[: self.seq_len]
            lr = 0.05

            for i in range(self.seq_len):
                tar = padded_targets[i]
                pre = preds[i]
                src = padded_tokens[i]

                if tar == 0:
                    continue

                for d in range(self.d_model):
                    spike_count = acc_spikes[i][d]
                    if spike_count == 0:
                        continue

                    # LTP: 正解トークンの decoder_w を強化
                    self.decoder_w[tar][d] = min(
                        2.0, self.decoder_w[tar][d] + lr
                    )
                    # LTD: 誤予測トークンを軽く抑制
                    if pre != tar:
                        self.decoder_w[pre][d] = max(
                            0.0, self.decoder_w[pre][d] - lr * 0.1
                        )

                    # [Fix5] embedding 更新: 正解への一致度に応じて発火確率を調整
                    # 正解への正の学習信号のみ使用（方向を固定）
                    target_w = self.decoder_w[tar][d]
                    if target_w > 0.5:
                        # この次元が正解に有効 → 入力確率を上げる
                        self.embedding[src][d] = min(
                            0.9, self.embedding[src][d] + lr * 0.05
                        )
                    else:
                        # まだ弱い → 変化させない（不安定になるため）
                        pass

        return preds
