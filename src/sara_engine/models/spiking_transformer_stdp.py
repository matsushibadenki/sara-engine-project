# src/sara_engine/models/spiking_transformer_stdp.py
# 日本語タイトル: スパイキング・トランスフォーマー（位置×トークン2Dデコーダ版）
#
# ===== 根本原因の確定診断 =====
#
# 症状: Epoch1で英語100%・日本語92%を達成後、Epoch5から劣化しEpoch15で崩壊。
#       これは3回連続で観察された再現性のある崩壊パターン。
#
# 根本原因: 「位置依存性の欠如」
#
#   acc_spikes[i] は「位置i」での発火パターンであり、
#   pos_signatures[i] が全サンプル共通のベースとして含まれる。
#   しかし旧decoder_w は decoder_w[token_id] という1D構造のため、
#   異なるサンプルで異なるトークンが位置iに来るたびに
#   同じ「位置iベース発火パターン」が複数のトークンに学習される。
#
#   例: 位置0 で 'H'(Hello),'R'(Rust),'E'(Energy),'こ'(こんにちは) を学習
#       → decoder_w[H], decoder_w[R], decoder_w[E], decoder_w[こ] が
#         全て「位置0の発火パターン」に引き寄せられて互いに近づく
#       → コサイン比較で識別不能になる → 崩壊
#
# 解決: decoder_w を [pos][token][d] の3次元に拡張
#   → 位置iの発火パターンは decoder_w[i][token] のみを更新
#   → 異なる位置の学習が互いに干渉しない
#   → メモリ: vocab×seq_len×d_model = 512×32×32 = 2MB 以下
#
# ===== 変更履歴 =====
# 修正H: decoder_w を [vocab][d] → [seq_len][vocab][d] の3次元に拡張
# 修正I: 初期値は全位置で token_signatures から設定（位置非依存スタート）
#         学習を通じて位置依存に育っていく
# 修正J: デコード時に decoder_w[i][v] を参照（位置iのベクトルを使用）
# 修正K: 学習時に decoder_w[i][tar] のみを更新（位置をまたいだ更新を廃止）
# 修正L: EMAの引き戻し対象も位置別に管理

import math
import random
from typing import List, Optional, Tuple, Union


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
            self.refractory = 2
            return True
        if self.v < 0:
            self.v = 0.0
        return False

    def reset(self) -> None:
        self.v = 0.0
        self.refractory = 0


class STDPAttention:
    def __init__(self, seq_len: int, d_model: int):
        self.seq_len = seq_len
        self.d_model = d_model
        self.attn_weights = [[0.0] * seq_len for _ in range(seq_len)]
        for i in range(seq_len):
            for j in range(i + 1):
                self.attn_weights[i][j] = math.exp(-(i - j) * 1.5)
        self.neurons = [
            [LIFNeuron(threshold=0.8, decay=0.5) for _ in range(d_model)]
            for _ in range(seq_len)
        ]

    def reset(self) -> None:
        for row in self.neurons:
            for n in row:
                n.reset()

    def step(self, x_spikes: List[List[bool]], t: int) -> List[List[bool]]:
        out = [[False] * self.d_model for _ in range(self.seq_len)]
        for i in range(self.seq_len):
            for d in range(self.d_model):
                curr = sum(
                    self.attn_weights[i][j]
                    for j in range(i + 1)
                    if x_spikes[j][d]
                )
                if self.neurons[i][d].step(curr):
                    out[i][d] = True
        return out


class SpikingFFN:
    def __init__(self, seq_len: int, d_model: int, d_ff: int):
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_ff = d_ff
        rng = random.Random(42)
        self.w1 = [
            [rng.uniform(0.0, 0.4) for _ in range(d_model)] for _ in range(d_ff)
        ]
        self.w2 = [
            [rng.uniform(0.0, 0.4) for _ in range(d_ff)] for _ in range(d_model)
        ]
        self.h_neurons = [
            [LIFNeuron(threshold=1.0, decay=0.5) for _ in range(d_ff)]
            for _ in range(seq_len)
        ]
        self.o_neurons = [
            [LIFNeuron(threshold=1.0, decay=0.5) for _ in range(d_model)]
            for _ in range(seq_len)
        ]

    def reset(self) -> None:
        for row in self.h_neurons:
            for n in row:
                n.reset()
        for row in self.o_neurons:
            for n in row:
                n.reset()

    def step(self, x_spikes: List[List[bool]], t: int) -> List[List[bool]]:
        out = [[False] * self.d_model for _ in range(self.seq_len)]
        for i in range(self.seq_len):
            active_d = [d for d in range(self.d_model) if x_spikes[i][d]]
            if not active_d:
                continue
            h_fired = []
            for h in range(self.d_ff):
                raw_sum = sum(self.w1[h][d] for d in active_d)
                if self.h_neurons[i][h].step(raw_sum * 2.5):
                    h_fired.append(h)
            if not h_fired:
                continue
            for d in range(self.d_model):
                raw_sum = sum(self.w2[d][h] for h in h_fired)
                if self.o_neurons[i][d].step(raw_sum * 2.5):
                    out[i][d] = True
        return out


class SpikingTransformerBlock:
    def __init__(self, seq_len: int, d_model: int, d_ff: int):
        self.attention = STDPAttention(seq_len, d_model)
        self.ffn = SpikingFFN(seq_len, d_model, d_ff)

    def reset(self) -> None:
        self.attention.reset()
        self.ffn.reset()

    def step(self, x: List[List[bool]], t: int) -> List[List[bool]]:
        attn_out = self.attention.step(x, t)
        seq = len(x)
        dim = len(x[0])
        m1 = [[x[i][d] or attn_out[i][d] for d in range(dim)] for i in range(seq)]
        ffn_out = self.ffn.step(m1, t)
        return [[m1[i][d] or ffn_out[i][d] for d in range(dim)] for i in range(seq)]


class SpikingTransformer:
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        d_ff: int,
        num_layers: int,
        ema_alpha: float = 0.05,
    ):
        """
        Parameters
        ----------
        vocab_size  : 通常トークン数（PAD は内部で +1 して管理）
        seq_len     : 最大シーケンス長
        d_model     : 隠れ次元数
        d_ff        : FFN 中間層次元数
        num_layers  : Transformer ブロック数
        ema_alpha   : EMAで初期署名方向に引き戻す強さ
                      0=引き戻しなし / 0.05 推奨（長期学習でも初期精度を維持）
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self._ema_alpha = ema_alpha

        self._total_vocab = vocab_size + 1
        self._pad_id: int = vocab_size

        sig_bits = min(6, d_model // 5)

        self.token_signatures = [[0.0] * d_model for _ in range(self._total_vocab)]
        for v in range(vocab_size):
            local_rng = random.Random(v + 1000)
            for idx in local_rng.sample(range(d_model), sig_bits):
                self.token_signatures[v][idx] = 1.0

        self.pos_signatures = [[0.0] * d_model for _ in range(seq_len)]
        for p in range(seq_len):
            local_rng = random.Random(p + 2000)
            for idx in local_rng.sample(range(d_model), 2):
                self.pos_signatures[p][idx] = 0.5

        self.layers = [
            SpikingTransformerBlock(seq_len, d_model, d_ff)
            for _ in range(num_layers)
        ]

        # 修正H: decoder_w を [seq_len][total_vocab][d_model] の3次元に拡張
        # 各位置の発火パターンはその位置専用のデコーダベクトルで解釈する
        # 修正I: 初期値は全位置で token_signatures から設定（位置非依存スタート）
        init_row = [[0.0] * d_model for _ in range(self._total_vocab)]
        for v in range(vocab_size):
            sig = self.token_signatures[v]
            norm = math.sqrt(sum(s * s for s in sig)) + 1e-9
            init_row[v] = [s / norm for s in sig]

        # decoder_w[pos][token_id][d]
        self.decoder_w = [
            [list(init_row[v]) for v in range(self._total_vocab)]
            for _ in range(seq_len)
        ]

        # 修正L: EMA 用スナップショットも位置別に保持
        self._init_decoder_w = [
            [list(init_row[v]) for v in range(self._total_vocab)]
            for _ in range(seq_len)
        ]

        self._lr_pos = 0.08
        # dirty フラグ: (pos, token_id) のペアで管理
        self._dirty: set = set()

    def _normalize_decoder(self, targets: Optional[List[Tuple[int, int]]] = None) -> None:
        """
        targets: [(pos, token_id), ...] のリスト。
                 None なら全位置・全トークンを正規化。
        """
        if targets is None:
            all_targets = [
                (p, v)
                for p in range(self.seq_len)
                for v in range(self._total_vocab)
            ]
        else:
            all_targets = targets

        for p, v in all_targets:
            row = self.decoder_w[p][v]
            norm = math.sqrt(sum(w * w for w in row)) + 1e-9
            self.decoder_w[p][v] = [w / norm for w in row]

    def flush_normalize(self) -> None:
        """
        エポック終了時に呼ぶ一括正規化 + EMA安定化。
        demo スクリプトのエポックループ末尾で呼ぶこと:

            for epoch in range(epochs):
                for tokens, targets in dataset:
                    model(tokens, targets)
                model.flush_normalize()
        """
        if not self._dirty:
            return

        dirty_list = list(self._dirty)
        self._normalize_decoder(dirty_list)

        if self._ema_alpha > 0:
            alpha = self._ema_alpha
            for p, v in dirty_list:
                self.decoder_w[p][v] = [
                    (1.0 - alpha) * self.decoder_w[p][v][d]
                    + alpha * self._init_decoder_w[p][v][d]
                    for d in range(self.d_model)
                ]
            self._normalize_decoder(dirty_list)

        self._dirty.clear()

    def __call__(
        self,
        tokens: List[int],
        target_tokens: Optional[List[int]] = None,
        simulation_steps: int = 50,
        return_input_len: bool = False,
    ) -> Union[List[int], Tuple[List[int], int]]:
        """
        Parameters
        ----------
        tokens           : 入力トークン列
        target_tokens    : 教師トークン列（None なら推論のみ）
        simulation_steps : LIF シミュレーションのタイムステップ数
        return_input_len : True → (preds, input_len) を返す（末尾ゴミ除去に使用）
        """
        input_len = len(tokens)
        padded_tokens = (tokens + [self._pad_id] * self.seq_len)[: self.seq_len]

        for layer in self.layers:
            layer.reset()

        acc_spikes = [[0.0] * self.d_model for _ in range(self.seq_len)]
        input_v = [[0.0] * self.d_model for _ in range(self.seq_len)]

        # --- Spike Simulation ---
        for t in range(simulation_steps):
            layer_in = []
            for i, tk in enumerate(padded_tokens):
                spks = []
                for d in range(self.d_model):
                    input_v[i][d] += (
                        self.token_signatures[tk][d] + self.pos_signatures[i][d]
                    ) * 0.4
                    if input_v[i][d] >= 1.0:
                        input_v[i][d] -= 1.0
                        spks.append(True)
                    else:
                        spks.append(False)
                layer_in.append(spks)

            for layer in self.layers:
                layer_in = layer.step(layer_in, t)

            for i in range(self.seq_len):
                for d in range(self.d_model):
                    if layer_in[i][d]:
                        acc_spikes[i][d] += 1.0

        # --- Background Noise Cancellation Decoding ---
        preds = []
        for i in range(self.seq_len):
            row_sum = sum(acc_spikes[i])
            if row_sum < 0.1:
                preds.append(self._pad_id)
                continue

            avg_s = row_sum / self.d_model
            diff_x = [max(0.0, s - avg_s) for s in acc_spikes[i]]
            norm_x = math.sqrt(sum(s * s for s in diff_x)) + 1e-9

            best_v = self._pad_id
            best_score = -2.0
            # 修正J: decoder_w[i][v] を参照（位置iのベクトルを使用）
            for v in range(self.vocab_size):
                dot = sum(
                    diff_x[d] * self.decoder_w[i][v][d]
                    for d in range(self.d_model)
                )
                score = dot / norm_x
                if score > best_score:
                    best_score = score
                    best_v = v
            preds.append(best_v)

        # --- 正例のみ Hebbian 学習 ---
        if target_tokens:
            padded_targets = (target_tokens + [self._pad_id] * self.seq_len)[
                : self.seq_len
            ]
            for i, (tar, pre) in enumerate(zip(padded_targets, preds)):
                if tar == self._pad_id:
                    continue

                avg_s = sum(acc_spikes[i]) / self.d_model
                diff_x = [max(0.0, s - avg_s) for s in acc_spikes[i]]
                norm_x = math.sqrt(sum(s * s for s in diff_x)) + 1e-9

                if norm_x > 0.1:
                    vec = [s / norm_x for s in diff_x]
                    for d in range(self.d_model):
                        # 修正K: decoder_w[i][tar] のみ更新（位置をまたいだ干渉を排除）
                        self.decoder_w[i][tar][d] += self._lr_pos * vec[d]

                    self._dirty.add((i, tar))

        if return_input_len:
            return preds, input_len
        return preds