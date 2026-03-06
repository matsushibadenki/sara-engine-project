# ディレクトリパス: src/sara_engine/models/strong_spiking_lm.py
# ファイルの日本語タイトル: 強いAI向けスパイキング言語モデル
# ファイルの目的や内容: 階層コンテキスト(l1+l2)とバイグラム文脈（直前トークン記憶）の
#   統合、頻度正規化学習、乗算的不応期によるコンテキスト依存テキスト生成。

import math
import os
import json
import pickle
import random
from typing import List, Optional, Dict, Tuple
from ..nn.module import SNNModule
from ..nn.predictive import SpikingPredictiveLayer


class StrongSpikingLMConfig:
    def __init__(self, vocab_size: int = 65536, embed_dim: int = 256, context_dim: int = 512):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_dim = context_dim

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "context_dim": self.context_dim,
        }


class StrongSpikingLM(SNNModule):
    def __init__(self, config: StrongSpikingLMConfig):
        super().__init__()
        self.config = config

        # 階層型予測レイヤーの構築
        self.layer1 = SpikingPredictiveLayer(
            config.vocab_size, config.embed_dim)
        self.layer2 = SpikingPredictiveLayer(
            config.embed_dim, config.context_dim)

        # --- Readout層 ---
        # バイグラム連想記憶: (前のトークン, 現在のトークン) → 次のトークンの予測
        # スペースのような高頻度トークンでも前のトークンが異なれば異なる予測ができる
        self.bigram_weights: Dict[Tuple[int, int], Dict[int, float]] = {}
        self.register_state("bigram_weights")

        # L1/L2コンテキストからの補助予測 (embed_dim + context_dim)
        self.ctx_features = config.embed_dim + config.context_dim
        self.ctx_weights: List[Dict[int, float]] = [{}
                                                    for _ in range(self.ctx_features)]
        self.register_state("ctx_weights")

        # ターゲットトークンの出現頻度を追跡（頻度正規化用）
        self.target_counts: Dict[int, int] = {}
        self.register_state("target_counts")

        # 内部状態
        self.prev_token_id: int = -1  # 直前のトークンID
        self.current_state: List[int] = []

    def forward_step(self, token_id: int, learning: bool = True,
                     refractory: Optional[List[int]] = None) -> int:
        # 上位層の現在の状態をトップダウン入力として利用
        td_spikes = list(self.layer2.last_state)

        # 階層順伝播 (予測符号化)
        l1_s, l1_p = self.layer1.forward(
            [token_id], top_down=td_spikes, learning=learning)
        l2_s, l2_p = self.layer2.forward(l1_s, learning=learning)

        # L1/L2コンテキスト特徴（readout補助用）
        l2_offset = self.config.embed_dim
        ctx_indices = l1_s + [s + l2_offset for s in l2_s]
        self.current_state = ctx_indices

        # --- 次トークンのポテンシャル計算 ---

        # (1) バイグラム予測: (前のトークン, 現在のトークン) → 次のトークン
        pot_bigram: Dict[int, float] = {}
        bigram_key = (self.prev_token_id, token_id)
        if bigram_key in self.bigram_weights:
            for t, w in self.bigram_weights[bigram_key].items():
                pot_bigram[t] = pot_bigram.get(t, 0.0) + w

        # (2) ユニグラム予測: 現在のトークン → 次のトークン（バイグラムのフォールバック）
        pot_unigram: Dict[int, float] = {}
        unigram_key = (-1, token_id)  # prev=-1は「任意の前トークン」
        if unigram_key in self.bigram_weights:
            for t, w in self.bigram_weights[unigram_key].items():
                pot_unigram[t] = pot_unigram.get(t, 0.0) + w

        # (3) L1/L2コンテキスト予測（個数で正規化）
        pot_ctx: Dict[int, float] = {}
        n_ctx = len(ctx_indices) if ctx_indices else 1
        for s in ctx_indices:
            if s < len(self.ctx_weights):
                for t, w in self.ctx_weights[s].items():
                    pot_ctx[t] = pot_ctx.get(t, 0.0) + w / n_ctx

        # 混合: バイグラム優先(50%) > ユニグラム(30%) > コンテキスト(20%)
        pot: Dict[int, float] = {}
        all_targets = set(pot_bigram.keys()) | set(
            pot_unigram.keys()) | set(pot_ctx.keys())
        for t in all_targets:
            pot[t] = (pot_bigram.get(t, 0.0) * 0.5
                      + pot_unigram.get(t, 0.0) * 0.3
                      + pot_ctx.get(t, 0.0) * 0.2)

        # 乗算的不応期ペナルティ (反復ループの物理的排除)
        if not learning and refractory and pot:
            for i, rt in enumerate(reversed(refractory)):
                if rt in pot:
                    pot[rt] *= (0.1 * i)

        # 直前トークンの更新
        self.prev_token_id = token_id

        if pot:
            best_id, best_p = max(
                pot.items(), key=lambda x: x[1] + random.uniform(0, 0.01))
            return best_id if best_p > 0.05 else 0
        return 0

    def learn_sequence(self, ids: List[int]) -> None:
        self.reset_state()
        for i in range(len(ids) - 1):
            curr_token = ids[i]
            self.forward_step(curr_token, learning=True)
            target = ids[i + 1]

            # ターゲット頻度の追跡
            self.target_counts[target] = self.target_counts.get(target, 0) + 1
            count = self.target_counts[target]

            # 頻度正規化した学習率
            lr = 1.0 / math.sqrt(count)

            # (1) バイグラム結合の学習: (前のトークン, 現在のトークン) → ターゲット
            prev_token = ids[i - 1] if i > 0 else -1
            bigram_key = (prev_token, curr_token)
            if bigram_key not in self.bigram_weights:
                self.bigram_weights[bigram_key] = {}
            self.bigram_weights[bigram_key][target] = min(
                5.0, self.bigram_weights[bigram_key].get(
                    target, 0.0) + lr * 2.0
            )

            # ユニグラムも同時に学習（フォールバック用）
            unigram_key = (-1, curr_token)
            if unigram_key not in self.bigram_weights:
                self.bigram_weights[unigram_key] = {}
            self.bigram_weights[unigram_key][target] = min(
                3.0, self.bigram_weights[unigram_key].get(target, 0.0) + lr
            )

            # (2) L1/L2コンテキスト結合の学習
            for s in self.current_state:
                if s < len(self.ctx_weights):
                    self.ctx_weights[s][target] = min(
                        3.0, self.ctx_weights[s].get(target, 0.0) + lr
                    )
                    # 競合抑制 (LTD)
                    for t in list(self.ctx_weights[s].keys()):
                        if t != target:
                            self.ctx_weights[s][t] -= 0.1
                            if self.ctx_weights[s][t] <= 0:
                                del self.ctx_weights[s][t]

            # バイグラム結合のLTD（競合抑制）
            for key in [bigram_key, unigram_key]:
                for t in list(self.bigram_weights[key].keys()):
                    if t != target:
                        self.bigram_weights[key][t] -= 0.05
                        if self.bigram_weights[key][t] <= 0:
                            del self.bigram_weights[key][t]

    def generate(self, ids: List[int], max_l: int = 50) -> List[int]:
        self.reset_state()
        curr = 0
        for tid in ids:
            curr = self.forward_step(tid, learning=False)

        if curr == 0:
            return []

        gen: List[int] = []
        refr: List[int] = []
        for _ in range(max_l):
            if curr == 0:
                break
            gen.append(curr)
            refr.append(curr)
            if len(refr) > 8:
                refr.pop(0)

            curr = self.forward_step(curr, learning=False, refractory=refr)
        return gen

    def reset_state(self) -> None:
        super().reset_state()
        self.layer1.reset_state()
        self.layer2.reset_state()
        self.current_state = []
        self.prev_token_id = -1

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=4)
        with open(os.path.join(path, "model_state.pkl"), "wb") as f:
            pickle.dump(self.state_dict(), f)

    @classmethod
    def from_pretrained(cls, path: str) -> "StrongSpikingLM":
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = StrongSpikingLMConfig(**json.load(f))

        m = cls(config)
        state_path = os.path.join(path, "model_state.pkl")
        if os.path.exists(state_path):
            with open(state_path, "rb") as f:
                m.load_state_dict(pickle.load(f))
        return m
