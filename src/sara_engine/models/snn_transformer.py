# パス: src/sara_engine/models/snn_transformer.py
#
# ファイルタイトル: スパイキング・トランスフォーマーモデル v1.5.0
#
# v1.4.3 からの主な変更点:
#   [追加] NeuronActivityTracker クラス
#          - 各ニューロンの「指数移動平均（EMA）発火率」を追跡する
#          - 旧 token_counts（単純カウント）を置き換え
#          - 生物的根拠: Turrigiano & Nelson (2004) シナプス恒常性
#
#   [追加] SynapticScalingManager クラス
#          - ニューロンの実際の発火率が目標発火率からずれたとき、
#            入力シナプス全体を一律スケーリングして目標に引き戻す
#          - マジックナンバー 0.85 を活動履歴から自動導出される係数に置き換え
#          - 生物的根拠: AMPA受容体の挿入/除去による全シナプス均等スケーリング
#
#   [追加] Lateral Inhibition (側抑制 / WTA) in forward_step
#          - 勝者ニューロンが発火した瞬間、他候補のポテンシャルを抑圧する
#          - 生物的根拠: Maass (2000) Liquid State Machine, 抑制性介在ニューロン
#
#   [変更] ALIF 閾値ジャンプの「1.5倍」マジックナンバーを廃止
#          - ジャンプ量をニューロンのEMA発火率から自動計算するように変更
#
#   [変更] update_weights の lr_scale を NeuronActivityTracker から取得
#          - 旧: 1.0 / sqrt(count)（整数カウントのみ）
#          - 新: EMA発火率の逆数に基づく適応的スケーリング
#
# 生物学的根拠の参照論文:
#   - Turrigiano & Nelson (2004) "Homeostatic plasticity in the developing
#     nervous system" Nature Reviews Neuroscience
#   - Turrigiano (2008) "The self-tuning neuron: synaptic scaling of
#     excitatory synapses" Cell
#   - Maass (2000) "Neural computation with winner-take-all as the only
#     nonlinear operation" NIPS
#   - Bellec et al. (2020) "A solution to the learning dilemma for recurrent
#     networks of spiking neurons" Nature Communications

from sara_engine.core.spike_attention import SpikeMultiPathwayAttention
from sara_engine.nn.attention import SpikeFuzzyAttention
from sara_engine import nn
from typing import List, Dict, Optional, Tuple
from collections import Counter
import operator
import pickle
import random
import os
import json
import math

# ---- 定数 ----------------------------------------------------------------
_MODEL_VERSION: str = "1.5.0"
_SYNAPSE_MAX_WEIGHT: float = 20.0
_SYNAPSE_PRUNE_THRESH: float = 1.0
_SYNAPSE_BUCKET_MAX: int = 8192
_SYNAPSE_PRUNE_TARGET: int = 4096

# ---- Synaptic Scaling 定数 -----------------------------------------------
# EMAの時定数。大きいほど過去の履歴を長く参照する。
# 生物的根拠: 皮質ニューロンの恒常性調整は時間スケールが時間〜日単位（Turrigiano 2008）
# ここでは学習ステップ換算で「過去約50ステップ」を参照する近似値として設定。
# この値はデータ量に依存せず「何ステップ分の記憶を持つか」という構造的パラメータ。
_ACTIVITY_EMA_ALPHA: float = 1.0 / 50.0

# Synaptic Scalingの学習率。生物的な恒常性調整は「遅い」ことが本質。
# 急激なスケーリングは学習を壊すため、小さく保つ。
_SCALING_LEARNING_RATE: float = 0.02

# Lateral Inhibitionの強度。
# 生物的根拠: 皮質の興奮性/抑制性比率は概ね 8:2（Maass 2000）。
# 勝者が他を完全に黙らせるのではなく、適度に抑圧する。
_LATERAL_INHIBITION_STRENGTH: float = 0.6


# ==========================================================================
# NeuronActivityTracker
# ==========================================================================
class NeuronActivityTracker:
    """
    各ニューロン（トークンID）の「指数移動平均（EMA）発火率」を追跡する。

    生物的対応: 皮質ニューロンが自身の平均発火率を
    カルシウムイオン（Ca2+）濃度として積算・追跡する機構。
    （Turrigiano & Nelson 2004 参照）

    旧 token_counts（単純な整数カウント）の代替。
    単純カウントは「古い情報と新しい情報を同等に扱う」問題があった。
    EMAは「最近の活動をより重く、古い活動を指数的に忘れる」ため
    より生物的なタイムスケールの近似となる。
    """

    def __init__(self, alpha: float = _ACTIVITY_EMA_ALPHA) -> None:
        # alpha: EMAの学習率。小さいほど「長い記憶」を持つ
        self._alpha: float = alpha
        # ニューロンIDごとの EMA 発火率（0.0〜1.0 の連続値）
        self._ema_rate: Dict[int, float] = {}
        # 互換性のための総ステップ数カウント（lr_scale 計算に使用）
        self._step_count: Dict[int, int] = {}

    def update(self, neuron_id: int, fired: bool) -> None:
        """
        1ステップごとに呼び出す。
        fired=True: そのニューロンが発火した
        fired=False: 発火しなかった（沈黙）

        EMA更新式: rate_new = (1 - alpha) * rate_old + alpha * x
        x = 1.0 (発火) or 0.0 (沈黙)
        """
        current = self._ema_rate.get(neuron_id, 0.0)
        signal = 1.0 if fired else 0.0
        self._ema_rate[neuron_id] = (1.0 - self._alpha) * current + self._alpha * signal
        count = self._step_count.get(neuron_id, 0) + 1
        self._step_count[neuron_id] = count

    def get_rate(self, neuron_id: int) -> float:
        """EMA発火率を返す。未観測ニューロンは 0.0 を返す。"""
        return self._ema_rate.get(neuron_id, 0.0)

    def get_step_count(self, neuron_id: int) -> int:
        """互換性用: 観測ステップ総数を返す。"""
        return self._step_count.get(neuron_id, 0)

    def get_lr_scale(self, neuron_id: int) -> float:
        """
        学習率スケールを返す。
        旧: 1.0 / sqrt(count)
        新: EMA発火率が高い（頻出）ほど学習率を下げる。
            未学習ニューロンは学習率 1.0 から始まる。
        """
        rate = self.get_rate(neuron_id)
        # rateが0に近いとき（未学習）は lr_scale = 1.0
        # rateが高い（頻出）ほど lr_scale は小さくなる
        # 分母の +0.01 はゼロ除算防止
        return 1.0 / (1.0 + rate * 10.0 + 0.01)

    def state_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            "ema_rate": dict(self._ema_rate),
            "step_count": dict(self._step_count),
        }
        return result

    def load_state_dict(self, state: Dict[str, object]) -> None:
        ema_raw = state["ema_rate"]
        step_raw = state["step_count"]
        assert isinstance(ema_raw, dict)
        assert isinstance(step_raw, dict)
        self._ema_rate = {int(k): float(v) for k, v in ema_raw.items()}
        self._step_count = {int(k): int(v) for k, v in step_raw.items()}


# ==========================================================================
# SynapticScalingManager
# ==========================================================================
class SynapticScalingManager:
    """
    Turrigiano型シナプス恒常性（Synaptic Homeostasis）の実装。

    生物的対応:
      ニューロンが目標発火率より「低すぎる」とき、
      AMPA受容体を新たに挿入してシナプス全体を均等に強化する。
      「高すぎる」とき、AMPA受容体を除去して均等に弱化する。
      （Turrigiano 2008 "The self-tuning neuron" Cell）

    重要な性質:
      - 全シナプスを「均等に」スケーリングするため、
        学習済みのシナプス強度の相対比は保たれる（記憶が消えない）。
      - BCM則と異なり、スケーリング主体（AMPA受容体制御）が
        生物学的に実証されている。

    旧コードの 0.85 ** count という固定ペナルティの代替。
    """

    @staticmethod
    def compute_scale_factor(
        actual_rate: float,
        target_rate: float,
    ) -> float:
        """
        実際のEMA発火率と目標発火率の差から、
        シナプスに乗じるスケール係数を計算する。

        actual_rate > target_rate: scale < 1.0 → シナプスを弱化（抑圧）
        actual_rate < target_rate: scale > 1.0 → シナプスを強化
        actual_rate ≈ target_rate: scale ≈ 1.0 → ほぼ変化なし

        スケール係数は 1.0 を中心とした小さな補正にとどめる
        （_SCALING_LEARNING_RATE で振れ幅を制限）。
        """
        error = target_rate - actual_rate
        # error > 0: 発火不足 → 強化
        # error < 0: 発火過多 → 抑制
        delta = _SCALING_LEARNING_RATE * error
        raw_scale = 1.0 + delta
        # 一度の調整で極端に変化しないよう [0.8, 1.2] にクリップ
        scale = max(0.8, min(1.2, raw_scale))
        return scale

    @staticmethod
    def apply_to_synapse_bucket(
        synapses: Dict[int, float],
        neuron_id: int,
        actual_rate: float,
        target_rate: float,
    ) -> None:
        """
        readout_synapses の1バケット（1スパイクニューロン分の出力結合）に
        Synaptic Scalingを適用する。

        neuron_id に対応するシナプス重みだけを対象とし、
        全重みを同じ係数で乗算する（均等スケーリング）。
        """
        if neuron_id not in synapses:
            return

        scale = SynapticScalingManager.compute_scale_factor(actual_rate, target_rate)

        # 均等スケーリング: 相対的な重みの比率は変わらない
        current = synapses[neuron_id]
        scaled = current * scale
        scaled = min(_SYNAPSE_MAX_WEIGHT, max(0.0, scaled))
        if scaled <= 0.0:
            del synapses[neuron_id]
        else:
            synapses[neuron_id] = scaled

    @staticmethod
    def compute_refractory_penalty(
        actual_rate: float,
        repeat_count: int,
    ) -> float:
        """
        生成時のリフラクトリ（不応期）ペナルティを
        活動履歴から自動計算する。

        旧コード: 0.85 ** count  ← マジックナンバー
        新コード: ニューロンの実際のEMA発火率と繰り返し回数から自動導出

        生物的根拠:
          繰り返し発火したニューロンは Na+ チャネルが不活性化し、
          次の発火に必要な電位回復に時間がかかる（不応期）。
          この「疲労の深さ」はそのニューロンの活動量に比例する。

        Returns:
          0.0〜1.0 の乗算係数（小さいほど強い抑圧）
        """
        # 高活性ニューロン（actual_rate が高い）は疲労しやすい → 強い抑圧
        # 低活性ニューロンは疲労しにくい → 弱い抑圧
        # actual_rate=0.0 のとき base_penalty=0.95（ほぼ抑圧なし）
        # actual_rate=1.0 のとき base_penalty=0.70（強い抑圧）
        base_penalty = 0.95 - actual_rate * 0.25
        base_penalty = max(0.5, min(0.95, base_penalty))
        penalty = base_penalty ** repeat_count
        return penalty


# ==========================================================================
# SynapseManager  (変更: update_weights に SynapticScaling を統合)
# ==========================================================================
class SynapseManager:
    @staticmethod
    def prune(synapses: Dict[int, float], protect_id: int) -> None:
        if len(synapses) <= _SYNAPSE_BUCKET_MAX:
            return

        weak_keys = [
            k for k, v in synapses.items()
            if v < _SYNAPSE_PRUNE_THRESH and k != protect_id
        ]
        for k in weak_keys:
            del synapses[k]

        if len(synapses) > _SYNAPSE_BUCKET_MAX:
            sorted_keys = sorted(synapses.keys(), key=lambda k: synapses[k])
            for k in sorted_keys[:_SYNAPSE_PRUNE_TARGET]:
                if k != protect_id:
                    del synapses[k]

    @staticmethod
    def update_weights(
        synapses: Dict[int, float],
        target_id: int,
        predicted_id: int,
        margin: float,
        lr_scale: float,
        activity_tracker: "NeuronActivityTracker",
        target_firing_rate: float,
    ) -> None:
        """
        シナプス重み更新 + Synaptic Scaling の統合処理。

        変更点:
          - lr_scale は呼び出し元で NeuronActivityTracker から取得済み
          - update後に SynapticScaling を適用し、
            target_id のシナプスを目標発火率に向けて自律調整する
          - 旧: total_weight > MAX*5.0 の一括スケールは残す（安全弁として）
        """
        is_correct = (predicted_id == target_id)

        if is_correct:
            reward_factor = max(0.5, 4.0 - margin)
            punish_factor = 0.2
        else:
            surprise = 1.0 + margin
            punish_factor = min(2.5, surprise * 1.5)
            reward_factor = 1.5

        current_w = synapses.get(target_id, 0.0)
        new_w = min(_SYNAPSE_MAX_WEIGHT, current_w + (1.5 * reward_factor * lr_scale))
        synapses[target_id] = new_w

        if not is_correct and predicted_id in synapses:
            synapses[predicted_id] -= punish_factor * 1.0
            if synapses[predicted_id] <= 0.0:
                del synapses[predicted_id]

        # === Synaptic Scaling: 目標発火率への自律的な恒常性調整 ===
        actual_rate = activity_tracker.get_rate(target_id)
        SynapticScalingManager.apply_to_synapse_bucket(
            synapses, target_id, actual_rate, target_firing_rate
        )

        # 安全弁: 総重みが上限を大幅に超えた場合の一括正規化（旧来処理を維持）
        total_weight = sum(synapses.values())
        max_total = _SYNAPSE_MAX_WEIGHT * 5.0
        if total_weight > max_total:
            scale = max_total / total_weight
            dead_keys = []
            for k in list(synapses.keys()):
                synapses[k] *= scale
                if synapses[k] < 0.1:
                    dead_keys.append(k)
            for k in dead_keys:
                del synapses[k]

        SynapseManager.prune(synapses, protect_id=target_id)

    @staticmethod
    def sample_temperature(candidates: List[Tuple[int, float]], temperature: float) -> int:
        weights = [pow(max(1e-9, item[1]), 1.0 / temperature) for item in candidates]
        total_weight = sum(weights)
        r = random.uniform(0.0, total_weight)
        cumulative = 0.0
        for item, w in zip(candidates, weights):
            cumulative += w
            if r <= cumulative:
                return item[0]
        return candidates[0][0]


# ==========================================================================
# NGramSpikeGenerator  (変更なし)
# ==========================================================================
class NGramSpikeGenerator:
    @staticmethod
    def generate_spikes(
        delay_buffer: List[int],
        num_ngram_levels: int,
        reservoir_size: int
    ) -> List[int]:
        spikes: set = set()

        for n in range(1, min(num_ngram_levels + 1, len(delay_buffer) + 1)):
            ngram = tuple(delay_buffer[:n])

            h = 0x811C9DC5
            for i, t in enumerate(ngram):
                h ^= t
                h = (h * 0x01000193) & 0xFFFFFFFF
                h ^= (i + 1) * 0x9E3779B9
                h = (h * 0x01000193) & 0xFFFFFFFF

            num_spikes = 8 + (n * 4)
            state = h
            if state == 0:
                state = 1

            for _ in range(num_spikes):
                state ^= (state << 13) & 0xFFFFFFFF
                state ^= (state >> 17) & 0xFFFFFFFF
                state ^= (state << 5) & 0xFFFFFFFF

                neuron_id = (state % reservoir_size) + (n - 1) * reservoir_size
                spikes.add(neuron_id)

        return list(spikes)


# ==========================================================================
# SNNTransformerConfig
# ==========================================================================
class SNNTransformerConfig:
    def __init__(
        self,
        vocab_size: int = 4096,
        embed_dim: int = 64,
        num_layers: int = 1,
        ffn_dim: int = 256,
        num_pathways: int = 4,
        dropout_p: float = 0.1,
        target_spikes_ratio: float = 0.15,
        use_fuzzy: bool = False,
        replay_count: int = 2,
    ) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.num_pathways = num_pathways
        self.dropout_p = dropout_p
        self.target_spikes_ratio = target_spikes_ratio
        self.use_fuzzy = use_fuzzy
        self.replay_count = replay_count

    def to_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            "model_version": _MODEL_VERSION,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "ffn_dim": self.ffn_dim,
            "num_pathways": self.num_pathways,
            "dropout_p": self.dropout_p,
            "target_spikes_ratio": self.target_spikes_ratio,
            "use_fuzzy": self.use_fuzzy,
            "replay_count": self.replay_count,
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "SNNTransformerConfig":
        filtered: Dict[str, object] = {
            k: v for k, v in data.items() if k != "model_version"
        }
        return cls(**filtered)


# ==========================================================================
# SNNTransformerBlock  (変更なし)
# ==========================================================================
class SNNTransformerBlock(nn.SNNModule):
    def __init__(self, config: SNNTransformerConfig) -> None:
        super().__init__()
        self.config = config

        target_spikes = max(1, int(config.embed_dim * config.target_spikes_ratio))

        self.norm1 = nn.SpikeLayerNorm(target_spikes=target_spikes)
        self.dropout1 = nn.SpikeDropout(p=config.dropout_p)

        if config.use_fuzzy:
            self.attention = SpikeFuzzyAttention(
                embed_dim=config.embed_dim,
                threshold=0.2,
                top_k=3,
            )
        else:
            self.attention = SpikeMultiPathwayAttention(
                embed_dim=config.embed_dim,
                num_pathways=config.num_pathways,
                context_size=128,
            )

        self.norm2 = nn.SpikeLayerNorm(target_spikes=target_spikes)
        self.dropout2 = nn.SpikeDropout(p=config.dropout_p)

        self.ffn = nn.Sequential(
            nn.LinearSpike(
                in_features=config.embed_dim,
                out_features=config.ffn_dim,
                density=0.2,
            ),
            nn.LinearSpike(
                in_features=config.ffn_dim,
                out_features=config.embed_dim,
                density=0.2,
            ),
        )
        self.max_block_spikes: int = max(1, config.embed_dim // 2)

    def forward(self, spikes: List[int], learning: bool = True) -> List[int]:
        norm_spikes1 = self.norm1(spikes, learning=learning)
        attn_out = self.attention.forward(norm_spikes1, learning=learning)
        drop_attn = self.dropout1(attn_out, learning=learning)
        res1_spikes: List[int] = list(set(spikes + drop_attn))

        norm_spikes2 = self.norm2(res1_spikes, learning=learning)
        ffn_out = self.ffn(norm_spikes2, learning=learning)
        drop_ffn = self.dropout2(ffn_out, learning=learning)
        res2_spikes: List[int] = list(set(res1_spikes + drop_ffn))

        if len(res2_spikes) > self.max_block_spikes:
            res2_spikes = sorted(res2_spikes)[:self.max_block_spikes]

        return res2_spikes


# ==========================================================================
# SpikingTransformerModel  (主要変更箇所)
# ==========================================================================
class SpikingTransformerModel(nn.SNNModule):
    def __init__(self, config: SNNTransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.context_length: int = 16
        self.reservoir_size: int = 8192
        self.num_ngram_levels: int = 5

        self.total_readout_size: int = (
            self.reservoir_size * self.num_ngram_levels
        ) + config.embed_dim

        layers: List[SNNTransformerBlock] = [
            SNNTransformerBlock(config) for _ in range(config.num_layers)
        ]
        self.transformer_layers = nn.Sequential(*layers)

        self.delay_buffer: List[int] = []

        self.readout_synapses: List[Dict[int, float]] = [
            {} for _ in range(self.total_readout_size)
        ]

        # === v1.5.0: token_counts を NeuronActivityTracker に置き換え ===
        # 旧: self.token_counts: Dict[int, int] = {}
        self.activity_tracker = NeuronActivityTracker(alpha=_ACTIVITY_EMA_ALPHA)

        # ALIF: 適応的閾値（会話単位でリセット）
        self.adaptive_thresholds: Dict[int, float] = {}

        self.register_state("readout_synapses")
        self.register_state("adaptive_thresholds")
        # NeuronActivityTracker は独自の state_dict を持つ
        # register_state は使わず save/load で個別に処理する

    def reset_state(self) -> None:
        """
        会話の区切りでスパイク状態をリセット。
        adaptive_thresholds（疲労）はリセットするが、
        activity_tracker（長期記憶）はリセットしない。

        生物的根拠:
          短期的な発火疲労（不応期）は会話ごとにリセットされるが、
          長期的なシナプス恒常性の基準（EMA発火率）は
          訓練を通じて蓄積される記憶であり、リセットしてはならない。
        """
        super().reset_state()
        self.delay_buffer.clear()
        self.adaptive_thresholds.clear()
        for layer in getattr(self.transformer_layers, "modules", []):
            if hasattr(layer, "attention"):
                layer.attention.reset_state()

    def _apply_lateral_inhibition(
        self,
        out_potentials: Dict[int, float],
        sorted_items: List[Tuple[int, float]],
    ) -> None:
        """
        Lateral Inhibition（側抑制）/ Winner-Take-All の適用。

        生物的根拠:
          皮質の抑制性介在ニューロン（GABAニューロン）は
          勝者ニューロンの発火と同時に他のニューロンを抑圧する。
          Maass (2000) の Liquid State Machine では
          興奮性:抑制性 = 8:2 の比率が情報処理効率を最大化することを示した。

        実装:
          - 勝者（最大ポテンシャル）と他候補のポテンシャル差に比例した抑制を加える
          - 完全な抑圧（0にする）ではなく _LATERAL_INHIBITION_STRENGTH で緩和
          - 勝者自身は変化しない

        Args:
          out_potentials: 更新対象の辞書（in-place変更）
          sorted_items: ポテンシャル降順にソート済みのリスト
        """
        if len(sorted_items) < 2:
            return

        winner_id = sorted_items[0][0]
        winner_pot = sorted_items[0][1]

        for tid, pot in sorted_items[1:]:
            if tid == winner_id:
                continue
            # 抑制量: 勝者との差に比例 × 強度係数
            # 差が大きいほど（勝者が圧倒的なほど）抑制は強くなる
            inhibition = (winner_pot - pot) * _LATERAL_INHIBITION_STRENGTH
            suppressed = pot - inhibition
            out_potentials[tid] = max(0.0, suppressed)

    def forward_step(
        self,
        token_id: int,
        learning: bool = True,
        target_id: Optional[int] = None,
        refractory_tokens: Optional[List[int]] = None,
        temperature: float = 0.6,
        fire_threshold: float = 1.0,
        debug: bool = False
    ) -> Tuple[int, Dict]:

        def _is_valid_output_token(tid: int) -> bool:
            if tid <= 0 or tid >= self.config.vocab_size:
                return False
            return True

        base_threshold = fire_threshold

        # === 1. ALIF: ニューロンの疲労回復（毎ステップごとに閾値が下がる） ===
        # 疲労回復の速度を「固定の 0.9」から「目標発火率に基づく動的な値」に変更。
        # 生物的根拠: 高活性ニューロンほど回復が遅い（Na+チャネルの不活性化からの回復）
        if not learning:
            target_rate = self.config.target_spikes_ratio
            keys_to_delete = []
            for tid, th in self.adaptive_thresholds.items():
                tid_rate = self.activity_tracker.get_rate(tid)
                # 高活性ニューロン（rate高）は回復係数が大きい → 回復が遅い
                # 低活性ニューロン（rate低）は回復係数が小さい → 回復が速い
                recovery_tau = 0.85 + tid_rate * 0.12
                recovery_tau = max(0.80, min(0.97, recovery_tau))
                new_th = th * recovery_tau + base_threshold * (1.0 - recovery_tau)
                if new_th <= base_threshold + 0.01:
                    keys_to_delete.append(tid)
                else:
                    self.adaptive_thresholds[tid] = new_th
            for tid in keys_to_delete:
                del self.adaptive_thresholds[tid]

        self.delay_buffer.insert(0, token_id)
        if len(self.delay_buffer) > self.context_length:
            self.delay_buffer.pop()

        res_spikes = NGramSpikeGenerator.generate_spikes(
            self.delay_buffer, self.num_ngram_levels, self.reservoir_size
        )

        unigram_spikes = [s for s in res_spikes if s < self.reservoir_size]
        block_input_set = set([s % self.config.embed_dim for s in unigram_spikes])
        block_input = sorted(list(block_input_set))

        block_spikes: List[int] = self.transformer_layers(block_input, learning=learning)

        block_offset = self.reservoir_size * self.num_ngram_levels
        readout_spikes: List[int] = list(
            set(res_spikes + [s + block_offset for s in block_spikes])
        )

        out_potentials: Dict[int, float] = {}
        for s in readout_spikes:
            if s < self.total_readout_size:
                for v_idx, w in self.readout_synapses[s].items():
                    current = out_potentials.get(v_idx, 0.0)
                    out_potentials[v_idx] = current + w

        # ポテンシャルの正規化: EMA発火率ベースに変更
        # 旧: math.log1p(fan_in) * 0.15 + 1.0（token_counts 依存）
        # 新: EMA発火率から自動計算。頻出ニューロンほど強く正規化される。
        if out_potentials:
            for v_idx in list(out_potentials.keys()):
                ema_rate = self.activity_tracker.get_rate(v_idx)
                # ema_rate=0のとき normalizer=1.0（正規化なし）
                # ema_rate=1のとき normalizer=2.5（強い正規化）
                normalizer = 1.0 + ema_rate * 1.5
                out_potentials[v_idx] /= normalizer

        # === 2. Refractory（不応期）ペナルティ ===
        # 旧: 固定 0.85 ** count
        # 新: SynapticScalingManager.compute_refractory_penalty によりEMA発火率から自動計算
        if not learning and refractory_tokens:
            recent_window = 10
            recent_tokens = refractory_tokens[-recent_window:]

            # 位置ベースの時系列減衰（最近のものほど強く抑圧）
            decay_factor = 0.1
            step_size = 0.9 / recent_window
            for r_tok in reversed(recent_tokens):
                if r_tok in out_potentials:
                    out_potentials[r_tok] *= decay_factor
                decay_factor += step_size
                if decay_factor > 1.0:
                    decay_factor = 1.0

            # 繰り返し回数ベースの不応期ペナルティ（EMA発火率から自動計算）
            counts = Counter(refractory_tokens)
            for r_tok, count in counts.items():
                if r_tok in out_potentials:
                    actual_rate = self.activity_tracker.get_rate(r_tok)
                    penalty = SynapticScalingManager.compute_refractory_penalty(
                        actual_rate, count
                    )
                    out_potentials[r_tok] *= penalty

        predicted_id = 0
        margin = 0.0
        debug_info: Dict[str, object] = {"top_k": [], "stop_reason": ""}

        if out_potentials:
            if not learning:
                for k in out_potentials:
                    out_potentials[k] *= random.uniform(0.95, 1.05)

            sorted_items = sorted(
                out_potentials.items(),
                key=operator.itemgetter(1),
                reverse=True,
            )

            if debug:
                debug_info["top_k"] = sorted_items[:5]

            if learning:
                if sorted_items[0][1] > 0.1:
                    predicted_id = sorted_items[0][0]
                    if len(sorted_items) > 1:
                        margin = sorted_items[0][1] - sorted_items[1][1]
                    else:
                        margin = sorted_items[0][1]
            else:
                # === 3. Lateral Inhibition（側抑制）===
                # 候補評価の前に WTA 抑制を適用する
                self._apply_lateral_inhibition(out_potentials, sorted_items)

                # 抑制後に再ソート
                sorted_items = sorted(
                    out_potentials.items(),
                    key=operator.itemgetter(1),
                    reverse=True,
                )

                top_k = sorted_items[:5]
                valid_candidates: List[Tuple[int, float]] = []

                # === 4. ALIF: トークン固有の適応的閾値での発火判定 ===
                for tid, pot in top_k:
                    if not _is_valid_output_token(tid):
                        continue
                    token_threshold = self.adaptive_thresholds.get(tid, base_threshold)
                    if pot > token_threshold:
                        valid_candidates.append((tid, pot))

                if not valid_candidates and top_k:
                    relaxed_base = base_threshold * 0.5
                    for tid, pot in top_k:
                        if not _is_valid_output_token(tid):
                            continue
                        token_threshold = self.adaptive_thresholds.get(tid, relaxed_base)
                        if pot > token_threshold:
                            valid_candidates.append((tid, pot))

                    if not valid_candidates:
                        debug_info["stop_reason"] = "All potentials below ALIF adaptive thresholds."

                if valid_candidates:
                    predicted_id = SynapseManager.sample_temperature(valid_candidates, temperature)

        # === 5. ALIF: 発火したニューロンの閾値ジャンプ ===
        # 旧: jump_target = max(actual_pot * 1.5, current_th + 2.0)  ← 1.5 がマジックナンバー
        # 新: ジャンプ量をそのニューロンのEMA発火率から自動計算
        #     高活性ニューロン（よく発火する）ほどジャンプが大きい（疲労しやすい）
        #     生物的根拠: 高活性ニューロンは Na+ チャネルの不活性化が蓄積しやすい
        if not learning and predicted_id != 0:
            current_th = self.adaptive_thresholds.get(predicted_id, base_threshold)
            actual_pot = out_potentials.get(predicted_id, 0.0)
            pred_rate = self.activity_tracker.get_rate(predicted_id)

            # ジャンプ倍率: EMA発火率が高いほど大きなジャンプ（1.3〜2.0倍）
            jump_multiplier = 1.3 + pred_rate * 0.7
            jump_target_a = actual_pot * jump_multiplier
            jump_target_b = current_th + (1.0 + pred_rate * 2.0)
            jump_target = max(jump_target_a, jump_target_b)
            self.adaptive_thresholds[predicted_id] = jump_target

            # 発火として活動追跡を更新（生成フェーズ）
            self.activity_tracker.update(predicted_id, fired=True)

        # === 6. 学習: 重み更新 + Synaptic Scaling ===
        if learning and target_id is not None:
            # NeuronActivityTrackerを更新
            self.activity_tracker.update(target_id, fired=True)

            # EMAベースの学習率スケール（旧: 1/sqrt(count)）
            lr_scale = self.activity_tracker.get_lr_scale(target_id)

            # 目標発火率: config から取得（マジックナンバーではなく設定値）
            target_firing_rate = self.config.target_spikes_ratio

            for s in readout_spikes:
                if s >= self.total_readout_size:
                    continue
                SynapseManager.update_weights(
                    self.readout_synapses[s],
                    target_id,
                    predicted_id,
                    margin,
                    lr_scale,
                    activity_tracker=self.activity_tracker,
                    target_firing_rate=target_firing_rate,
                )

        return predicted_id, debug_info

    def learn_sequence(self, input_ids: List[int]) -> None:
        sequence = input_ids + [0]
        for _ in range(self.config.replay_count):
            self.reset_state()
            for i in range(len(sequence) - 1):
                self.forward_step(
                    sequence[i],
                    learning=True,
                    target_id=sequence[i + 1],
                )

    def generate(
        self,
        input_ids: List[int],
        max_length: int = 150,
        temperature: float = 0.6,
        fire_threshold: float = 0.5,
        debug: bool = False
    ) -> Tuple[List[int], List[Dict]]:
        self.reset_state()

        first_pred = 0
        debug_logs: List[Dict] = []

        for token_id in input_ids:
            first_pred, d_info = self.forward_step(
                token_id,
                learning=False,
                temperature=temperature,
                fire_threshold=fire_threshold,
                debug=debug
            )
            if debug:
                debug_logs.append(d_info)

        generated_ids: List[int] = []
        current_token = first_pred

        for _ in range(max_length):
            if current_token <= 1:
                break

            generated_ids.append(current_token)

            current_token, d_info = self.forward_step(
                current_token,
                learning=False,
                refractory_tokens=generated_ids,
                temperature=temperature,
                fire_threshold=fire_threshold,
                debug=debug
            )

            if debug and d_info.get("stop_reason"):
                debug_logs.append(d_info)

        return generated_ids, debug_logs

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)

        config_path = os.path.join(save_directory, "config.json")
        config_dict = self.config.to_dict()
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

        state_path = os.path.join(save_directory, "model_state.pkl")
        model_state = self.state_dict()
        # NeuronActivityTracker を個別にシリアライズ
        tracker_state = self.activity_tracker.state_dict()
        save_bundle: Dict[str, object] = {
            "model_state": model_state,
            "activity_tracker": tracker_state,
        }
        with open(state_path, "wb") as f:
            pickle.dump(save_bundle, f)

        print(f"[SpikingTransformerModel] Saved to '{save_directory}' (v{_MODEL_VERSION})")

    @classmethod
    def from_pretrained(cls, save_directory: str) -> "SpikingTransformerModel":
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            raw: Dict[str, object] = json.load(f)

        config = SNNTransformerConfig.from_dict(raw)
        model = cls(config)

        state_path = os.path.join(save_directory, "model_state.pkl")
        if os.path.exists(state_path):
            with open(state_path, "rb") as f:
                bundle = pickle.load(f)

            # v1.4.x との後方互換: 旧フォーマット（bundle が dict でない）への対応
            if isinstance(bundle, dict) and "model_state" in bundle:
                model.load_state_dict(bundle["model_state"])
                tracker_raw = bundle.get("activity_tracker")
                if isinstance(tracker_raw, dict):
                    model.activity_tracker.load_state_dict(tracker_raw)
            else:
                # 旧バージョンのモデルステートをそのまま読み込む
                model.load_state_dict(bundle)

        saved_version = raw.get("model_version", "unknown")
        print(
            f"[SpikingTransformerModel] Loaded from '{save_directory}' "
            f"(saved version: {saved_version}, current: {_MODEL_VERSION})"
        )
        return model