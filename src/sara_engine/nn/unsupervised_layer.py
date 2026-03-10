# ディレクトリパス: src/sara_engine/nn/unsupervised_layer.py
# ファイルの日本語タイトル: 教師なし自己組織化スパイキング層
# ファイルの目的や内容: Greedy Layer-wise Learning に最適化された自己組織化
#   スパイキング層。競合的WTA、側方抑制、ホメオスタシス適応閾値、凍結機能を提供する。

import random
import math
from typing import List, Dict, Optional, Any

from .module import SNNModule
from ..learning.homeostasis import NeuronActivityTracker


class UnsupervisedSpikeLayer(SNNModule):
    """教師なし学習に特化した自己組織化スパイキング層。

    特徴:
        - Winner-Takes-All (WTA) による競合的スパース表現
        - 側方抑制 (Lateral Inhibition) による多様な受容野の形成
        - ホメオスタシス適応閾値による安定した発火率の維持
        - 凍結/解凍 (freeze/unfreeze) 機能
        - 競合的ヘビアン学習 (STDP ベース)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        density: float = 0.3,
        k_winners: int = 0,
        target_rate: float = 0.05,
        stdp_lr: float = 0.05,
        lateral_inhibition: float = 0.5,
        threshold: float = 0.5,
    ) -> None:
        """
        Args:
            in_features: 入力ニューロン数。
            out_features: 出力ニューロン数。
            density: シナプス結合密度 (0.0〜1.0)。
            k_winners: WTA で残す発火ニューロン数。0 の場合は 10% を自動設定。
            target_rate: ホメオスタシスの目標発火率。
            stdp_lr: STDP 学習率。
            lateral_inhibition: 側方抑制の強さ (0.0=無効, 1.0=最大)。
            threshold: 基本発火閾値。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.stdp_lr = stdp_lr
        self.lateral_inhibition = lateral_inhibition
        self.base_threshold = threshold

        # WTA のk値設定
        self.k_winners = k_winners if k_winners > 0 else max(
            1, int(out_features * 0.1))

        # --- シナプス重み: weights[pre_id][post_id] = weight ---
        self.weights: List[Dict[int, float]] = [{} for _ in range(in_features)]
        for i in range(in_features):
            num_conn = max(1, int(out_features * density))
            targets = random.sample(
                range(out_features), min(num_conn, out_features))
            for t in targets:
                self.weights[i][t] = random.uniform(0.2, 0.6)
        self.register_state("weights")

        # --- 適応閾値 (ニューロンごと) ---
        self.adaptive_thresholds: List[float] = [threshold] * out_features
        self.register_state("adaptive_thresholds")

        # --- ホメオスタシス追跡器 ---
        self._activity_tracker = NeuronActivityTracker(
            decay=0.95,
            slow_decay=0.99,
            blend_fast=0.6,
        )
        self.target_rate = target_rate

        # --- 凍結フラグ ---
        self._frozen: bool = False

    def freeze(self) -> None:
        """重みと閾値を凍結し、学習を無効にする。"""
        self._frozen = True

    def unfreeze(self) -> None:
        """凍結を解除して学習を再開可能にする。"""
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        """凍結状態を返す。"""
        return self._frozen

    def reset_state(self) -> None:
        """動的状態のリセット（重みは保持）。"""
        self._activity_tracker.reset()

    def forward(
        self,
        in_spikes: List[int],
        learning: bool = False,
    ) -> List[int]:
        """入力スパイクを処理してWTA後の出力スパイクを返す。

        処理パイプライン:
            1. 膜電位集積 (Integrate)
            2. 適応閾値による発火判定
            3. Winner-Takes-All 競合選択
            4. 側方抑制の適用
            5. STDP 学習 (learning=True かつ非凍結時)
            6. ホメオスタシス更新

        Args:
            in_spikes: 入力スパイクのインデックスリスト。
            learning: True で STDP 学習を有効化（凍結時は無視）。

        Returns:
            出力スパイクのインデックスリスト。
        """
        # --- 1. 膜電位集積 ---
        potentials = [0.0] * self.out_features
        for s in in_spikes:
            if s < self.in_features:
                for t, w in self.weights[s].items():
                    potentials[t] += w

        # --- 2. 適応閾値による発火判定 ---
        candidates: List[int] = []
        for i in range(self.out_features):
            if potentials[i] > self.adaptive_thresholds[i]:
                candidates.append(i)

        # --- 3. Winner-Takes-All ---
        if len(candidates) > self.k_winners:
            candidates.sort(key=lambda x: potentials[x], reverse=True)
            candidates = candidates[: self.k_winners]

        # --- 4. 側方抑制 ---
        out_spikes: List[int]
        is_training = learning and not self._frozen
        if self.lateral_inhibition > 0.0 and len(candidates) > 1:
            out_spikes = self._apply_lateral_inhibition(candidates, potentials, is_training=is_training)
        else:
            out_spikes = candidates

        # 出力が空の場合、最大膜電位のニューロンを強制発火（デッドレイヤー防止）
        if not out_spikes and potentials:
            best = max(range(self.out_features), key=lambda x: potentials[x])
            if potentials[best] > 0.0:
                out_spikes = [best]

        # --- 5. STDP 学習 ---
        if is_training:
            self._apply_competitive_stdp(in_spikes, out_spikes)

        # --- 6. ホメオスタシス更新 ---
        # 凍結時は閾値の適応を行わない（決定論的動作の保証）
        if not self._frozen:
            self._update_homeostasis(out_spikes)

        return out_spikes

    # ------------------------------------------------------------------
    # 内部メソッド
    # ------------------------------------------------------------------

    def _apply_lateral_inhibition(
        self,
        candidates: List[int],
        potentials: List[float],
        is_training: bool = True,
    ) -> List[int]:
        """側方抑制: 勝者ニューロンが近隣を抑制して多様な受容野を形成する。

        膜電位で降順にソートし、すでに抑制されたニューロンは除外する。
        """
        sorted_cands = sorted(
            candidates, key=lambda x: potentials[x], reverse=True)
        surviving: List[int] = []
        suppressed: set[int] = set()

        for neuron_id in sorted_cands:
            if neuron_id in suppressed:
                continue
            surviving.append(neuron_id)

            # 近隣ニューロンを抑制
            inhibition_radius = max(1, int(self.out_features * 0.05))
            for offset in range(-inhibition_radius, inhibition_radius + 1):
                neighbor = neuron_id + offset
                if 0 <= neighbor < self.out_features and neighbor != neuron_id:
                    # 推論・凍結時は決定論的、学習時は確率的に抑制
                    if is_training:
                        if random.random() < self.lateral_inhibition:
                            suppressed.add(neighbor)
                    else:
                        if self.lateral_inhibition >= 0.5:
                            suppressed.add(neighbor)

        return surviving

    def _apply_competitive_stdp(
        self,
        in_spikes: List[int],
        out_spikes: List[int],
    ) -> None:
        """競合的ヘビアン学習 (STDP ベース)。

        - LTP: 入力と出力が同時に発火 → 結合強化
        - LTD: 入力が発火したが出力が発火しない → 結合弱化
        - 重みは [0.0, 3.0] にクリップ
        """
        out_set = set(out_spikes)
        weight_cap = 3.0

        for s in in_spikes:
            if s >= self.in_features:
                continue
            for t in list(self.weights[s].keys()):
                w = self.weights[s][t]
                if t in out_set:
                    # LTP: Soft-bound で飽和を抑制
                    delta = self.stdp_lr * (weight_cap - w) / weight_cap
                    self.weights[s][t] = min(weight_cap, w + delta)
                else:
                    # LTD: 軽い弱化
                    delta = self.stdp_lr * 0.3 * w / weight_cap
                    new_w = w - delta
                    if new_w <= 1e-4:
                        del self.weights[s][t]
                    else:
                        self.weights[s][t] = new_w

    def _update_homeostasis(self, out_spikes: List[int]) -> None:
        """ホメオスタシス: 発火率に基づいて各ニューロンの適応閾値を調整する。"""
        self._activity_tracker.step()
        out_set = set(out_spikes)

        for neuron_id in range(self.out_features):
            fired = neuron_id in out_set
            self._activity_tracker.update(neuron_id, fired)
            rate = self._activity_tracker.get_rate(neuron_id)

            # 目標発火率からの逸脱に基づいて閾値を調整
            error = rate - self.target_rate
            adaptation = 0.01 * math.tanh(error * 10.0)
            self.adaptive_thresholds[neuron_id] = max(
                0.05,
                min(2.0, self.adaptive_thresholds[neuron_id] + adaptation),
            )

    # ------------------------------------------------------------------
    # 永続化
    # ------------------------------------------------------------------

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
    ) -> Dict[str, Any]:
        """状態辞書を返す。"""
        result = super().state_dict(destination, prefix)
        result[prefix + "frozen"] = self._frozen
        return result

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        strict: bool = False,
    ) -> None:
        """状態を復元する。"""
        super().load_state_dict(state_dict, strict)
        self._frozen = bool(state_dict.get("frozen", False))