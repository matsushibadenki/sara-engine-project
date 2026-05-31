# src/sara_engine/learning/three_factor_learning.py
# 三要素学習モジュール
# プレ/ポストシナプス活動 + 報酬信号の3要素による学習規則を実装する。
# 報酬ベースライン追跡、LTP/LTD トレースの分離、state_dict 対応を含む。

from typing import Dict, Tuple, Any


class ThreeFactorLearningManager:
    """三要素学習則（Pre-Post-Reward）を管理する。

    拡張機能:
        - 報酬ベースライン（移動平均）を追跡し、差分報酬（サプライズ）で学習
        - LTP (Long-Term Potentiation) と LTD (Long-Term Depression) のトレースを分離
        - state_dict / load_state_dict による永続化サポート
    """

    def __init__(
        self,
        lr: float = 0.01,
        trace_decay: float = 0.95,
        baseline_decay: float = 0.99,
        use_rpe: bool = True,
    ) -> None:
        """
        Args:
            lr: 学習率。
            trace_decay: トレースの時間減衰率 (0〜1)。
            baseline_decay: 報酬ベースラインの指数移動平均減衰率。
            use_rpe: True の場合、報酬予測誤差 (RPE) を使用して学習する。
        """
        self.lr = lr
        self.trace_decay = trace_decay
        self.baseline_decay = baseline_decay
        self.use_rpe = use_rpe

        # LTP トレース: (pre_id, post_id) → trace_value (正の相関)
        self._ltp_traces: Dict[Tuple[int, int], float] = {}
        # LTD トレース: (pre_id, post_id) → trace_value (負の相関)
        self._ltd_traces: Dict[Tuple[int, int], float] = {}
        # 後方互換性のための統合ビュー
        self._traces: Dict[Tuple[int, int], float] = {}
        # 報酬ベースライン（期待報酬の移動平均）
        self.reward_baseline: float = 0.0
        # 累計報酬カウント
        self.reward_count: int = 0

    def update_trace(
        self,
        pre_id: int,
        post_id: int,
        strength: float,
        time: float,
    ) -> None:
        """シナプスペアの活動トレースを更新する。

        Args:
            pre_id: シナプス前ニューロンのID。
            post_id: シナプス後ニューロンのID。
            strength: 活動の強度（正:LTP方向, 負:LTD方向）。
            time: 現在の時刻（使用しないが互換性のため保持）。
        """
        key = (pre_id, post_id)

        if strength >= 0:
            prev = self._ltp_traces.get(key, 0.0)
            self._ltp_traces[key] = prev * self.trace_decay + strength
        else:
            prev = self._ltd_traces.get(key, 0.0)
            self._ltd_traces[key] = prev * self.trace_decay + abs(strength)

        # 統合ビューの更新
        ltp = self._ltp_traces.get(key, 0.0)
        ltd = self._ltd_traces.get(key, 0.0)
        self._traces[key] = ltp - ltd

    def decay_all_traces(self) -> None:
        """全トレースに時間減衰を適用する。"""
        keys_to_remove: list[Tuple[int, int]] = []
        for key in self._ltp_traces:
            self._ltp_traces[key] *= self.trace_decay
            if self._ltp_traces[key] < 1e-6:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self._ltp_traces[key]

        keys_to_remove = []
        for key in self._ltd_traces:
            self._ltd_traces[key] *= self.trace_decay
            if self._ltd_traces[key] < 1e-6:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self._ltd_traces[key]

        # 統合ビュー再構築
        self._traces.clear()
        all_keys = set(self._ltp_traces.keys()) | set(self._ltd_traces.keys())
        for key in all_keys:
            net = self._ltp_traces.get(key, 0.0) - \
                self._ltd_traces.get(key, 0.0)
            if abs(net) > 1e-8:
                self._traces[key] = net

    def apply_reward(self, reward: float) -> Dict[Tuple[int, int], float]:
        """報酬シグナルに基づく重み更新量を計算する。

        use_rpe=True の場合、報酬予測誤差 (RPE) を使用:
            δ = reward - baseline
            Δw = lr × trace × δ

        Args:
            reward: 環境からの報酬値。

        Returns:
            {(pre_id, post_id): delta_w} の重み更新量辞書。
        """
        # RPEの計算
        if self.use_rpe:
            effective_reward = reward - self.reward_baseline
        else:
            effective_reward = reward

        # 報酬ベースラインの更新
        self.reward_count += 1
        self.reward_baseline = (
            self.baseline_decay * self.reward_baseline
            + (1.0 - self.baseline_decay) * reward
        )

        # 重み更新量の計算
        updates: Dict[Tuple[int, int], float] = {}
        for key, trace in self._traces.items():
            updates[key] = self.lr * effective_reward * trace
        return updates

    def reset(self) -> None:
        """全内部状態をクリアする。"""
        self._ltp_traces.clear()
        self._ltd_traces.clear()
        self._traces.clear()
        self.reward_baseline = 0.0
        self.reward_count = 0

    def state_dict(self) -> Dict[str, Any]:
        """永続化用の状態辞書を返す。"""
        return {
            "ltp_traces": {f"{k[0]}_{k[1]}": v for k, v in self._ltp_traces.items()},
            "ltd_traces": {f"{k[0]}_{k[1]}": v for k, v in self._ltd_traces.items()},
            "reward_baseline": self.reward_baseline,
            "reward_count": self.reward_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """状態を復元する。"""
        self._ltp_traces.clear()
        for k, v in state.get("ltp_traces", {}).items():
            parts = str(k).split("_")
            if len(parts) == 2:
                self._ltp_traces[(int(parts[0]), int(parts[1]))] = float(v)

        self._ltd_traces.clear()
        for k, v in state.get("ltd_traces", {}).items():
            parts = str(k).split("_")
            if len(parts) == 2:
                self._ltd_traces[(int(parts[0]), int(parts[1]))] = float(v)

        self.reward_baseline = float(state.get("reward_baseline", 0.0))
        self.reward_count = int(state.get("reward_count", 0))

        # 統合ビュー再構築
        self._traces.clear()
        all_keys = set(self._ltp_traces.keys()) | set(self._ltd_traces.keys())
        for key in all_keys:
            net = self._ltp_traces.get(key, 0.0) - \
                self._ltd_traces.get(key, 0.0)
            if abs(net) > 1e-8:
                self._traces[key] = net
