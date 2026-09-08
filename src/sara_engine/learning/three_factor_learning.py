# src/sara_engine/learning/three_factor_learning.py
# 三要素学習モジュール
# プレ/ポストシナプス活動 + 報酬信号の3要素による学習規則を実装する。
# 報酬ベースライン追跡、LTP/LTD トレースの分離、state_dict 対応を含む。

import math
from typing import Any, Dict, Optional, Tuple


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
        max_traces: int = 256,
        max_trace_age: float = 64.0,
        trace_floor: float = 1e-6,
    ) -> None:
        """
        Args:
            lr: 学習率。
            trace_decay: トレースの時間減衰率 (0〜1)。
            baseline_decay: 報酬ベースラインの指数移動平均減衰率。
            use_rpe: True の場合、報酬予測誤差 (RPE) を使用して学習する。
        """
        if not 0.0 <= float(trace_decay) <= 1.0:
            raise ValueError("trace_decay must be between 0.0 and 1.0")
        if not 0.0 <= float(baseline_decay) <= 1.0:
            raise ValueError("baseline_decay must be between 0.0 and 1.0")
        if int(max_traces) < 1:
            raise ValueError("max_traces must be positive")
        if not math.isfinite(float(max_trace_age)) or float(max_trace_age) <= 0.0:
            raise ValueError("max_trace_age must be finite and positive")
        if not math.isfinite(float(trace_floor)) or float(trace_floor) < 0.0:
            raise ValueError("trace_floor must be finite and non-negative")
        self.lr = float(lr)
        self.trace_decay = float(trace_decay)
        self.baseline_decay = float(baseline_decay)
        self.use_rpe = use_rpe
        self.max_traces = int(max_traces)
        self.max_trace_age = float(max_trace_age)
        self.trace_floor = float(trace_floor)

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
        self._last_seen: Dict[Tuple[int, int], float] = {}
        self._current_time: Optional[float] = None
        self.eviction_count: int = 0
        self.last_update_event_cost: int = 0
        self.last_reward_event_cost: int = 0

    @property
    def trace_count(self) -> int:
        return len(self._last_seen)

    def _rebuild_combined(self) -> None:
        self._traces.clear()
        all_keys = set(self._ltp_traces) | set(self._ltd_traces)
        for key in all_keys:
            net = self._ltp_traces.get(key, 0.0) - self._ltd_traces.get(key, 0.0)
            if abs(net) > self.trace_floor:
                self._traces[key] = net
        for key in list(self._last_seen):
            if key not in self._ltp_traces and key not in self._ltd_traces:
                self._last_seen.pop(key, None)

    def _remove_trace(self, key: Tuple[int, int]) -> None:
        self._ltp_traces.pop(key, None)
        self._ltd_traces.pop(key, None)
        self._traces.pop(key, None)
        self._last_seen.pop(key, None)

    def _advance_time(self, time: float) -> None:
        current_time = float(time)
        if not math.isfinite(current_time):
            raise ValueError("time must be finite")
        if self._current_time is None:
            self._current_time = current_time
            return
        if current_time < self._current_time:
            raise ValueError("time must be monotonic")
        elapsed = current_time - self._current_time
        if elapsed > 0.0:
            decay_factor = self.trace_decay ** elapsed
            for key in list(self._ltp_traces):
                self._ltp_traces[key] *= decay_factor
                if self._ltp_traces[key] <= self.trace_floor:
                    del self._ltp_traces[key]
            for key in list(self._ltd_traces):
                self._ltd_traces[key] *= decay_factor
                if self._ltd_traces[key] <= self.trace_floor:
                    del self._ltd_traces[key]
        self._current_time = current_time
        for key, last_seen in list(self._last_seen.items()):
            if current_time - last_seen > self.max_trace_age:
                self._remove_trace(key)
        self._rebuild_combined()

    def _admit_trace(self, key: Tuple[int, int]) -> None:
        if key in self._traces or key in self._ltp_traces or key in self._ltd_traces:
            return
        if len(self._last_seen) < self.max_traces:
            return
        victim = min(
            self._last_seen,
            key=lambda candidate: (
                abs(self._traces.get(candidate, 0.0)),
                self._last_seen[candidate],
                candidate,
            ),
        )
        self._remove_trace(victim)
        self.eviction_count += 1

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
            time: Monotonic event time used for decay and expiry.
        """
        strength = float(strength)
        if not math.isfinite(strength):
            raise ValueError("strength must be finite")
        self._advance_time(time)
        if strength == 0.0:
            self.last_update_event_cost = len(self._last_seen)
            return
        key = (pre_id, post_id)
        self._admit_trace(key)

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
        self._last_seen[key] = float(time)
        self.last_update_event_cost = min(self.max_traces, len(self._last_seen)) + 1

    def decay_all_traces(self, steps: float = 1.0) -> None:
        """全トレースに時間減衰を適用する。"""
        steps = float(steps)
        if not math.isfinite(steps) or steps < 0.0:
            raise ValueError("steps must be finite and non-negative")
        start = self._current_time if self._current_time is not None else 0.0
        self._advance_time(start + steps)

    def apply_reward(
        self,
        reward: float,
        *,
        time: Optional[float] = None,
    ) -> Dict[Tuple[int, int], float]:
        """報酬シグナルに基づく重み更新量を計算する。

        use_rpe=True の場合、報酬予測誤差 (RPE) を使用:
            δ = reward - baseline
            Δw = lr × trace × δ

        Args:
            reward: 環境からの報酬値。

        Returns:
            {(pre_id, post_id): delta_w} の重み更新量辞書。
        """
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        if time is not None:
            self._advance_time(time)

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
        self.last_reward_event_cost = len(self._traces)
        return updates

    def reset(self) -> None:
        """全内部状態をクリアする。"""
        self._ltp_traces.clear()
        self._ltd_traces.clear()
        self._traces.clear()
        self._last_seen.clear()
        self._current_time = None
        self.reward_baseline = 0.0
        self.reward_count = 0
        self.eviction_count = 0
        self.last_update_event_cost = 0
        self.last_reward_event_cost = 0

    def state_dict(self) -> Dict[str, Any]:
        """永続化用の状態辞書を返す。"""
        return {
            "ltp_traces": {f"{k[0]}_{k[1]}": v for k, v in self._ltp_traces.items()},
            "ltd_traces": {f"{k[0]}_{k[1]}": v for k, v in self._ltd_traces.items()},
            "reward_baseline": self.reward_baseline,
            "reward_count": self.reward_count,
            "last_seen": {f"{k[0]}_{k[1]}": v for k, v in self._last_seen.items()},
            "current_time": self._current_time,
            "max_traces": self.max_traces,
            "max_trace_age": self.max_trace_age,
            "trace_floor": self.trace_floor,
            "eviction_count": self.eviction_count,
            "last_update_event_cost": self.last_update_event_cost,
            "last_reward_event_cost": self.last_reward_event_cost,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """状態を復元する。"""
        admitted_keys: set[Tuple[int, int]] = set()
        self._ltp_traces.clear()
        for k, v in state.get("ltp_traces", {}).items():
            parts = str(k).split("_")
            if len(parts) == 2:
                key = (int(parts[0]), int(parts[1]))
                if key not in admitted_keys and len(admitted_keys) >= self.max_traces:
                    continue
                admitted_keys.add(key)
                self._ltp_traces[key] = float(v)

        self._ltd_traces.clear()
        for k, v in state.get("ltd_traces", {}).items():
            parts = str(k).split("_")
            if len(parts) == 2:
                key = (int(parts[0]), int(parts[1]))
                if key not in admitted_keys and len(admitted_keys) >= self.max_traces:
                    continue
                admitted_keys.add(key)
                self._ltd_traces[key] = float(v)

        self.reward_baseline = float(state.get("reward_baseline", 0.0))
        self.reward_count = int(state.get("reward_count", 0))
        raw_current_time = state.get("current_time")
        self._current_time = float(raw_current_time) if raw_current_time is not None else None
        self._last_seen.clear()
        for k, v in state.get("last_seen", {}).items():
            parts = str(k).split("_")
            if len(parts) == 2:
                key = (int(parts[0]), int(parts[1]))
                if key in admitted_keys:
                    self._last_seen[key] = float(v)
        default_seen = self._current_time if self._current_time is not None else 0.0
        for key in set(self._ltp_traces) | set(self._ltd_traces):
            self._last_seen.setdefault(key, default_seen)
        self.eviction_count = int(state.get("eviction_count", 0))
        self.last_update_event_cost = int(state.get("last_update_event_cost", 0))
        self.last_reward_event_cost = int(state.get("last_reward_event_cost", 0))

        # 統合ビュー再構築
        self._rebuild_combined()
