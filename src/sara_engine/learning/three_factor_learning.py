# src/sara_engine/learning/three_factor_learning.py
# 三要素学習モジュール
# プレ/ポストシナプス活動 + 報酬信号の3要素による学習規則を実装する

from typing import Dict, Tuple


class ThreeFactorLearningManager:
    """三要素学習則（Pre-Post-Reward）を管理する。"""

    def __init__(self, lr: float = 0.01, trace_decay: float = 0.95) -> None:
        self.lr = lr
        self.trace_decay = trace_decay
        self._traces: Dict[Tuple[int, int], float] = {}

    def update_trace(self, pre_id: int, post_id: int, strength: float, time: float) -> None:
        key = (pre_id, post_id)
        prev = self._traces.get(key, 0.0)
        self._traces[key] = prev * self.trace_decay + strength

    def apply_reward(self, reward: float) -> Dict[Tuple[int, int], float]:
        updates: Dict[Tuple[int, int], float] = {}
        for key, trace in self._traces.items():
            updates[key] = self.lr * reward * trace
        return updates

    def reset(self) -> None:
        self._traces.clear()
