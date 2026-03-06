# src/sara_engine/learning/homeostasis.py
# ホメオスタシス（恒常性維持）モジュール
# ニューロンの発火率追跡とシナプスのスケーリングを管理する

from typing import Dict
import math


class NeuronActivityTracker:
    """ニューロンの発火率を追跡する。"""

    def __init__(
        self,
        decay: float = 0.99,
        slow_decay: float = 0.999,
        blend_fast: float = 0.7,
    ) -> None:
        self._fast_rates: Dict[int, float] = {}
        self._slow_rates: Dict[int, float] = {}
        self._decay = decay
        self._slow_decay = slow_decay
        self._blend_fast = blend_fast

    def step(self) -> None:
        """Apply global decay each simulation step to keep rates bounded and time-local."""
        if not self._fast_rates and not self._slow_rates:
            return
        for neuron_id in list(self._fast_rates.keys()):
            decayed = self._fast_rates[neuron_id] * self._decay
            if decayed < 1e-10:
                del self._fast_rates[neuron_id]
            else:
                self._fast_rates[neuron_id] = decayed
        for neuron_id in list(self._slow_rates.keys()):
            decayed = self._slow_rates[neuron_id] * self._slow_decay
            if decayed < 1e-10:
                del self._slow_rates[neuron_id]
            else:
                self._slow_rates[neuron_id] = decayed

    def update(self, neuron_id: int, fired: bool = True) -> None:
        prev_fast = self._fast_rates.get(neuron_id, 0.0)
        prev_slow = self._slow_rates.get(neuron_id, 0.0)
        if fired:
            # EMA: keep bounded in [0, 1]
            self._fast_rates[neuron_id] = prev_fast + (1.0 - self._decay)
            self._slow_rates[neuron_id] = prev_slow + (1.0 - self._slow_decay)
            return
        self._fast_rates[neuron_id] = prev_fast
        self._slow_rates[neuron_id] = prev_slow

    def get_rate(self, neuron_id: int) -> float:
        fast = self._fast_rates.get(neuron_id, 0.0)
        slow = self._slow_rates.get(neuron_id, 0.0)
        blended = (self._blend_fast * fast) + ((1.0 - self._blend_fast) * slow)
        return max(0.0, min(1.0, blended))

    def get_global_rate(self) -> float:
        """Return population average over currently active tracked neurons."""
        keys = set(self._fast_rates.keys()) | set(self._slow_rates.keys())
        if not keys:
            return 0.0
        total = 0.0
        for neuron_id in keys:
            total += self.get_rate(neuron_id)
        return total / float(len(keys))

    def reset(self) -> None:
        self._fast_rates.clear()
        self._slow_rates.clear()


class SynapticScalingManager:
    """恒常性シナプススケーリングを管理する。"""

    def __init__(
        self,
        target_rate: float = 0.05,
        scaling_lr: float = 0.01,
        min_scale: float = 0.95,
        max_scale: float = 1.05,
        deadband: float = 0.05,
        global_weight: float = 0.3,
    ) -> None:
        self.target_rate = target_rate
        self.scaling_lr = scaling_lr
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.deadband = deadband
        self.global_weight = global_weight

    def compute_scaling_factor(self, current_rate: float, population_rate: float | None = None) -> float:
        target = max(1e-6, self.target_rate)
        local_rel_error = (target - current_rate) / target
        global_rel_error = 0.0
        if population_rate is not None:
            global_rel_error = (target - population_rate) / target
        rel_error = ((1.0 - self.global_weight) * local_rel_error) + (self.global_weight * global_rel_error)

        if abs(rel_error) <= self.deadband:
            return 1.0

        smooth_error = math.tanh(rel_error)
        factor = 1.0 + self.scaling_lr * smooth_error
        return max(self.min_scale, min(self.max_scale, factor))
