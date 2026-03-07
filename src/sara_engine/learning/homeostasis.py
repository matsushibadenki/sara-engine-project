# src/sara_engine/learning/homeostasis.py
# ホメオスタシス（恒常性維持）モジュール
# ニューロンの発火率追跡とシナプスのスケーリングを管理する

from typing import Dict, Iterable, Any
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


class AdaptiveThresholdHomeostasis:
    """Per-unit homeostatic controller for threshold/fatigue style regulation."""

    def __init__(
        self,
        target_rate: float = 0.1,
        adaptation_rate: float = 0.05,
        decay: float = 0.95,
        min_threshold: float = 0.0,
        max_threshold: float = 1.0,
        global_weight: float = 0.0,
    ) -> None:
        self.target_rate = target_rate
        self.adaptation_rate = adaptation_rate
        self.decay = decay
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.global_weight = global_weight
        self.thresholds: Dict[int, float] = {}
        self.rates: Dict[int, float] = {}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "thresholds": self.thresholds,
            "rates": self.rates,
            "target_rate": self.target_rate,
            "adaptation_rate": self.adaptation_rate,
            "decay": self.decay,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "global_weight": self.global_weight,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.thresholds = {int(k): float(v) for k, v in state.get("thresholds", {}).items()}
        self.rates = {int(k): float(v) for k, v in state.get("rates", {}).items()}
        self.target_rate = float(state.get("target_rate", self.target_rate))
        self.adaptation_rate = float(state.get("adaptation_rate", self.adaptation_rate))
        self.decay = float(state.get("decay", self.decay))
        self.min_threshold = float(state.get("min_threshold", self.min_threshold))
        self.max_threshold = float(state.get("max_threshold", self.max_threshold))
        self.global_weight = float(state.get("global_weight", self.global_weight))

    def reset(self) -> None:
        self.thresholds.clear()
        self.rates.clear()

    def update(self, active_ids: Iterable[int], population_size: int | None = None) -> None:
        active_set = set(active_ids)
        tracked_ids = set(self.thresholds.keys()) | set(self.rates.keys()) | active_set
        if not tracked_ids:
            return

        global_error = 0.0
        if population_size and population_size > 0:
            global_rate = len(active_set) / float(population_size)
            global_error = global_rate - self.target_rate

        for unit_id in tracked_ids:
            prev_rate = self.rates.get(unit_id, 0.0)
            fired = 1.0 if unit_id in active_set else 0.0
            rate = prev_rate * self.decay + fired * (1.0 - self.decay)
            self.rates[unit_id] = rate

            local_error = rate - self.target_rate
            error = ((1.0 - self.global_weight) * local_error) + (self.global_weight * global_error)
            threshold = self.thresholds.get(unit_id, self.min_threshold)
            threshold += self.adaptation_rate * error
            threshold = max(self.min_threshold, min(self.max_threshold, threshold))

            if threshold <= self.min_threshold + 1e-6 and rate < self.target_rate * 0.25:
                self.thresholds.pop(unit_id, None)
                if rate < 1e-6:
                    self.rates.pop(unit_id, None)
            else:
                self.thresholds[unit_id] = threshold

    def get_threshold(self, unit_id: int, default: float = 0.0) -> float:
        return self.thresholds.get(unit_id, default)

    def modulate(self, unit_id: int, score: float, strength: float = 1.0) -> float:
        threshold = self.thresholds.get(unit_id, 0.0)
        if threshold <= 0.0:
            return score
        return score / (1.0 + threshold * strength)
