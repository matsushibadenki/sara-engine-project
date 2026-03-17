# Directory Path: src/sara_engine/learning/homeostasis.py
# English Title: Homeostatic Plasticity with BCM Sliding Threshold
# Purpose/Content: ニューロンの発火率追跡とシナプススケーリングを管理。完全な後方互換性を維持しながら、ステップループのO(1)遅延評価による超省エネ化と、BCM理論に基づく動的目標発火率によるメタ可塑性を実現。

from typing import Dict, Iterable, Any
import math

class NeuronActivityTracker:
    """遅延評価(Lazy Evaluation)による O(1) 発火率トラッカー。後方互換性維持版。"""

    def __init__(
        self,
        decay: float = 0.99,
        slow_decay: float = 0.999,
        blend_fast: float = 0.7,
    ) -> None:
        self._fast_rates: Dict[int, float] = {}
        self._slow_rates: Dict[int, float] = {}
        self._last_update_step: Dict[int, int] = {}
        
        # オリジナルの変数名を維持
        self._decay = decay
        self._slow_decay = slow_decay
        self._blend_fast = blend_fast
        self._current_step = 0

    def step(self) -> None:
        """O(1) のグローバルタイムカウンターのインクリメントのみ行う。"""
        self._current_step += 1

    def _apply_lazy_decay(self, neuron_id: int) -> None:
        """アクセス時に、経過したステップ数に応じて一括で厳密な減衰を適用する"""
        last_step = self._last_update_step.get(neuron_id, self._current_step)
        dt = self._current_step - last_step
        
        if dt > 0:
            if neuron_id in self._fast_rates:
                # べき乗計算によりオリジナルのループと数学的に完全に一致するO(1)減衰
                self._fast_rates[neuron_id] *= (self._decay ** dt)
                if self._fast_rates[neuron_id] < 1e-10:
                    del self._fast_rates[neuron_id]
                    
            if neuron_id in self._slow_rates:
                self._slow_rates[neuron_id] *= (self._slow_decay ** dt)
                if self._slow_rates[neuron_id] < 1e-10:
                    del self._slow_rates[neuron_id]
                    
            self._last_update_step[neuron_id] = self._current_step

    def update(self, neuron_id: int, fired: bool = True) -> None:
        self._apply_lazy_decay(neuron_id)
        prev_fast = self._fast_rates.get(neuron_id, 0.0)
        prev_slow = self._slow_rates.get(neuron_id, 0.0)
        
        if fired:
            self._fast_rates[neuron_id] = prev_fast + (1.0 - self._decay)
            self._slow_rates[neuron_id] = prev_slow + (1.0 - self._slow_decay)
        else:
            self._fast_rates[neuron_id] = prev_fast
            self._slow_rates[neuron_id] = prev_slow
            
        self._last_update_step[neuron_id] = self._current_step

    def get_rate(self, neuron_id: int) -> float:
        self._apply_lazy_decay(neuron_id)
        fast = self._fast_rates.get(neuron_id, 0.0)
        slow = self._slow_rates.get(neuron_id, 0.0)
        blended = (self._blend_fast * fast) + ((1.0 - self._blend_fast) * slow)
        return max(0.0, min(1.0, blended))

    def get_global_rate(self) -> float:
        keys = set(self._fast_rates.keys()) | set(self._slow_rates.keys())
        if not keys:
            return 0.0
        total = sum(self.get_rate(nid) for nid in keys)
        return total / float(len(keys))

    def reset(self) -> None:
        self._fast_rates.clear()
        self._slow_rates.clear()
        self._last_update_step.clear()
        self._current_step = 0


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
    """BCM理論に基づくスライディング目標発火率を用いた適応的閾値コントローラ"""

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
        self.dynamic_targets: Dict[int, float] = {} # BCMスライディング用

    def state_dict(self) -> Dict[str, Any]:
        return {
            "thresholds": self.thresholds,
            "rates": self.rates,
            "dynamic_targets": self.dynamic_targets,
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
        self.dynamic_targets = {int(k): float(v) for k, v in state.get("dynamic_targets", {}).items()}
        self.target_rate = float(state.get("target_rate", self.target_rate))
        self.adaptation_rate = float(state.get("adaptation_rate", self.adaptation_rate))
        self.decay = float(state.get("decay", self.decay))
        self.min_threshold = float(state.get("min_threshold", self.min_threshold))
        self.max_threshold = float(state.get("max_threshold", self.max_threshold))
        self.global_weight = float(state.get("global_weight", self.global_weight))

    def reset(self) -> None:
        self.thresholds.clear()
        self.rates.clear()
        self.dynamic_targets.clear()

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

            # BCM スライディング閾値 (目標発火率を自己組織化)
            current_target = self.dynamic_targets.get(unit_id, self.target_rate)
            new_target = current_target * 0.99 + (rate ** 2) * 0.01
            new_target = max(self.target_rate * 0.5, min(self.target_rate * 2.0, new_target))
            self.dynamic_targets[unit_id] = new_target

            local_error = rate - new_target
            error = ((1.0 - self.global_weight) * local_error) + (self.global_weight * global_error)
            
            threshold = self.thresholds.get(unit_id, self.min_threshold)
            threshold += self.adaptation_rate * error
            threshold = max(self.min_threshold, min(self.max_threshold, threshold))

            if threshold <= self.min_threshold + 1e-6 and rate < new_target * 0.25:
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