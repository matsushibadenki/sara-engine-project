# src/sara_engine/learning/homeostasis.py
# ホメオスタシス（恒常性維持）モジュール
# ニューロンの発火率追跡とシナプスのスケーリングを管理する

from typing import Dict


class NeuronActivityTracker:
    """ニューロンの発火率を追跡する。"""

    def __init__(self, decay: float = 0.99) -> None:
        self._rates: Dict[int, float] = {}
        self._decay = decay

    def step(self) -> None:
        """Apply global decay each simulation step to keep rates bounded and time-local."""
        if not self._rates:
            return
        for neuron_id in list(self._rates.keys()):
            decayed = self._rates[neuron_id] * self._decay
            if decayed < 1e-8:
                del self._rates[neuron_id]
            else:
                self._rates[neuron_id] = decayed

    def update(self, neuron_id: int, fired: bool = True) -> None:
        prev = self._rates.get(neuron_id, 0.0)
        if fired:
            # EMA: keep bounded in [0, 1]
            self._rates[neuron_id] = prev + (1.0 - self._decay)
        else:
            self._rates[neuron_id] = prev

    def get_rate(self, neuron_id: int) -> float:
        return self._rates.get(neuron_id, 0.0)

    def reset(self) -> None:
        self._rates.clear()


class SynapticScalingManager:
    """恒常性シナプススケーリングを管理する。"""

    def __init__(
        self,
        target_rate: float = 0.05,
        scaling_lr: float = 0.01,
        min_scale: float = 0.95,
        max_scale: float = 1.05,
    ) -> None:
        self.target_rate = target_rate
        self.scaling_lr = scaling_lr
        self.min_scale = min_scale
        self.max_scale = max_scale

    def compute_scaling_factor(self, current_rate: float) -> float:
        error = self.target_rate - current_rate
        factor = 1.0 + self.scaling_lr * error
        return max(self.min_scale, min(self.max_scale, factor))
