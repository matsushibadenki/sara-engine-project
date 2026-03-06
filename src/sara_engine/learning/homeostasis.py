# src/sara_engine/learning/homeostasis.py
# ホメオスタシス（恒常性維持）モジュール
# ニューロンの発火率追跡とシナプスのスケーリングを管理する

from typing import Dict


class NeuronActivityTracker:
    """ニューロンの発火率を追跡する。"""

    def __init__(self, decay: float = 0.99) -> None:
        self._rates: Dict[int, float] = {}
        self._decay = decay

    def update(self, neuron_id: int, fired: bool = True) -> None:
        prev = self._rates.get(neuron_id, 0.0)
        self._rates[neuron_id] = prev * self._decay + (1.0 if fired else 0.0)

    def get_rate(self, neuron_id: int) -> float:
        return self._rates.get(neuron_id, 0.0)

    def reset(self) -> None:
        self._rates.clear()


class SynapticScalingManager:
    """恒常性シナプススケーリングを管理する。"""

    def __init__(self, target_rate: float = 0.05, scaling_lr: float = 0.01) -> None:
        self.target_rate = target_rate
        self.scaling_lr = scaling_lr

    def compute_scaling_factor(self, current_rate: float) -> float:
        error = self.target_rate - current_rate
        return 1.0 + self.scaling_lr * error
