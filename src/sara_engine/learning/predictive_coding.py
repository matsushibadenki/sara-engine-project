# src/sara_engine/learning/predictive_coding.py
# 予測符号化モジュール
# 予測誤差に基づく学習規則を実装する

from typing import Dict


class PredictiveCodingManager:
    """予測符号化に基づく学習を管理する。"""

    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate
        self._predictions: Dict[int, float] = {}

    def predict(self, neuron_id: int) -> float:
        return self._predictions.get(neuron_id, 0.0)

    def update(self, neuron_id: int, actual: float) -> float:
        predicted = self._predictions.get(neuron_id, 0.0)
        error = actual - predicted
        self._predictions[neuron_id] = predicted + self.learning_rate * error
        return error

    def reset(self) -> None:
        self._predictions.clear()
