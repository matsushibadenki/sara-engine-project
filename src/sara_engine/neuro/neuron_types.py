# src/sara_engine/neuro/neuron_types.py
# ニューロンタイプ管理モジュール
# 興奮性/抑制性ニューロンの分類を管理する

import random
from typing import Dict


class NeuronTypeManager:
    """ニューロンのタイプ（興奮性/抑制性）を管理する。"""

    def __init__(self, inhibitory_ratio: float = 0.2) -> None:
        self.inhibitory_ratio = inhibitory_ratio
        self._types: Dict[int, str] = {}

    def get_type(self, neuron_id: int) -> str:
        if neuron_id not in self._types:
            self._types[neuron_id] = (
                "inhibitory" if random.random() < self.inhibitory_ratio else "excitatory"
            )
        return self._types[neuron_id]

    def is_inhibitory(self, neuron_id: int) -> bool:
        return self.get_type(neuron_id) == "inhibitory"

    def get_sign(self, neuron_id: int) -> float:
        return -1.0 if self.is_inhibitory(neuron_id) else 1.0
