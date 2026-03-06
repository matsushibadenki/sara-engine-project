# src/sara_engine/learning/structural_plasticity.py
# 構造的可塑性モジュール
# シナプスの生成・刈り込みを管理する

from typing import Dict, Tuple


class StructuralPlasticityManager:
    """構造的可塑性（シナプスの生成・刈り込み）を管理する。"""

    def __init__(self, prune_threshold: float = 0.1, growth_rate: float = 0.01) -> None:
        self.prune_threshold = prune_threshold
        self.growth_rate = growth_rate

    def should_prune(self, weight: float) -> bool:
        return abs(weight) < self.prune_threshold

    def prune_synapses(
        self,
        synapses: list,
        max_size: int,
        target_size: int,
    ) -> None:
        """弱いシナプスを刈り込む。"""
        for s_dict in synapses:
            if len(s_dict) > max_size:
                sorted_items = sorted(s_dict.items(), key=lambda x: abs(
                    x[1][0]) if isinstance(x[1], tuple) else abs(x[1]))
                to_remove = len(s_dict) - target_size
                for key, _ in sorted_items[:to_remove]:
                    del s_dict[key]
