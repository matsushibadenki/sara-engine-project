# Directory Path: src/sara_engine/utils/stochastic_computing.py
# English Title: Stochastic Computing Utilities
# Purpose/Content: Provides lightweight probability-bitstream style approximations for CPU-first score aggregation without dense matrix operations.

import random
from typing import Dict, Hashable, Optional


def clamp_probability(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


class StochasticAccumulator:
    """
    Lightweight stochastic-computing helper for low-state score approximation.

    The implementation intentionally avoids dense tensor logic and keeps all
    operations in simple scalar loops so it remains compatible with the
    project's CPU-first runtime constraints.
    """

    def __init__(self, bit_count: int = 64, seed: int = 7):
        self.bit_count = max(8, int(bit_count))
        self.seed = int(seed)

    def _rng(self, salt: int) -> random.Random:
        return random.Random(self.seed + (salt * 104729))

    def approximate_probability(self, probability: float, salt: int = 0) -> float:
        probability = clamp_probability(probability)
        rng = self._rng(salt)
        ones = 0
        for _ in range(self.bit_count):
            if rng.random() < probability:
                ones += 1
        return ones / float(self.bit_count)

    def approximate_product(self, left: float, right: float, salt: int = 0) -> float:
        left = clamp_probability(left)
        right = clamp_probability(right)
        left_rng = self._rng((salt * 2) + 1)
        right_rng = self._rng((salt * 2) + 2)
        ones = 0
        for _ in range(self.bit_count):
            if left_rng.random() < left and right_rng.random() < right:
                ones += 1
        return ones / float(self.bit_count)

    def approximate_scores(
        self,
        scores: Dict[Hashable, float],
        confidence_weight: Optional[float] = None,
    ) -> Dict[Hashable, float]:
        if not scores:
            return {}

        max_score = max(float(value) for value in scores.values())
        if max_score <= 0.0:
            return {key: 0.0 for key in scores}

        approximated: Dict[Hashable, float] = {}
        normalized_weight = clamp_probability(confidence_weight) if confidence_weight is not None else None
        for offset, key in enumerate(sorted(scores.keys(), key=lambda item: str(item))):
            normalized_score = clamp_probability(float(scores[key]) / max_score)
            if normalized_weight is None:
                approximated[key] = self.approximate_probability(normalized_score, salt=offset + 1)
            else:
                approximated[key] = self.approximate_product(
                    normalized_score,
                    normalized_weight,
                    salt=offset + 1,
                )
        return approximated

    def argmax(
        self,
        scores: Dict[Hashable, float],
        confidence_weight: Optional[float] = None,
    ) -> Optional[Hashable]:
        approximated = self.approximate_scores(scores, confidence_weight=confidence_weight)
        if not approximated:
            return None
        return max(
            approximated.items(),
            key=lambda item: (item[1], float(scores.get(item[0], 0.0)), str(item[0])),
        )[0]

    def state_units(self) -> int:
        return max(1, self.bit_count // 32)
