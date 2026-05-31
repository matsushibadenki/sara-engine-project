# {
#     "//": "ディレクトリパス: src/sara_engine/learning/neural_assembly.py",
#     "//": "ファイルの日本語タイトル: ニューラル・アセンブリ・トラッカー",
#     "//": "ファイルの目的や内容: ネットワーク内で自発的に形成される「ニューロンの機能集団（Assembly）」を監視・分析するためのモジュール。概念の自己組織化プロセスを可視化する。"
# }

from collections import Counter, defaultdict, deque
from itertools import combinations
from typing import Any, Deque, Dict, List, Optional, Set, Tuple


class NeuralAssemblyTracker:
    """Tracks stable sparse co-activation groups without backpropagation or dense matrices."""

    def __init__(
        self,
        window_size: int = 20,
        min_group_size: int = 3,
        activation_threshold: Optional[int] = None,
        max_group_size: int = 8,
        max_candidates_per_step: int = 64,
        max_tracked_assemblies: int = 128,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if min_group_size <= 1:
            raise ValueError("min_group_size must be greater than one")
        if max_group_size < min_group_size:
            raise ValueError("max_group_size must be greater than or equal to min_group_size")
        if max_candidates_per_step <= 0:
            raise ValueError("max_candidates_per_step must be positive")
        if max_tracked_assemblies <= 0:
            raise ValueError("max_tracked_assemblies must be positive")

        self.window_size = int(window_size)
        self.min_group_size = int(min_group_size)
        self.activation_threshold = int(activation_threshold) if activation_threshold is not None else None
        self.max_group_size = int(max_group_size)
        self.max_candidates_per_step = int(max_candidates_per_step)
        self.max_tracked_assemblies = int(max_tracked_assemblies)
        self.spike_history: Deque[Set[int]] = deque(maxlen=self.window_size)
        self.assemblies: Dict[Tuple[int, ...], int] = defaultdict(int)

    def record_step(self, fired_ids: List[int]) -> Dict[str, Any]:
        """Records one sparse firing step and refreshes the bounded assembly map."""

        fired_set = {int(neuron_id) for neuron_id in fired_ids}
        if fired_set:
            self.spike_history.append(fired_set)
            self._recompute_assemblies()
        return self.get_assembly_report(limit=5)

    def get_active_assemblies(self, min_support: Optional[int] = None, limit: Optional[int] = None) -> List[Set[int]]:
        """Returns stable assemblies sorted by support, size, and deterministic neuron order."""

        threshold = self._support_threshold(min_support)
        ranked = self._ranked_assemblies(threshold)
        if limit is not None:
            ranked = ranked[: max(0, int(limit))]
        return [set(group) for group, _support in ranked]

    def get_assembly_report(self, limit: int = 10) -> Dict[str, Any]:
        """Returns an auditable snapshot for release gates and benchmarks."""

        threshold = self._support_threshold(None)
        ranked = self._ranked_assemblies(threshold)[: max(0, int(limit))]
        window_occupancy = len(self.spike_history) / float(max(self.window_size, 1))
        return {
            "window_size": self.window_size,
            "window_count": len(self.spike_history),
            "window_occupancy": round(window_occupancy, 6),
            "min_group_size": self.min_group_size,
            "support_threshold": threshold,
            "candidate_count": len(self.assemblies),
            "active_assembly_count": len(self._ranked_assemblies(threshold)),
            "top_assemblies": [
                {
                    "neurons": list(group),
                    "support": support,
                    "support_ratio": round(support / float(max(len(self.spike_history), 1)), 6),
                }
                for group, support in ranked
            ],
        }

    def _recompute_assemblies(self) -> None:
        counts: Counter[Tuple[int, ...]] = Counter()
        history = list(self.spike_history)

        for fired_set in history:
            for group in self._candidate_groups(fired_set):
                counts[group] += 1

        # Pairwise intersections recover stable concept cores even when each step has extra noisy spikes.
        for left_index, left in enumerate(history):
            for right in history[left_index + 1 :]:
                overlap = left & right
                if len(overlap) >= self.min_group_size:
                    for group in self._candidate_groups(overlap):
                        counts[group] += 1

        ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
        self.assemblies = defaultdict(int, ranked[: self.max_tracked_assemblies])

    def _candidate_groups(self, fired_set: Set[int]) -> List[Tuple[int, ...]]:
        if len(fired_set) < self.min_group_size:
            return []

        ordered = tuple(sorted(fired_set))
        candidates: List[Tuple[int, ...]] = []
        if len(ordered) <= self.max_group_size:
            candidates.append(ordered)

        for group in combinations(ordered, self.min_group_size):
            candidates.append(group)
            if len(candidates) >= self.max_candidates_per_step:
                break
        return candidates

    def _support_threshold(self, override: Optional[int]) -> int:
        if override is not None:
            return max(1, int(override))
        if self.activation_threshold is not None:
            return max(1, self.activation_threshold)
        return max(2, min(4, max(1, len(self.spike_history)) // 3))

    def _ranked_assemblies(self, min_support: int) -> List[Tuple[Tuple[int, ...], int]]:
        return sorted(
            ((group, support) for group, support in self.assemblies.items() if support >= min_support),
            key=lambda item: (-item[1], -len(item[0]), item[0]),
        )
