# [配置するディレクトリのパス]: ./src/snn_models/modular_snn.py
# [ファイルの日本語タイトル]: モジュール型・階層的SNNアーキテクチャ
# [ファイルの目的や内容]:
# Network layers and sparse connections are separated so larger modular SNNs can
# be composed without dense matrix operations.

import random
from typing import Dict, List, Tuple

class LIFLayer:
    def __init__(self, n_nodes: int, label: str = ""):
        if n_nodes <= 0:
            raise ValueError("n_nodes must be positive")
        self.n = n_nodes
        self.label = label
        self.v = [0.0] * n_nodes
        self.spikes = [False] * n_nodes
        self.traces = [0.0] * n_nodes
        self.threshold = 1.0
        self.decay = 0.90
        self.trace_decay = 0.85

    def integrate(self, currents: List[float]) -> List[bool]:
        """Integrates sparse currents and updates LIF spikes."""

        for index in range(self.n):
            current = currents[index] if index < len(currents) else 0.0
            self.v[index] = self.v[index] * self.decay + current
            if self.v[index] >= self.threshold:
                self.spikes[index] = True
                self.v[index] = 0.0
                self.traces[index] += 1.0
            else:
                self.spikes[index] = False
            self.traces[index] *= self.trace_decay
        return list(self.spikes)

class STDPConnection:
    def __init__(
        self,
        pre_layer: LIFLayer,
        post_layer: LIFLayer,
        conn_type: str = "all_to_all",
        *,
        seed: int = 11,
        initial_min: float = 0.05,
        initial_max: float = 0.25,
    ):
        self.pre = pre_layer
        self.post = post_layer
        self.rng = random.Random(seed)
        self.initial_min = float(initial_min)
        self.initial_max = float(initial_max)
        self.synapses: List[Tuple[int, int, float]] = []
        self._post_index: Dict[int, List[int]] = {}
        self._initialize_weights(conn_type)

    def _initialize_weights(self, conn_type: str) -> None:
        self.synapses.clear()
        if conn_type == "all_to_all":
            pairs = ((pre, post) for pre in range(self.pre.n) for post in range(self.post.n))
        elif conn_type == "one_to_one":
            pairs = ((index, index) for index in range(min(self.pre.n, self.post.n)))
        elif conn_type == "local":
            radius = max(1, self.pre.n // max(1, self.post.n))
            pairs = []
            for post in range(self.post.n):
                center = int(post * self.pre.n / max(1, self.post.n))
                start = max(0, center - radius)
                end = min(self.pre.n, center + radius + 1)
                pairs.extend((pre, post) for pre in range(start, end))
        else:
            raise ValueError(f"Unsupported connection type: {conn_type}")

        for pre, post in pairs:
            self.synapses.append((pre, post, self.rng.uniform(self.initial_min, self.initial_max)))
        self._rebuild_post_index()

    def propagate(self) -> List[float]:
        """Returns post-layer currents from active pre-layer spikes."""

        currents = [0.0] * self.post.n
        for pre, post, weight in self.synapses:
            if self.pre.spikes[pre]:
                currents[post] += weight
        return currents

    def update_weights(
        self,
        *,
        a_plus: float = 0.01,
        a_minus: float = 0.012,
        w_min: float = 0.0,
        w_max: float = 1.0,
    ) -> None:
        """Applies local STDP only to explicitly stored sparse synapses."""

        updated: List[Tuple[int, int, float]] = []
        for pre, post, weight in self.synapses:
            if self.post.spikes[post]:
                weight += a_plus * self.pre.traces[pre]
            if self.pre.spikes[pre]:
                weight -= a_minus * self.post.traces[post]
            weight = min(w_max, max(w_min, weight))
            updated.append((pre, post, weight))
        self.synapses = updated
        self._rebuild_post_index()

    def _rebuild_post_index(self) -> None:
        index: Dict[int, List[int]] = {}
        for synapse_index, (_pre, post, _weight) in enumerate(self.synapses):
            index.setdefault(post, []).append(synapse_index)
        self._post_index = index
