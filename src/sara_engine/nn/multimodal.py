# ディレクトリパス: src/sara_engine/nn/multimodal.py
# ファイルの日本語タイトル: マルチモーダル連合学習層
# ファイルの目的や内容: 異なる感覚（テキスト、画像等）からのスパイクを同期させ、STDPによって相関を学習することで、クロスモーダルな想起を可能にするモジュール。忘却(Decay)と刈り込み(Pruning)による長期安定稼働をサポート。
from typing import List, Dict, Optional
from .module import SNNModule

class CrossModalAssociator(SNNModule):
    """
    Associative memory layer that connects spike sets from different input sources.
    When a word like "apple" and an image of a "red circle" are presented simultaneously,
    the connection between their spike patterns is strengthened by STDP (Hebbian learning).
    Synapses are represented by sparse dictionaries without using matrix operations.
    """
    def __init__(self, dim_a: int, dim_b: int, density: float = 0.3, decay_rate: float = 0.99, learning_rate: float = 0.2):
        super().__init__()
        self.dim_a = dim_a
        self.dim_b = dim_b
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        
        # Connections from A to B, and B to A (bidirectional association)
        # Structure: {source_spike: {target_spike: weight}}
        self.weights_a2b: Dict[int, Dict[int, float]] = {}
        self.weights_b2a: Dict[int, Dict[int, float]] = {}
        
        self.register_state("weights_a2b")
        self.register_state("weights_b2a")

    def forward(self, spikes_a: Optional[List[int]] = None, spikes_b: Optional[List[int]] = None, learning: bool = False, threshold: float = 0.5) -> Dict[str, List[int]]:
        """
        Recall one input from the other, or learn an association if both inputs are present.
        """
        spikes_a_list: List[int] = spikes_a or []
        spikes_b_list: List[int] = spikes_b or []
        
        recall_b: List[int] = []
        recall_a: List[int] = []

        # Recall B from A
        if spikes_a_list:
            potentials_b: Dict[int, float] = {}
            for s in spikes_a_list:
                if s in self.weights_a2b:
                    for target, w in self.weights_a2b[s].items():
                        potentials_b[target] = potentials_b.get(target, 0.0) + w
            recall_b = [k for k, v in sorted(potentials_b.items(), key=lambda x: x[1], reverse=True) if v > threshold]

        # Recall A from B
        if spikes_b_list:
            potentials_a: Dict[int, float] = {}
            for s in spikes_b_list:
                if s in self.weights_b2a:
                    for target, w in self.weights_b2a[s].items():
                        potentials_a[target] = potentials_a.get(target, 0.0) + w
            recall_a = [k for k, v in sorted(potentials_a.items(), key=lambda x: x[1], reverse=True) if v > threshold]

        # Learning (STDP & Homeostasis)
        if learning:
            # Memory decay and pruning
            self._decay_and_prune(self.weights_a2b)
            self._decay_and_prune(self.weights_b2a)
            
            # Strengthen co-occurring patterns
            if spikes_a_list and spikes_b_list:
                self._update_associative_weights(spikes_a_list, spikes_b_list)

        return {"recall_a": recall_a, "recall_b": recall_b}

    def _update_associative_weights(self, spikes_a: List[int], spikes_b: List[int]) -> None:
        """Strengthen connections between spikes that fire simultaneously (LTP)"""
        # Strengthen A -> B connections
        for a in spikes_a:
            if a not in self.weights_a2b: 
                self.weights_a2b[a] = {}
            for b in spikes_b:
                current_w = self.weights_a2b[a].get(b, 0.0)
                self.weights_a2b[a][b] = min(3.0, current_w + self.learning_rate)
        
        # Strengthen B -> A connections
        for b in spikes_b:
            if b not in self.weights_b2a: 
                self.weights_b2a[b] = {}
            for a in spikes_a:
                current_w = self.weights_b2a[b].get(a, 0.0)
                self.weights_b2a[b][a] = min(3.0, current_w + self.learning_rate)

    def _decay_and_prune(self, weights: Dict[int, Dict[int, float]]) -> None:
        """Forget unused synapses and remove minute synapses"""
        empty_sources = []
        for src, targets in weights.items():
            to_remove = []
            for tgt in targets:
                targets[tgt] *= self.decay_rate
                if targets[tgt] < 0.05:  # Pruning threshold
                    to_remove.append(tgt)
            for tgt in to_remove:
                del targets[tgt]
            if not targets:
                empty_sources.append(src)
        for src in empty_sources:
            del weights[src]