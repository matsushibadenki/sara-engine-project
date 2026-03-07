from ..nn.attention import SDRFuzzyAttention
from ..nn.module import SNNModule
from typing import Dict, List, Tuple
# ディレクトリパス: src/sara_engine/models/spiking_multimodal_hub.py
# ファイルの日本語タイトル: マルチモーダルSNN連想ハブ
# ファイルの目的や内容: 複数のモダリティ（視覚、テキスト、音声など）からのスパイクを同一の連想記憶空間で結びつける。Fuzzy Attentionを統合し、純粋なスパイクの共起（Hebbian Learning）によるクロスモーダル検索を実現。
class SpikingMultimodalHub(SNNModule):
    """
    Phase 2: Multi-source Integration Hub
    A model that integrates and associates different sensory inputs such as text, image, and audio in a shared SNN space.
    It supports recall from ambiguous inputs (Fuzzy Recall) using an attention mechanism.
    """

    def __init__(self, modalities: List[str], shared_space_size: int = 4096, learning_rate: float = 0.3, decay_rate: float = 0.98, max_weight: float = 3.0):
        super().__init__()
        self.modalities = modalities
        self.shared_space_size = shared_space_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.max_weight = max_weight

        # Fuzzy Attention layer for smoothing and associating context in the shared space
        self.attention = SDRFuzzyAttention(
            sdr_size=shared_space_size, threshold=0.2)

        # Dictionary representing sparse connections. No matrices are used.
        # Bidirectional connections between each modality and the shared SNN space
        # Structure: {modality: {"to_shared": {src: {tgt: w}}, "from_shared": {src: {tgt: w}}}}
        self.hub_synapses: Dict[str,
                                Dict[str, Dict[int, Dict[int, float]]]] = {}

        for mod in modalities:
            self.hub_synapses[mod] = {
                "to_shared": {},
                "from_shared": {}
            }

        self.register_state("hub_synapses")

    def forward(self, inputs: Dict[str, List[int]], learning: bool = True) -> Tuple[List[int], Dict[str, List[int]]]:
        """
        Integrates inputs from each modality in the shared space and infers (recalls) information for missing modalities.
        """
        # 1. Integrate bottom-up stimuli from each modality to the shared space
        global_potentials: Dict[int, float] = {}
        for mod, spikes in inputs.items():
            if mod not in self.modalities:
                continue
            weights = self.hub_synapses[mod]["to_shared"]
            for s in spikes:
                if s in weights:
                    for t, w in weights[s].items():
                        global_potentials[t] = global_potentials.get(
                            t, 0.0) + w
                else:
                    # Project unknown spikes randomly to the shared space (Random Indexing approach)
                    target_neuron = (s * hash(mod)) % self.shared_space_size
                    global_potentials[target_neuron] = global_potentials.get(
                        target_neuron, 0.0) + 1.0

        # 2. Spike firing in the shared space (K-Winner-Take-All)
        active_k = max(1, int(self.shared_space_size * 0.05))  # 5% Sparsity
        sorted_global = sorted(global_potentials.items(),
                               key=lambda x: x[1], reverse=True)
        global_spikes = [idx for idx,
                         pot in sorted_global[:active_k] if pot > 0.5]

        # 3. Context refinement and association by Fuzzy Attention
        global_spikes = self.attention.forward(
            query=global_spikes, key=global_spikes, value=global_spikes)

        # 4. Top-down prediction (recall) from the shared space to each modality
        predictions: Dict[str, List[int]] = {
            mod: [] for mod in self.modalities}
        for mod in self.modalities:
            pred_pots: Dict[int, float] = {}
            weights = self.hub_synapses[mod]["from_shared"]
            for gs in global_spikes:
                if gs in weights:
                    for tgt, w in weights[gs].items():
                        pred_pots[tgt] = pred_pots.get(tgt, 0.0) + w
            predictions[mod] = [idx for idx, p in sorted(
                pred_pots.items(), key=lambda x: x[1], reverse=True)[:50] if p > 0.8]

        # 5. Learning (Hebbian Learning & Pruning)
        if learning and inputs:
            self._learn_and_prune(inputs, global_spikes)

        return global_spikes, predictions

    def _learn_and_prune(self, inputs: Dict[str, List[int]], global_spikes: List[int]) -> None:
        """
        Updates bidirectional connections between the shared space and each modality, and prunes unnecessary synapses.
        """
        for mod in self.modalities:
            to_shared = self.hub_synapses[mod]["to_shared"]
            from_shared = self.hub_synapses[mod]["from_shared"]

            # Decay and prune all synapses
            self._decay_synapses(to_shared)
            self._decay_synapses(from_shared)

            if mod not in inputs:
                continue

            mod_spikes = inputs[mod]

            # LTP: Bottom-up (Modality -> Shared)
            for ms in mod_spikes:
                if ms not in to_shared:
                    to_shared[ms] = {}
                for gs in global_spikes:
                    to_shared[ms][gs] = min(
                        self.max_weight, to_shared[ms].get(gs, 0.0) + self.learning_rate)

            # LTP: Top-down (Shared -> Modality)
            for gs in global_spikes:
                if gs not in from_shared:
                    from_shared[gs] = {}
                for ms in mod_spikes:
                    from_shared[gs][ms] = min(
                        self.max_weight, from_shared[gs].get(ms, 0.0) + self.learning_rate)

    def _decay_synapses(self, weights: Dict[int, Dict[int, float]]) -> None:
        """Synapse decay and Pruning (Energy saving / memory optimization)"""
        empty_sources = []
        for src, targets in weights.items():
            to_remove = []
            for tgt in targets:
                targets[tgt] *= self.decay_rate
                if targets[tgt] < 0.05:
                    to_remove.append(tgt)
            for tgt in to_remove:
                del targets[tgt]
            if not targets:
                empty_sources.append(src)
        for src in empty_sources:
            del weights[src]

    def reset_state(self) -> None:
        super().reset_state()
        self.attention.reset_state()
