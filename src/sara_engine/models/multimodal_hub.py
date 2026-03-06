"""
{
    "//": "ディレクトリパス: src/sara_engine/models/multimodal_hub.py",
    "//": "ファイルの日本語タイトル: マルチソース統合SNNハブ",
    "//": "ファイルの目的や内容: 視覚、言語、その他のモダリティからのSDRを同一空間で統合し、相関を学習する。PyTorch代替としてのマルチモーダルネットワーク。"
}
"""
from typing import List, Dict, Tuple
from ..nn.module import SNNModule
from ..nn.attention import SDRFuzzyAttention


class MultimodalSNNHub(SNNModule):
    """
    Integrates multiple modalities (e.g., visual and textual spikes) into a shared semantic SNN space.
    Employs Hebbian learning for cross-modal association without backpropagation.
    """

    def __init__(self, shared_space_size: int, modality_names: List[str]):
        super().__init__()
        self.shared_space_size = shared_space_size
        self.modality_names = modality_names

        # Cross-modal binding weights: modality -> shared_neuron -> modality_neuron -> weight
        self.cross_modal_synapses: Dict[str, Dict[int, Dict[int, float]]] = {
            modality: {i: {} for i in range(shared_space_size)}
            for modality in modality_names
        }

        # Attention mechanism for resolving multimodal context
        self.attention = SDRFuzzyAttention(
            sdr_size=shared_space_size, threshold=0.2)

        self.register_state("cross_modal_synapses")

    def forward(self, modality_inputs: Dict[str, List[int]], learning: bool = True) -> Tuple[List[int], Dict[str, List[int]]]:
        """
        Merges modality-specific SDRs into a global context SDR, and optionally updates binding weights.
        """
        global_potentials = {i: 0.0 for i in range(self.shared_space_size)}

        # 1. Integrate spikes from all modalities into a shared space
        for mod, spikes in modality_inputs.items():
            if mod not in self.modality_names:
                continue
            for spike in spikes:
                if spike < self.shared_space_size:
                    global_potentials[spike] += 1.0

        # 2. Extract global concept (Top-K Sparse Firing)
        # 5% sparsity for energy efficiency
        active_k = max(1, int(self.shared_space_size * 0.05))
        sorted_global = sorted(global_potentials.items(),
                               key=lambda x: x[1], reverse=True)
        global_spikes = [idx for idx,
                         pot in sorted_global[:active_k] if pot > 0.5]

        # 3. Refine using Fuzzy Attention (Contextual smoothing / Auto-association)
        global_spikes = self.attention.forward(
            query=global_spikes, key=global_spikes, value=global_spikes)

        # 4. Predict cross-modal associations (e.g. Text -> Imagine Image)
        predictions = {mod: [] for mod in self.modality_names}
        for mod in self.modality_names:
            pred_pots = {i: 0.0 for i in range(self.shared_space_size)}
            for gs in global_spikes:
                for target, w in self.cross_modal_synapses[mod].get(gs, {}).items():
                    pred_pots[target] += w
            predictions[mod] = [idx for idx, p in pred_pots.items() if p > 0.5]

        # 5. Local Hebbian Learning for Cross-Modal Binding
        if learning:
            global_set = set(global_spikes)
            for mod, spikes in modality_inputs.items():
                mod_set = set(spikes)
                for gs in global_set:
                    # LTP (Long-Term Potentiation)
                    for ms in mod_set:
                        current_w = self.cross_modal_synapses[mod][gs].get(
                            ms, 0.0)
                        self.cross_modal_synapses[mod][gs][ms] = min(
                            1.0, current_w + 0.1)

                    # LTD (Long-Term Depression) / Synaptic Pruning for energy efficiency
                    for target in list(self.cross_modal_synapses[mod][gs].keys()):
                        if target not in mod_set:
                            self.cross_modal_synapses[mod][gs][target] -= 0.02
                            if self.cross_modal_synapses[mod][gs][target] <= 0:
                                del self.cross_modal_synapses[mod][gs][target]

        return global_spikes, predictions
