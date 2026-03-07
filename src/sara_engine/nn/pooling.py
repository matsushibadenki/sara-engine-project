# ディレクトリパス: src/sara_engine/nn/pooling.py
# ファイルの日本語タイトル: 階層的SDRプーリング層
# ファイルの目的や内容: 視覚や言語の階層的特徴抽出を行うための、局所スパイク統合・抽象化レイヤー。受容野(Receptive Fields)の概念を利用し、行列演算なしで自己組織化的に特徴を抽出する。
from typing import List, Dict, Set
import random
from .module import SNNModule

class HierarchicalSDRPooling(SNNModule):
    """
    Aggregates lower-level spikes into higher-level abstract representations.
    Biologically inspired by receptive fields in visual and auditory cortices.
    Utilizes unsupervised Hebbian learning to form abstract concepts without backpropagation.
    """
    def __init__(self, in_size: int, out_size: int, compression_ratio: float = 0.5, learning_rate: float = 0.05):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.compression_ratio = compression_ratio
        self.learning_rate = learning_rate
        
        # Receptive fields: Mapping from higher abstraction to lower features
        # Structure: {out_neuron_id: {in_neuron_id: weight}}
        self.receptive_fields: Dict[int, Dict[int, float]] = {}
        for out_neuron in range(out_size):
            # Connect to a localized / random subset of input neurons (sparse connectivity)
            num_connections = max(1, int(in_size * 0.1)) # 10% coverage initially
            selected_inputs = random.sample(range(in_size), num_connections)
            self.receptive_fields[out_neuron] = {inp: random.uniform(0.1, 0.5) for inp in selected_inputs}
            
        self.register_state("receptive_fields")
        
    def forward(self, in_spikes: List[int], learning: bool = True) -> List[int]:
        """
        Pools spikes to create a more abstract, compressed SDR.
        """
        if not in_spikes:
            return []

        in_set = set(in_spikes)
        potentials: Dict[int, float] = {}
        
        # Integrate spikes within each receptive field
        for out_neuron, rf in self.receptive_fields.items():
            overlap_score = 0.0
            for inp_spike in in_spikes:
                if inp_spike in rf:
                    overlap_score += rf[inp_spike]
            if overlap_score > 0:
                potentials[out_neuron] = overlap_score
            
        # Dynamic K-WTA (Winner-Take-All) firing to maintain sparsity and energy efficiency
        # Number of active neurons dynamically scales with input density and compression ratio
        active_ratio = len(in_spikes) / max(1, self.in_size)
        k = max(1, int(self.out_size * self.compression_ratio * active_ratio))
        
        sorted_neurons = sorted(potentials.items(), key=lambda x: x[1], reverse=True)
        # Firing threshold is adaptive based on the top potential
        threshold = sorted_neurons[0][1] * 0.3 if sorted_neurons else 0.5
        
        out_spikes = [neuron for neuron, pot in sorted_neurons[:k] if pot > threshold]
        
        # Unsupervised Learning: Self-Organizing Map (SOM) / Hebbian approach
        if learning and out_spikes:
            self._update_receptive_fields(in_set, out_spikes)
            
        out_spikes.sort()
        return out_spikes

    def _update_receptive_fields(self, in_set: Set[int], out_spikes: List[int]) -> None:
        """
        Update the receptive fields based on pre- and post-synaptic spike coincidence.
        Strengthens connections to active inputs and weakens connections to inactive ones.
        """
        for out_neuron in out_spikes:
            rf = self.receptive_fields[out_neuron]
            to_remove = []
            
            # LTP (Long-Term Potentiation) for active inputs
            for inp in in_set:
                if inp in rf:
                    rf[inp] = min(1.0, rf[inp] + self.learning_rate)
                else:
                    # Synaptogenesis: create new connection with low probability
                    if random.random() < 0.01:
                        rf[inp] = 0.1
                        
            # LTD (Long-Term Depression) for inactive inputs within the receptive field
            for inp in rf:
                if inp not in in_set:
                    rf[inp] -= self.learning_rate * 0.5
                    if rf[inp] <= 0.01:
                        to_remove.append(inp)
                        
            # Prune dead synapses (Energy saving)
            for inp in to_remove:
                del rf[inp]