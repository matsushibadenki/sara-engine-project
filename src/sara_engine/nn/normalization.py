_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/nn/normalization.py",
    "//": "ファイルの日本語タイトル: スパイキング層正規化 (LayerNorm代替)",
    "//": "ファイルの目的や内容: ネットワーク内の発火率を一定に保つための恒常性(Homeostasis)メカニズム。TransformerのLayerNormのSNN版として機能する。"
}

import random
from typing import List
from .module import SNNModule

class SpikeLayerNorm(SNNModule):
    """
    Biological alternative to Layer Normalization (Homeostatic Plasticity).
    Normalizes the firing rate by dynamically restricting the maximum number of spikes
    and injecting noise if activity drops too low, preventing spike explosion/vanishing.
    """
    def __init__(self, target_spikes: int):
        super().__init__()
        self.target_spikes = target_spikes

    def forward(self, spikes: List[int], learning: bool = False) -> List[int]:
        num_spikes = len(spikes)
        
        if num_spikes > self.target_spikes:
            # Randomly downsample to target rate to prevent explosion
            return random.sample(spikes, self.target_spikes)
            
        elif num_spikes < self.target_spikes // 2 and learning and num_spikes > 0:
            # Add intrinsic noise spikes if activity is too low during learning
            # to prevent vanishing spikes (Dead neurons)
            out_spikes = list(spikes)
            needed = (self.target_spikes // 2) - num_spikes
            max_val = max(spikes) * 2 if spikes else 1000
            for _ in range(needed):
                noise_spike = random.randint(0, max_val)
                if noise_spike not in out_spikes:
                    out_spikes.append(noise_spike)
            return out_spikes
            
        return spikes