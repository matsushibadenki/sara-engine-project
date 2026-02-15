_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/normalization.py",
    "//": "タイトル: Spike Intrinsic Plasticity (SNN版 Layer Normalization)",
    "//": "目的: TransformersのLayerNormのSNN代替。ニューロンの閾値と発火率を動的に調整し、過剰発火を抑制、沈黙状態を救済するホメオスタシス機構。"
}

import numpy as np
from typing import List

class SpikeIntrinsicPlasticity:
    def __init__(self, d_model: int, target_rate: float = 0.1, adapt_rate: float = 0.01):
        self.d_model = d_model
        self.target_rate = target_rate
        self.adapt_rate = adapt_rate
        self.firing_rates = np.zeros(d_model, dtype=np.float32)
        
    def compute(self, input_spikes: List[int], learning: bool = False) -> List[int]:
        valid_in = [s for s in input_spikes if s < self.d_model]
        
        if learning:
            active_mask = np.zeros(self.d_model, dtype=np.float32)
            for s in valid_in:
                active_mask[s] = 1.0
            
            for i in range(self.d_model):
                self.firing_rates[i] = (1.0 - self.adapt_rate) * self.firing_rates[i] + self.adapt_rate * active_mask[i]
            
        output_spikes = []
        
        for s in valid_in:
            if self.firing_rates[s] > self.target_rate * 1.5:
                drop_prob = min(0.9, (self.firing_rates[s] - self.target_rate * 1.5) / self.target_rate)
                if np.random.rand() > drop_prob:
                    output_spikes.append(s)
            else:
                output_spikes.append(s)
                
        if learning:
            for i in range(self.d_model):
                if self.firing_rates[i] < self.target_rate * 0.1:
                    if np.random.rand() < (self.target_rate * 0.05):
                        output_spikes.append(i)
                        
        return list(set(output_spikes))
