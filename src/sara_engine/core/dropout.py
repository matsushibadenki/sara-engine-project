_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/dropout.py",
    "//": "タイトル: Stochastic Synapse (SNN版 Dropout)",
    "//": "目的: TransformersのDropoutのSNN代替。確率的なシナプス伝達不良を模倣し、学習時の過適合（過学習）を防ぐ。"
}

import numpy as np
from typing import List

class SpikeDropout:
    def __init__(self, drop_rate: float = 0.1):
        self.drop_rate = drop_rate
        
    def compute(self, input_spikes: List[int], learning: bool = False) -> List[int]:
        if not learning or self.drop_rate <= 0.0:
            return input_spikes
            
        output_spikes = []
        for s in input_spikes:
            if np.random.rand() > self.drop_rate:
                output_spikes.append(s)
        return output_spikes
