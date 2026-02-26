_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/nn/dropout.py",
    "//": "ファイルの日本語タイトル: スパイク・ドロップアウト",
    "//": "ファイルの目的や内容: 学習時にスパイクを一定確率で遮断し、局所学習(STDP)の過学習を防ぐTransformerのDropoutのSNN版。"
}

import random
from typing import List
from .module import SNNModule

class SpikeDropout(SNNModule):
    """
    Biological alternative to Dropout.
    Randomly drops spikes during the learning phase to encourage robust feature extraction.
    """
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, spikes: List[int], learning: bool = False) -> List[int]:
        if not learning or self.p == 0.0:
            return spikes
            
        out_spikes = []
        for s in spikes:
            # Drop spikes with probability p
            if random.random() > self.p:
                out_spikes.append(s)
                
        return out_spikes