_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/encoders/audio.py",
    "//": "タイトル: 聴覚SDRエンコーダ",
    "//": "目的: 音声の周波数スペクトルを直接SDRへマッピングする。ImportError修正のためクラス名を変更。"
}

import numpy as np
from typing import List

class AudioSpikeEncoder:
    def __init__(self, output_size: int = 2048, density: float = 0.02):
        self.output_size = output_size
        self.density = density
        self.rng = np.random.RandomState(99)
        self.projection = self.rng.randint(0, output_size, size=5000)

    def encode(self, frequencies: List[float]) -> List[int]:
        sdr_set = set()
        for i, val in enumerate(frequencies):
            if val > 0.5:
                mapped_idx = self.projection[i % len(self.projection)]
                sdr_set.add(mapped_idx)
        
        target_n = int(self.output_size * self.density)
        sdr_list = sorted(list(sdr_set))
        if len(sdr_list) > target_n:
            sdr_list = sdr_list[:target_n]
            
        return sdr_list