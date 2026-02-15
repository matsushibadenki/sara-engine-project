_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/encoders/vision.py",
    "//": "タイトル: 視覚SDRエンコーダ",
    "//": "目的: 画像のピクセル・特徴量配列を直接SDRへマッピングする。クラス名をImageSpikeEncoderへ変更。"
}

import numpy as np
from typing import List

class ImageSpikeEncoder:
    def __init__(self, output_size: int = 2048, density: float = 0.02):
        self.output_size = output_size
        self.density = density
        self.rng = np.random.RandomState(42)
        # 空間特徴をSDRの1次元配列へマッピングする固定プロジェクション
        self.projection = self.rng.randint(0, output_size, size=10000)

    def encode(self, image_features: List[float]) -> List[int]:
        """画像特徴量(輝度やエッジ強度)を閾値判定してSDRへ変換"""
        sdr_set = set()
        for i, val in enumerate(image_features):
            if val > 0.6:  # 発火閾値
                mapped_idx = self.projection[i % len(self.projection)]
                sdr_set.add(mapped_idx)
        
        target_n = int(self.output_size * self.density)
        sdr_list = sorted(list(sdr_set))
        if len(sdr_list) > target_n:
            sdr_list = sdr_list[:target_n]
            
        return sdr_list