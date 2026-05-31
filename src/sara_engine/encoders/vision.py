"""
{
    "//": "ディレクトリパス: src/sara_engine/encoders/vision.py",
    "//": "タイトル: 視覚SDRエンコーダ - 特徴量強度ソート版",
    "//": "目的: 行列演算に依存せず、画像特徴量の強い部分を正確にSDRへ反映する。インデックスの偏りを防ぐ。"
}
"""

import random
from typing import List

class ImageSpikeEncoder:
    def __init__(self, output_size: int = 2048, density: float = 0.02):
        self.output_size = output_size
        self.density = density
        self.rng = random.Random(42)
        # 空間特徴をSDRの1次元配列へマッピングする固定プロジェクション
        self.projection = [self.rng.randint(0, output_size - 1) for _ in range(10000)]

    def encode(self, image_features: List[float]) -> List[int]:
        """画像特徴量(輝度やエッジ強度)の強い部分を優先してSDRへ変換"""
        target_n = int(self.output_size * self.density)
        
        # 特徴量の強さとインデックスをペアにして、強い順にソート（行列演算を使わないアプローチ）
        scored_features = [(val, i) for i, val in enumerate(image_features) if val > 0.6]
        scored_features.sort(key=lambda x: x[0], reverse=True)
        
        sdr_set = set()
        for _, idx in scored_features:
            mapped_idx = self.projection[idx % len(self.projection)]
            sdr_set.add(mapped_idx)
            if len(sdr_set) >= target_n:
                break
                
        return sorted(list(sdr_set))