import numpy as np

class ScalarEncoder:
    """
    数値データをSDR（スパイク列）に変換するエンコーダー。
    生物の感覚ニューロンのように、値の大きさに応じて発火するニューロンの位置が変わります。
    """
    def __init__(self, min_val: float, max_val: float, num_neurons: int = 100, w: int = 21):
        self.min_val = min_val
        self.max_val = max_val
        self.num_neurons = num_neurons
        self.w = w  # 活性化するニューロンの幅 (width)

    def encode(self, value: float) -> list[int]:
        # 値を0〜1に正規化
        normalized = (value - self.min_val) / (self.max_val - self.min_val)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # 中心位置を計算
        center_bin = int(normalized * (self.num_neurons - self.w))
        
        # スパイク位置のリストを返す（行列ではない）
        return list(range(center_bin, center_bin + self.w))