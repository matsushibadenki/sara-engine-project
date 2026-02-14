_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/utils/encoder.py",
    "//": "タイトル: ユニバーサル・スパイクエンコーダー",
    "//": "目的: ROADMAP 3に基づき、様々なデータをスパイク列に変換するユーティリティ。行列演算なし。"
}

import numpy as np
from typing import List

class SpikeEncoder:
    """
    データをスパイクに変換するエンコーダー。
    省エネルギーかつエッジデバイスでの動作を想定した設計。
    """
    def __init__(self, d_model: int):
        self.d_model = d_model

    def rate_encode(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> List[int]:
        """
        [Rate Coding] 値の大きさを発火確率に変換する。
        """
        normalized = (value - min_val) / (max_val - min_val)
        normalized = np.clip(normalized, 0, 1)
        
        # 行列演算を避け、個別に判定
        spikes = []
        for i in range(self.d_model):
            if np.random.random() < normalized * 0.2: # 最大発火率20%
                spikes.append(i)
        return spikes

    def pop_encode(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> List[int]:
        """
        [Population Coding] 値に対応する特定のニューロン群を発火させる。
        """
        center = int(((value - min_val) / (max_val - min_val)) * self.d_model)
        width = max(1, int(self.d_model * 0.05))
        
        start = max(0, center - width // 2)
        end = min(self.d_model, center + width // 2)
        return list(range(start, end))

    def text_to_temporal_spikes(self, text: str) -> List[List[int]]:
        """
        テキストを時系列のスパイク列に変換する。
        """
        sequence = []
        for char in text:
            # 各文字に固有のPopulationパターンを割り当て
            rng = np.random.RandomState(ord(char))
            num_spikes = max(1, int(self.d_model * 0.05))
            spikes = rng.choice(self.d_model, num_spikes, replace=False).tolist()
            sequence.append(spikes)
        return sequence