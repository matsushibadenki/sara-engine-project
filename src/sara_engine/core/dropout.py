_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/dropout.py",
    "//": "タイトル: Stochastic Synapse (SNN版 Dropout)",
    "//": "目的: 行列演算を使わず、Python標準のrandomモジュールのみでシナプス伝達の確率的失敗（ドロップアウト）を再現する。"
}

import random
from typing import List

class SpikeDropout:
    """
    SNNにおけるドロップアウト層。
    学習時に指定された確率でスパイクの伝達をランダムに遮断することで、
    特定のニューロンへの過剰な依存（過学習）を防ぐ。
    """
    def __init__(self, drop_rate: float = 0.1):
        self.drop_rate = drop_rate
        
    def compute(self, input_spikes: List[int], learning: bool = False) -> List[int]:
        """
        入力スパイクのリストを受け取り、ドロップアウト処理後のリストを返す。
        
        Args:
            input_spikes: 発火したニューロンのインデックスのリスト
            learning: 学習モードかどうか（推論時はドロップアウトしない）
            
        Returns:
            フィルタリングされたスパイクのリスト
        """
        # 学習時以外、またはドロップ率が0以下の場合は何もしない
        if not learning or self.drop_rate <= 0.0:
            return input_spikes
            
        output_spikes = []
        # 標準のrandom.random()で判定
        for s in input_spikes:
            if random.random() > self.drop_rate:
                output_spikes.append(s)
                
        return output_spikes