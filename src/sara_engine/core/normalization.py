_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/normalization.py",
    "//": "タイトル: Spike Intrinsic Plasticity (SNN版 Layer Normalization)",
    "//": "目的: 標準リストで各ニューロンの発火率を追跡し、閾値調整によるホメオスタシスを実現する。"
}

import random
from typing import List

class SpikeIntrinsicPlasticity:
    """
    ニューロンの内部可塑性（Intrinsic Plasticity）を用いた正規化層。
    各ニューロンの平均発火率を追跡し、目標レートを超えた場合に抑制的な確率的フィルタリングを行うことで、
    ネットワーク全体の活動レベルを一定範囲に保つ（Layer Normalizationの代替）。
    """
    def __init__(self, d_model: int, target_rate: float = 0.1, adapt_rate: float = 0.01):
        self.d_model = d_model
        self.target_rate = target_rate
        self.adapt_rate = adapt_rate
        # Pythonの標準リストで状態を保持
        self.firing_rates: List[float] = [0.0] * d_model
        
    def compute(self, input_spikes: List[int], learning: bool = False) -> List[int]:
        """
        発火率に基づく動的なフィルタリングを行う。
        """
        # 入力インデックスの範囲チェック（安全策）
        valid_in = [s for s in input_spikes if s < self.d_model]
        
        if learning:
            # 発火率の移動平均を更新
            # 外部ライブラリのベクトル演算を使わず、発火したニューロンとそれ以外を個別に更新
            
            # 1. 全ニューロンを減衰させる（非発火を反映）
            decay_factor = 1.0 - self.adapt_rate
            for i in range(self.d_model):
                self.firing_rates[i] *= decay_factor
            
            # 2. 今回発火したニューロンのレートを上昇させる
            for s in valid_in:
                # 減衰済みなので、adapt_rate * 1.0 を加算するだけでよい
                # (元: r = r*(1-a) + a*1)
                self.firing_rates[s] += self.adapt_rate
            
        output_spikes = []
        
        # 過剰に発火しているニューロンの出力を確率的に抑制
        threshold_rate = self.target_rate * 1.5
        
        for s in valid_in:
            rate = self.firing_rates[s]
            if rate > threshold_rate:
                # 目標を超えた分に応じてドロップ確率を計算
                excess = rate - threshold_rate
                drop_prob = min(0.9, excess / self.target_rate)
                
                if random.random() > drop_prob:
                    output_spikes.append(s)
            else:
                # 目標以下の場合はそのまま通過
                output_spikes.append(s)
                
        return output_spikes

    def reset(self):
        """状態のリセット"""
        self.firing_rates = [0.0] * self.d_model