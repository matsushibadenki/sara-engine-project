_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/cortex.py",
    "//": "タイトル: 大脳皮質 (Cortex)",
    "//": "目的: DynamicLiquidLayerのラッパーとして、生物学的なパラメータ管理を行う。"
}

import random
from typing import List, Optional

class CortexLayer:
    """
    DynamicLiquidLayerをラップし、より生物学的なパラメータセット（層構造など）を管理しやすくするためのクラス。
    """
    def __init__(self, input_size: int, hidden_size: int, layer_type: str = "L2/3"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_type = layer_type
        
        # 層タイプに応じたパラメータ設定
        if layer_type == "L2/3":
            decay = 0.95
            density = 0.05
        elif layer_type == "L4":
            decay = 0.8
            density = 0.1
        elif layer_type == "L5/6":
            decay = 0.98
            density = 0.02
        else:
            decay = 0.9
            density = 0.05
            
        # 内部状態（簡易実装：実体はLayers.pyのものを使う想定だが、ここでは単体でも動くようにパラメータ保持）
        self.decay = decay
        self.density = density
        self.v: List[float] = [0.0] * hidden_size
        self.activity_ema: List[float] = [0.0] * hidden_size
        self.target_rate = 0.02
        self.dynamic_thresh: List[float] = [1.0] * hidden_size
        self.base_thresh = 1.0

    def forward(self, input_indices: List[int]) -> List[int]:
        """簡易的なフォワードパス"""
        fired_indices = []
        
        # 単純な入力統合（重みなし、接続チェックなしの簡易モデル）
        # ※本来はLayers.pyを使うべきだが、Cortexとして独立している場合のロジック
        input_set = set(input_indices)
        
        for i in range(self.hidden_size):
            self.v[i] *= self.decay
            
            # ランダムな接続をシミュレート（実際は固定すべきだが、ここでは軽量化のため確率的処理）
            # 入力スパイク数に応じて確率的に電位上昇
            hits = 0
            for _ in range(len(input_indices)):
                if random.random() < self.density:
                    hits += 1
            
            self.v[i] += hits * 1.0
            
            if self.v[i] >= self.dynamic_thresh[i]:
                fired_indices.append(i)
                self.v[i] = 0.0
                
        # ホメオスタシス
        ema_decay = 0.1
        for i in range(self.hidden_size):
            is_fired = 1.0 if i in fired_indices else 0.0
            self.activity_ema[i] = (1 - ema_decay) * self.activity_ema[i] + ema_decay * is_fired
            
            diff = self.activity_ema[i] - self.target_rate
            self.dynamic_thresh[i] += diff * 0.1
            if self.dynamic_thresh[i] < self.base_thresh:
                self.dynamic_thresh[i] = self.base_thresh

        return fired_indices