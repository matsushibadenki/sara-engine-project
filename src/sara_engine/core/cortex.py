_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/cortex.py",
    "//": "タイトル: 大脳皮質 (Cortex)",
    "//": "目的: DynamicLiquidLayerのラッパーとして、生物学的なパラメータ管理とコンパートメント化されたカラム構造を提供する。"
}

import random
from typing import List, Optional, Dict

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
            
        # 内部状態
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
        for i in range(self.hidden_size):
            self.v[i] *= self.decay
            
            # ランダムな接続をシミュレート
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


class CorticalColumn:
    """
    複数のコンパートメント（サブネットワーク）を持ち、コンテキストに応じて
    発火経路を切り替える大脳皮質カラムのモデル。
    破滅的忘却を防ぐためのモジュール化された構造を持つ。
    """
    def __init__(self, input_size: int, hidden_size_per_comp: int, compartment_names: List[str], target_rate: float = 0.05):
        self.input_size = input_size
        self.hidden_size_per_comp = hidden_size_per_comp
        self.compartments: Dict[str, CortexLayer] = {}
        
        for name in compartment_names:
            # 各コンパートメントに独立したCortexLayerを割り当てる
            layer = CortexLayer(input_size=input_size, hidden_size=hidden_size_per_comp)
            layer.target_rate = target_rate
            self.compartments[name] = layer
            
    def forward_latent_chain(self, active_inputs: List[int], prev_active_hidden: List[int], 
                             current_context: str, learning: bool = False, 
                             reward_signal: float = 0.0) -> List[int]:
        """
        指定されたコンテキストのコンパートメントのみを駆動し、発火連鎖をシミュレートする。
        他のコンテキストは完全に遮断されるため、干渉（破滅的忘却）を防ぐ。
        """
        if current_context not in self.compartments:
            return []
            
        target_layer = self.compartments[current_context]
        
        # 本来はSTDPや報酬信号による学習のロジックが含まれるが、ここではフォワードパスのみ実行
        fired_indices = target_layer.forward(active_inputs)
        
        return fired_indices
        
    def get_compartment_states(self) -> Dict[str, Dict[str, int]]:
        """
        各コンパートメントの現在の状態（膜電位が0より大きい残存ニューロン数）を取得する。
        """
        states = {}
        for name, layer in self.compartments.items():
            active_count = sum(1 for v in layer.v if v > 0.0)
            states[name] = {
                "active_neurons": active_count
            }
        return states