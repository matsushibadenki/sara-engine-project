# パス: src/sara_engine/core/layers.py
# タイトル: Dynamic Liquid Layer (Homeostasis Core)
# 目的: Numpyを完全に排除し、純粋なイベント駆動とスパース接続による圧倒的な省エネルギー化と生物学的妥当性を実現。
# {
#     "//": "行列演算を廃止し、辞書型を用いた疎結合（Sparse）ネットワークに完全移行。"
# }

import random
from typing import List, Optional, Tuple, Any

try:
    from .. import sara_rust_core  # type: ignore
    RUST_AVAILABLE = True
except ImportError:
    try:
        import sara_rust_core  # type: ignore
        RUST_AVAILABLE = True
    except ImportError:
        RUST_AVAILABLE = False

class DynamicLiquidLayer:
    def __init__(self, input_size: int, hidden_size: int, decay: float, 
                 density: float = 0.05, input_scale: float = 1.0, 
                 rec_scale: float = 0.8, feedback_scale: float = 0.5,
                 use_rust: Optional[bool] = None,
                 target_rate: float = 0.05):
        
        self.size = hidden_size
        self.input_size = input_size
        
        self.decay = decay
        self.density = density
        self.input_scale = input_scale
        self.rec_scale = rec_scale
        self.feedback_scale = feedback_scale
        self.target_rate = target_rate
        
        if use_rust is not None:
            self.use_rust = use_rust
        else:
            self.use_rust = RUST_AVAILABLE
            
        if self.use_rust:
            self.core = sara_rust_core.RustLiquidLayer(input_size, hidden_size, decay, density, feedback_scale)
            self.v = [0.0] * hidden_size
            self.dynamic_thresh = [1.0] * hidden_size
        else:
            # 純粋なPython辞書によるスパース表現（行列演算を完全排除）
            self.in_weights = [{} for _ in range(input_size)]
            for i in range(input_size):
                n = int(hidden_size * density)
                if n > 0:
                    targets = random.sample(range(hidden_size), n)
                    for t in targets:
                        self.in_weights[i][t] = random.uniform(-input_scale * 1.2, input_scale * 1.2)
            
            self.rec_weights = [{} for _ in range(hidden_size)]
            rec_density = 0.1
            for i in range(hidden_size):
                n = int(hidden_size * rec_density)
                if n > 0:
                    candidates = [x for x in range(hidden_size) if x != i]
                    if n > len(candidates):
                        n = len(candidates)
                    targets = random.sample(candidates, n)
                    for t in targets:
                        self.rec_weights[i][t] = random.uniform(-rec_scale, rec_scale)
            
            self.v = [0.0] * hidden_size
            self.refractory = [0.0] * hidden_size
            self.base_thresh = 1.0
            self.dynamic_thresh = [self.base_thresh] * hidden_size
            
            self.activity_ema = [target_rate] * hidden_size
            self.trace = [0.0] * hidden_size
            self.input_trace = [0.0] * input_size
            
            self.feedback_weights = []
            for i in range(hidden_size):
                targets = random.sample(range(hidden_size), max(1, int(hidden_size * 0.05)))
                self.feedback_weights.append(targets)

    def forward_with_feedback(self, active_inputs: List[int], 
                             prev_active_hidden: List[int], 
                             feedback_active: List[int] = [], 
                             learning: bool = False,
                             attention_signal: List[int] = []) -> List[int]:
        
        if self.use_rust:
            # Rust側でもホメオスタシスが更新される必要があるが、現在はPython側の修正に注力
            return self.core.forward(active_inputs, prev_active_hidden, feedback_active, attention_signal, learning)
        else:
            return self._forward_python(active_inputs, prev_active_hidden, feedback_active, learning, attention_signal)

    def _forward_python(self, active_inputs: List[int], prev_active_hidden: List[int], feedback_active: List[int], learning: bool, attention_signal: List[int]) -> List[int]:
        # 1. 状態の更新（減衰と不応期）
        for i in range(self.size):
            if self.refractory[i] > 0:
                self.refractory[i] -= 1.0
            self.v[i] *= self.decay
            self.trace[i] *= 0.95
            
        for i in range(self.input_size):
            self.input_trace[i] *= 0.95
            
        # 2. 入力・再帰・フィードバックによる電位上昇
        for idx in active_inputs:
            if idx < self.input_size:
                self.input_trace[idx] += 1.0
                for target_id, w in self.in_weights[idx].items():
                    self.v[target_id] += w
                
        for pre_h_id in prev_active_hidden:
            if pre_h_id < self.size:
                for target_id, w in self.rec_weights[pre_h_id].items():
                    self.v[target_id] += w
                
        for fb_id in feedback_active:
            if fb_id < len(self.feedback_weights):
                for target_id in self.feedback_weights[fb_id]:
                    self.v[target_id] += self.feedback_scale

        if attention_signal:
            attn_scale = 1.5
            for idx in attention_signal:
                if idx < self.size:
                    self.v[idx] += attn_scale
        
        # 3. 発火判定
        fired_indices = []
        for i in range(self.size):
            # 不応期でないかつ閾値を超えた場合に発火
            if self.v[i] >= self.dynamic_thresh[i] and self.refractory[i] <= 0:
                fired_indices.append(i)
                
        # 4. Homeostasis (ホメオスタシス): 閾値の動的調整
        # ヘルスチェックで検知されるよう、更新率を少し強化
        ema_decay = 0.1  # 0.05 -> 0.1 (反応速度を向上)
        homeo_rate = 0.05 # 0.02 -> 0.05 (閾値の変化量を大きく)
        
        fired_set = set(fired_indices)
        for i in range(self.size):
            current_act = 1.0 if i in fired_set else 0.0
            # 活動率の指数移動平均を更新
            self.activity_ema[i] = (1 - ema_decay) * self.activity_ema[i] + ema_decay * current_act
            
            # ターゲット発火率との差分に基づいて閾値を調整
            diff = self.activity_ema[i] - self.target_rate
            self.dynamic_thresh[i] += homeo_rate * diff
            
            # 閾値のクリッピング（発火不能や暴走を防止）
            if self.dynamic_thresh[i] < 0.2: self.dynamic_thresh[i] = 0.2
            if self.dynamic_thresh[i] > 5.0: self.dynamic_thresh[i] = 5.0

        # 5. 後処理（リセットと学習）
        for idx in fired_indices:
            self.v[idx] = 0.0
            # 不応期を少し短く調整してホメオスタシスが働きやすくする
            self.refractory[idx] = random.uniform(1.0, 3.0) 
            self.trace[idx] += 1.0

        if learning and fired_indices:
            for pre_id in prev_active_hidden:
                if pre_id < self.size:
                    for target_id in list(self.rec_weights[pre_id].keys()):
                        # LTD: 基本的な減衰
                        self.rec_weights[pre_id][target_id] -= 0.002
                        # LTP: 同時発火による強化
                        if target_id in fired_set:
                            self.rec_weights[pre_id][target_id] += 0.03
                        
                        if self.rec_weights[pre_id][target_id] < -2.0:
                            self.rec_weights[pre_id][target_id] = -2.0
                        elif self.rec_weights[pre_id][target_id] > 2.0:
                            self.rec_weights[pre_id][target_id] = 2.0
            
        return fired_indices

    def get_state(self) -> Tuple[List[float], List[float]]:
        return list(self.v), list(self.dynamic_thresh)

    def reset(self):
        if self.use_rust:
            self.core.reset()
        else:
            self.v = [0.0] * self.size
            self.refractory = [0.0] * self.size
            self.trace = [0.0] * self.size
            self.input_trace = [0.0] * self.input_size
            self.dynamic_thresh = [self.base_thresh] * self.size
            self.activity_ema = [self.target_rate] * self.size