# パス: src/sara_engine/core/layers.py
# タイトル: Dynamic Liquid Layer (Homeostasis Core)
# 目的: 確実な負のフィードバックによる省エネ性能の担保。
# {
#     "//": "発火時に閾値を直接引き上げ、過剰なエネルギー消費を物理的に抑制するロジックを実装。"
# }

import random
from typing import List, Optional, Tuple, Any

class DynamicLiquidLayer:
    def __init__(self, input_size: int, hidden_size: int, decay: float, 
                 density: float = 0.05, input_scale: float = 1.0, 
                 rec_scale: float = 0.8, feedback_scale: float = 0.5,
                 use_rust: bool = False,
                 target_rate: float = 0.05):
        
        self.size = hidden_size
        self.input_size = input_size
        self.decay = decay
        self.target_rate = target_rate
        self.feedback_scale = feedback_scale
        
        # 行列演算を排除した辞書型による疎結合管理
        self.in_weights = [{} for _ in range(input_size)]
        for i in range(input_size):
            n = int(hidden_size * density)
            if n > 0:
                targets = random.sample(range(hidden_size), n)
                for t in targets:
                    self.in_weights[i][t] = random.uniform(-input_scale, input_scale)
        
        self.rec_weights = [{} for _ in range(hidden_size)]
        rec_density = 0.1
        for i in range(hidden_size):
            n = int(hidden_size * rec_density)
            candidates = [x for x in range(hidden_size) if x != i]
            n = min(n, len(candidates))
            if n > 0:
                targets = random.sample(candidates, n)
                for t in targets:
                    self.rec_weights[i][t] = random.uniform(-rec_scale, rec_scale)
        
        self.v = [0.0] * hidden_size
        self.refractory = [0.0] * hidden_size
        self.base_thresh = 1.0
        self.dynamic_thresh = [self.base_thresh] * hidden_size
        self.activity_ema = [0.0] * hidden_size # 0から開始して上昇を検知
        
        self.feedback_weights = []
        for i in range(hidden_size):
            targets = random.sample(range(hidden_size), max(1, int(hidden_size * 0.05)))
            self.feedback_weights.append(targets)

    def forward_with_feedback(self, active_inputs: List[int], 
                             prev_active_hidden: List[int], 
                             feedback_active: List[int] = [], 
                             learning: bool = False,
                             attention_signal: List[int] = []) -> List[int]:
        
        # 1. 減衰と不応期の進行
        for i in range(self.size):
            if self.refractory[i] > 0:
                self.refractory[i] -= 1.0
            self.v[i] *= self.decay

        # 2. 入力（イベント駆動）
        for idx in active_inputs:
            if idx < self.input_size:
                for target_id, w in self.in_weights[idx].items():
                    self.v[target_id] += w
                
        # 3. 再帰
        for pre_h_id in prev_active_hidden:
            if pre_h_id < self.size:
                for target_id, w in self.rec_weights[pre_h_id].items():
                    self.v[target_id] += w
        
        # 4. フィードバック/アテンション (省略なし)
        if feedback_active:
            for fb_id in feedback_active:
                if fb_id < len(self.feedback_weights):
                    for target_id in self.feedback_weights[fb_id]:
                        self.v[target_id] += self.feedback_scale
        
        if attention_signal:
            for idx in attention_signal:
                if idx < self.size:
                    self.v[idx] += 1.5

        # 5. 発火判定
        fired_indices = []
        for i in range(self.size):
            if self.v[i] >= self.dynamic_thresh[i] and self.refractory[i] <= 0:
                fired_indices.append(i)
                
        # 6. ホメオスタシス（負のフィードバックを非対称化）
        ema_decay = 0.2  # 更新を高速化
        homeo_up_rate = 1.0 # 上昇感度
        homeo_down_rate = 0.001 # 下降感度（上昇の1000分の1）
        
        for i in range(self.size):
            current_act = 1.0 if i in fired_indices else 0.0
            self.activity_ema[i] = (1 - ema_decay) * self.activity_ema[i] + ema_decay * current_act
            
            diff = self.activity_ema[i] - self.target_rate
            
            # 過剰発火時は即座に閾値を引き上げる
            if diff > 0:
                self.dynamic_thresh[i] += homeo_up_rate * diff
            else:
                # 低活動時は「忘却」レベルの超低速で閾値を下げる
                self.dynamic_thresh[i] += homeo_down_rate * diff
            
            # 物理的限界のクリッピング
            if self.dynamic_thresh[i] < self.base_thresh:
                self.dynamic_thresh[i] = self.base_thresh
            if self.dynamic_thresh[i] > 10.0:
                self.dynamic_thresh[i] = 10.0

        # 7. リセットと強めの不応期
        for idx in fired_indices:
            self.v[idx] = 0.0
            self.refractory[idx] = 4.0 
            
        return fired_indices

    def get_state(self) -> Tuple[List[float], List[float]]:
        return list(self.v), list(self.dynamic_thresh)

    def reset(self):
        self.v = [0.0] * self.size
        self.refractory = [0.0] * self.size
        self.dynamic_thresh = [self.base_thresh] * self.size
        self.activity_ema = [0.0] * self.size