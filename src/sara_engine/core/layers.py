# パス: src/sara_engine/core/layers.py
# タイトル: Dynamic Liquid Layer (Homeostasis Core)
# 目的: 活動依存型即時加算による確実なホメオスタシスの実現。
# {
#     "//": "発火時に閾値を直接加算(0.1以上)することで、減衰による相殺を物理的に不可能にし、省エネ性能を担保する。"
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
        self.base_thresh = 1.0
        
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
        self.dynamic_thresh = [self.base_thresh] * hidden_size
        self.activity_ema = [0.0] * hidden_size
        
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
        
        # 4. フィードバックとアテンション
        if feedback_active:
            for fb_id in feedback_active:
                if fb_id < len(self.feedback_weights):
                    for target_id in self.feedback_weights[fb_id]:
                        self.v[target_id] += self.feedback_scale
        if attention_signal:
            for idx in attention_signal:
                if idx < self.size: self.v[idx] += 1.5

        # 5. 発火判定
        fired_indices = []
        for i in range(self.size):
            if self.v[i] >= self.dynamic_thresh[i] and self.refractory[i] <= 0:
                fired_indices.append(i)
                
        # 6. ホメオスタシス（修正：即時加算ロジック）
        ema_decay = 0.1
        homeo_up_rate = 1.0 
        
        for i in range(self.size):
            is_fired = i in fired_indices
            self.activity_ema[i] = (1 - ema_decay) * self.activity_ema[i] + (ema_decay if is_fired else 0.0)
            
            # 発火時：即座に閾値を加算（最小0.1のステップを保証して相殺を防ぐ）
            if is_fired:
                diff = max(0, self.activity_ema[i] - self.target_rate)
                self.dynamic_thresh[i] += 0.1 + (diff * homeo_up_rate)
            else:
                # 非発火時：極めてゆっくりとベースラインへ減衰
                if self.dynamic_thresh[i] > self.base_thresh:
                    # 毎ステップ現在の差の0.5%しか減衰させない
                    self.dynamic_thresh[i] -= (self.dynamic_thresh[i] - self.base_thresh) * 0.005

            # クリッピング
            if self.dynamic_thresh[i] < self.base_thresh:
                self.dynamic_thresh[i] = self.base_thresh
            if self.dynamic_thresh[i] > 15.0:
                self.dynamic_thresh[i] = 15.0

        # 7. リセットと不応期
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