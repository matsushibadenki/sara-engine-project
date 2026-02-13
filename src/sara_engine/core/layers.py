_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/layers.py",
    "//": "タイトル: Dynamic Liquid Layer (Homeostasis Core)",
    "//": "目的: 移動平均を用いた堅牢なホメオスタシス機能を実装。"
}

import numpy as np
from typing import List, Optional, Tuple

# Rust Core Import Check
try:
    from .. import sara_rust_core
    RUST_AVAILABLE = True
except ImportError:
    try:
        import sara_rust_core
        RUST_AVAILABLE = True
    except ImportError:
        RUST_AVAILABLE = False

class DynamicLiquidLayer:
    """
    Hybrid Layer with Robust Homeostasis.
    ニューロン自身が目標発火率(target_rate)を維持するように閾値を自動調整します。
    """
    def __init__(self, input_size: int, hidden_size: int, decay: float, 
                 density: float = 0.05, input_scale: float = 1.0, 
                 rec_scale: float = 0.8, feedback_scale: float = 0.5,
                 use_rust: Optional[bool] = None,
                 target_rate: float = 0.05): # [New] 目標発火率 (5%)
        
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
            # Rust実装の初期化
            self.core = sara_rust_core.RustLiquidLayer(input_size, hidden_size, decay, density, feedback_scale)
            # 互換性用ダミー
            self.in_indices = []
            self.in_weights = []
            self.rec_indices = []
            self.rec_weights = []
            self.v = np.zeros(hidden_size, dtype=np.float32)
            self.refractory = np.zeros(hidden_size, dtype=np.float32)
            self.dynamic_thresh = np.ones(hidden_size, dtype=np.float32)
            self.trace = np.zeros(hidden_size, dtype=np.float32)
            self.input_trace = np.zeros(input_size, dtype=np.float32)
            self.activity_ema = np.zeros(hidden_size, dtype=np.float32)
        else:
            # Python実装
            self.in_indices = []
            self.in_weights = []
            self.rec_indices = []
            self.rec_weights = []
            
            # Init Input Weights
            for i in range(input_size):
                n = int(hidden_size * density)
                if n > 0:
                    idx = np.random.choice(hidden_size, n, replace=False).astype(np.int32)
                    w = np.random.uniform(-input_scale * 1.2, input_scale * 1.2, n).astype(np.float32)
                    self.in_indices.append(idx)
                    self.in_weights.append(w)
                else:
                    self.in_indices.append(np.array([], dtype=np.int32))
                    self.in_weights.append(np.array([], dtype=np.float32))
            
            # Init Recurrent Weights
            rec_density = 0.1
            for i in range(hidden_size):
                n = int(hidden_size * rec_density)
                if n > 0:
                    idx = np.random.choice(hidden_size, n, replace=False).astype(np.int32)
                    idx = idx[idx != i]
                    w = np.random.uniform(-rec_scale, rec_scale, len(idx)).astype(np.float32)
                    self.rec_indices.append(idx)
                    self.rec_weights.append(w)
                else:
                    self.rec_indices.append(np.array([], dtype=np.int32))
                    self.rec_weights.append(np.array([], dtype=np.float32))
            
            self.v = np.zeros(hidden_size, dtype=np.float32)
            self.refractory = np.zeros(hidden_size, dtype=np.float32)
            self.base_thresh = 1.0 # 基準閾値
            self.dynamic_thresh = np.ones(hidden_size, dtype=np.float32) * self.base_thresh
            
            # [New] Activity Moving Average (活動履歴)
            # 初期値は目標値にしておく（最初から暴走しないように）
            self.activity_ema = np.ones(hidden_size, dtype=np.float32) * target_rate
            
            # STDP Traces
            self.trace = np.zeros(hidden_size, dtype=np.float32)
            self.input_trace = np.zeros(input_size, dtype=np.float32)
            
            self.feedback_weights = []
            rng = np.random.RandomState(42)
            for i in range(hidden_size):
                targets = rng.choice(hidden_size, int(hidden_size * 0.05), replace=False)
                self.feedback_weights.append(targets)

    def forward_with_feedback(self, active_inputs: List[int], 
                             prev_active_hidden: List[int], 
                             feedback_active: List[int] = [], 
                             learning: bool = False,
                             attention_signal: List[int] = []) -> List[int]:
        
        if self.use_rust:
            return self.core.forward(active_inputs, prev_active_hidden, feedback_active, attention_signal, learning)
        else:
            return self._forward_python(active_inputs, prev_active_hidden, feedback_active, learning, attention_signal)

    def _forward_python(self, active_inputs, prev_active_hidden, feedback_active, learning, attention_signal):
        # 1. Decay & Refractory
        self.refractory = np.maximum(0, self.refractory - 1)
        self.v *= self.decay
        self.trace *= 0.95
        self.input_trace *= 0.95
        
        if active_inputs:
            self.input_trace[active_inputs] += 1.0

        # 2. Integration (Input + Recurrent + Feedback + Attention)
        # (処理速度優先のため、ループ展開は維持)
        for pre_id in active_inputs:
            if pre_id < len(self.in_indices):
                targets = self.in_indices[pre_id]
                ws = self.in_weights[pre_id]
                if len(targets) > 0: self.v[targets] += ws
        
        for pre_h_id in prev_active_hidden:
            if pre_h_id < len(self.rec_indices):
                targets = self.rec_indices[pre_h_id]
                ws = self.rec_weights[pre_h_id]
                if len(targets) > 0: self.v[targets] += ws
        
        if feedback_active:
            for fb_id in feedback_active:
                if fb_id < len(self.feedback_weights):
                    targets = self.feedback_weights[fb_id]
                    self.v[targets] += self.feedback_scale

        if attention_signal:
            attn_scale = 1.5
            for idx in attention_signal:
                if idx < self.size: self.v[idx] += attn_scale
        
        # 3. Fire Condition
        ready_mask = (self.v >= self.dynamic_thresh) & (self.refractory <= 0)
        fired_indices = np.where(ready_mask)[0].tolist()
        
        # 4. Post-Fire Updates
        fired_arr = np.array(fired_indices, dtype=int)
        
        # [New] Homeostasis: Exponential Moving Average Update
        # 発火したニューロンは 1.0、しなかったニューロンは 0.0 を混ぜる
        ema_decay = 0.05 # 時定数 (ゆっくり追従)
        current_activity = np.zeros(self.size, dtype=np.float32)
        if fired_indices:
            current_activity[fired_arr] = 1.0
        
        self.activity_ema = (1 - ema_decay) * self.activity_ema + ema_decay * current_activity
        
        # [New] Threshold Adaptation based on Error
        # 活動しすぎ (ema > target) -> 閾値を上げる (+)
        # 活動しなさすぎ (ema < target) -> 閾値を下げる (-)
        homeo_rate = 0.02 # 調整スピード
        diff = self.activity_ema - self.target_rate
        self.dynamic_thresh += homeo_rate * diff
        
        # Reset & Refractory
        if fired_indices:
            self.v[fired_arr] = 0.0
            # 不応期をランダムにして同期発火を防ぐ
            self.refractory[fired_arr] = np.random.uniform(2.0, 5.0, size=len(fired_arr))
            self.trace[fired_arr] += 1.0

        # 安全装置: 閾値が極端にならないようにクリップ
        np.clip(self.dynamic_thresh, 0.1, 5.0, out=self.dynamic_thresh)
        
        # 5. STDP Learning
        if learning and fired_indices:
            # Recurrent Weights Only for now
            for pre_id in prev_active_hidden:
                if pre_id < len(self.rec_indices):
                    targets = self.rec_indices[pre_id]
                    ws = self.rec_weights[pre_id]
                    
                    # LTD: Pre発火 -> Post不発 (一律減衰)
                    ws -= 0.002 
                    
                    # LTP: Pre発火 -> Post発火 (強化)
                    mask = np.isin(targets, fired_arr)
                    if np.any(mask):
                        ws[mask] += 0.03 # LTD分を上回る強化
                        np.clip(ws, -2.0, 2.0, out=ws)
            
        return fired_indices

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.v.copy(), self.dynamic_thresh.copy()

    def reset(self):
        if self.use_rust:
            self.core.reset()
        else:
            self.v.fill(0)
            self.refractory.fill(0)
            self.trace.fill(0)
            self.input_trace.fill(0)
            self.dynamic_thresh.fill(self.base_thresh)
            self.activity_ema.fill(self.target_rate)