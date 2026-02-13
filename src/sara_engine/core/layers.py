import numpy as np
from typing import List, Optional

# Rust Core Import Check
try:
    # 同じパッケージ内の sara_rust_core をインポート
    from .. import sara_rust_core
    RUST_AVAILABLE = True
except ImportError:
    # 開発環境などでパスが通っていない場合のフォールバック
    try:
        import sara_rust_core
        RUST_AVAILABLE = True
    except ImportError:
        RUST_AVAILABLE = False

class DynamicLiquidLayer:
    """
    Hybrid Layer: Rustが利用可能ならRust版を、そうでなければPython版を使用するラッパー。
    誤差逆伝播法を使わず、スパイク通信のみで計算を行う。
    """
    def __init__(self, input_size: int, hidden_size: int, decay: float, 
                 density: float = 0.05, input_scale: float = 1.0, 
                 rec_scale: float = 0.8, feedback_scale: float = 0.5):
        
        self.use_rust = RUST_AVAILABLE
        self.size = hidden_size
        
        if self.use_rust:
            # Rust実装の初期化
            self.core = sara_rust_core.RustLiquidLayer(input_size, hidden_size, decay, density, feedback_scale)
            # 互換性用ダミー属性
            self.in_indices = []
            self.in_weights = []
            self.rec_indices = []
            self.rec_weights = []
            self.dynamic_thresh = [] 
        else:
            # Python実装 (Fallback)
            self.decay = decay
            self.feedback_scale = feedback_scale
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
            self.base_thresh = 1.3 if decay < 0.8 else 1.4
            self.dynamic_thresh = np.ones(hidden_size, dtype=np.float32) * self.base_thresh
            
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
        self.refractory = np.maximum(0, self.refractory - 1)
        self.v *= self.decay
        
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
        
        ready_mask = (self.v >= self.dynamic_thresh) & (self.refractory <= 0)
        candidates_indices = np.where(ready_mask)[0]
        
        fired_indices = []
        max_spikes = int(self.size * 0.10)
        
        if len(candidates_indices) > 0:
            if len(candidates_indices) > max_spikes:
                potentials = self.v[candidates_indices]
                top_indices = np.argsort(potentials)[-max_spikes:]
                fired_indices = candidates_indices[top_indices].tolist()
                excess_ratio = len(candidates_indices) / max_spikes
                penalty = 0.05 * np.log(excess_ratio)
                self.dynamic_thresh[fired_indices] += penalty
            else:
                fired_indices = candidates_indices.tolist()
            
            fired_arr = np.array(fired_indices, dtype=int)
            self.v[fired_arr] = 0.0
            ref_periods = np.random.uniform(2.0, 5.0, size=len(fired_arr))
            self.refractory[fired_arr] = ref_periods
            self.dynamic_thresh[fired_arr] += 0.03

        not_fired_mask = np.ones(self.size, dtype=bool)
        if fired_indices: not_fired_mask[np.array(fired_indices, dtype=int)] = False
        
        self.dynamic_thresh[not_fired_mask] -= 0.005
        np.clip(self.dynamic_thresh, 0.3, 5.0, out=self.dynamic_thresh)
        
        if learning and fired_indices and prev_active_hidden:
            fired_arr = np.array(fired_indices, dtype=int)
            for pre_id in prev_active_hidden:
                if pre_id < len(self.rec_indices):
                    targets = self.rec_indices[pre_id]
                    ws = self.rec_weights[pre_id]
                    # 安全装置
                    if len(targets) != len(ws):
                        min_len = min(len(targets), len(ws))
                        targets = targets[:min_len]
                        ws = ws[:min_len]
                    mask = np.isin(targets, fired_arr)
                    if np.any(mask):
                        ws[mask] += 0.01
                        np.clip(ws, -2.0, 2.0, out=ws)
        return fired_indices

    def reset(self):
        if self.use_rust:
            self.core.reset()
        else:
            self.v.fill(0)
            self.refractory.fill(0)
            self.dynamic_thresh.fill(self.base_thresh)