{
    "//": "ディレクトリパス: src/snn_models/spatiotemporal_stdp.py",
    "//": "タイトル: 階層的・再帰的SNN向け 全層STDP学習モジュール",
    "//": "目的: NeuroFEMからの熱変動を感覚野と皮質野を通して学習する。mypyの相対インポートエラーを解決するため、絶対インポートを利用する。"
}

import math
import random
from typing import List, Optional

try:
    from sara_engine import sara_rust_core  # type: ignore
    RUST_AVAILABLE = True
except ImportError:
    try:
        import sara_rust_core  # type: ignore
        RUST_AVAILABLE = True
    except ImportError:
        RUST_AVAILABLE = False

class SpatioTemporalSNN:
    def __init__(self, n_in: int = 201, n_sensory: int = 50, n_cortex: int = 20, dt: float = 1.0, use_rust: Optional[bool] = None):
        self.n_in = n_in
        self.n_sensory = n_sensory
        self.n_cortex = n_cortex
        self.dt = dt
        
        # Rust拡張の自動フォールバック判定
        if use_rust is None:
            self.use_rust = RUST_AVAILABLE
        else:
            self.use_rust = use_rust and RUST_AVAILABLE

        # 他のファイルへの影響を抑えるため、Rust側の実装有無を動的にチェックする
        if self.use_rust and hasattr(sara_rust_core, 'RustSpatioTemporalSNN'):
            print("SpatioTemporalSNN: Rust core initialized.")
            self.core = sara_rust_core.RustSpatioTemporalSNN(n_in, n_sensory, n_cortex, dt)
        else:
            self.use_rust = False
            print("SpatioTemporalSNN: Python fallback mode.")
            
            # ニューロンの膜電位と発火フラグ
            self.v_sensory = [0.0 for _ in range(n_sensory)]
            self.spike_sensory = [False for _ in range(n_sensory)]
            self.v_cortex = [0.0 for _ in range(n_cortex)]
            self.spike_cortex = [False for _ in range(n_cortex)]
            
            # LIFニューロンのパラメータ
            self.v_rest = 0.0
            self.v_thresh = 0.45 
            self.v_reset = 0.0
            self.tau_m = 10.0
            
            # STDPのトレース
            self.trace_in = [0.0 for _ in range(n_in)]
            self.trace_sensory = [0.0 for _ in range(n_sensory)]
            self.trace_cortex = [0.0 for _ in range(n_cortex)]
            self.tau_trace = 20.0
            
            # STDP学習率と境界パラメータ
            self.A_plus = 0.01  
            self.A_minus = 0.01 
            self.w_max = 2.0
            self.w_decay = 0.0001 
            
            # --- 結合と重みの初期化 ---
            
            # 1. Input -> Sensory (局所受容野)
            self.W_in_sens = [[0.0 for _ in range(n_sensory)] for _ in range(n_in)]
            self.sensory_conn_pre = [[] for _ in range(n_sensory)] 
            self.sensory_conn_post = [[] for _ in range(n_in)]     
            
            rf_size = 12 
            for j in range(n_sensory):
                start_idx = int(j * (n_in - rf_size) / max(1, n_sensory - 1))
                for i in range(start_idx, min(n_in, start_idx + rf_size)):
                    self.W_in_sens[i][j] = random.uniform(0.1, 0.5)
                    self.sensory_conn_pre[j].append(i)
                    self.sensory_conn_post[i].append(j)
                    
            # 2. Sensory -> Cortex (全結合)
            self.W_sens_ctx = [[random.uniform(0.1, 0.5) for _ in range(n_cortex)] for _ in range(n_sensory)]
            
            # 3. Cortex -> Cortex (リカレント結合)
            self.W_ctx_ctx = [[random.uniform(0.0, 0.3) if i != j else 0.0 for j in range(n_cortex)] for i in range(n_cortex)]

    def step(self, heat_data: List[float]):
        if self.use_rust:
            self.core.step(heat_data)
            return

        spike_in = [False for _ in range(self.n_in)]
        
        # --- 1. 入力層の発火判定とトレース更新 ---
        for i in range(self.n_in):
            p_fire = min(1.0, heat_data[i] * 2.0)
            if random.random() < p_fire:
                spike_in[i] = True
                self.trace_in[i] += 1.0 
                
        decay_factor = math.exp(-self.dt / self.tau_trace)
        for i in range(self.n_in):
            self.trace_in[i] *= decay_factor
            
        # --- 2. Sensory Layer の膜電位更新と発火判定 ---
        for j in range(self.n_sensory):
            I_syn = 0.0
            for i in self.sensory_conn_pre[j]:
                if spike_in[i]:
                    I_syn += self.W_in_sens[i][j]
                    
            self.v_sensory[j] += (-(self.v_sensory[j] - self.v_rest) + I_syn) * (self.dt / self.tau_m)
            
            if self.v_sensory[j] >= self.v_thresh:
                self.v_sensory[j] = self.v_reset
                self.spike_sensory[j] = True
                self.trace_sensory[j] += 1.0
            else:
                self.spike_sensory[j] = False
                
        for j in range(self.n_sensory):
            self.trace_sensory[j] *= decay_factor
            
        # --- 3. Cortex Layer の膜電位更新と発火判定 ---
        I_syn_ctx = [0.0 for _ in range(self.n_cortex)]
        for j in range(self.n_cortex):
            for i in range(self.n_sensory):
                if self.spike_sensory[i]:
                    I_syn_ctx[j] += self.W_sens_ctx[i][j]
                    
            for k in range(self.n_cortex):
                if self.spike_cortex[k]:
                    I_syn_ctx[j] += self.W_ctx_ctx[k][j]
                    
        for j in range(self.n_cortex):
            self.v_cortex[j] += (-(self.v_cortex[j] - self.v_rest) + I_syn_ctx[j]) * (self.dt / self.tau_m)
            if self.v_cortex[j] >= self.v_thresh:
                self.v_cortex[j] = self.v_reset
                self.spike_cortex[j] = True
                self.trace_cortex[j] += 1.0
            else:
                self.spike_cortex[j] = False
                
        for j in range(self.n_cortex):
            self.trace_cortex[j] *= decay_factor

        # --- 4. 全層STDPによる自己組織化 ---
        self._update_stdp_in_sens(spike_in)
        self._update_stdp_sens_ctx()
        self._update_stdp_ctx_ctx()

    def _update_stdp_in_sens(self, spike_in: List[bool]):
        for j in range(self.n_sensory):
            if self.spike_sensory[j]: 
                for i in self.sensory_conn_pre[j]:
                    self.W_in_sens[i][j] += self.A_plus * self.trace_in[i]
                    if self.W_in_sens[i][j] > self.w_max:
                        self.W_in_sens[i][j] = self.w_max
                        
        for i in range(self.n_in):
            if spike_in[i]: 
                for j in self.sensory_conn_post[i]:
                    self.W_in_sens[i][j] -= self.A_minus * self.trace_sensory[j]
                    if self.W_in_sens[i][j] < 0.0:
                        self.W_in_sens[i][j] = 0.0

    def _update_stdp_sens_ctx(self):
        for j in range(self.n_cortex):
            if self.spike_cortex[j]: 
                for i in range(self.n_sensory):
                    self.W_sens_ctx[i][j] += self.A_plus * self.trace_sensory[i]
                    if self.W_sens_ctx[i][j] > self.w_max:
                        self.W_sens_ctx[i][j] = self.w_max
                        
        for i in range(self.n_sensory):
            if self.spike_sensory[i]: 
                for j in range(self.n_cortex):
                    self.W_sens_ctx[i][j] -= self.A_minus * self.trace_cortex[j]
                    if self.W_sens_ctx[i][j] < 0.0:
                        self.W_sens_ctx[i][j] = 0.0

    def _update_stdp_ctx_ctx(self):
        for j in range(self.n_cortex):
            if self.spike_cortex[j]:
                for i in range(self.n_cortex):
                    if i != j:
                        self.W_ctx_ctx[i][j] += self.A_plus * self.trace_cortex[i]
                        self.W_ctx_ctx[i][j] -= self.w_decay * self.W_ctx_ctx[i][j]
                        if self.W_ctx_ctx[i][j] > self.w_max:
                            self.W_ctx_ctx[i][j] = self.w_max
                            
        for i in range(self.n_cortex):
            if self.spike_cortex[i]:
                for j in range(self.n_cortex):
                    if i != j:
                        self.W_ctx_ctx[i][j] -= self.A_minus * self.trace_cortex[j]
                        if self.W_ctx_ctx[i][j] < 0.0:
                            self.W_ctx_ctx[i][j] = 0.0