# [配置するディレクトリのパス]: ./src/snn_models/spatiotemporal_stdp.py
# [ファイルの日本語タイトル]: 階層的・再帰的SNN向け 全層STDP学習モジュール
# [ファイルの目的や内容]: 
# NeuroFEMからの201ノードの時系列熱変動を感覚野（Sensory Layer: 局所特徴抽出）と
# 皮質野（Cortex: リカレント結合）を通して学習する。
# GPUや行列演算、誤差逆伝播法を一切使用せず、純粋なPythonのループ処理と
# リスト内包表記のみでSTDP（スパイクタイミング依存可塑性）を実装する。

import math
import random

class SpatioTemporalSNN:
    def __init__(self, n_in=201, n_sensory=50, n_cortex=20, dt=1.0):
        self.n_in = n_in
        self.n_sensory = n_sensory
        self.n_cortex = n_cortex
        self.dt = dt
        
        # ニューロンの膜電位と発火フラグ
        self.v_sensory = [0.0 for _ in range(n_sensory)]
        self.spike_sensory = [False for _ in range(n_sensory)]
        self.v_cortex = [0.0 for _ in range(n_cortex)]
        self.spike_cortex = [False for _ in range(n_cortex)]
        
        # LIFニューロンのパラメータ（発火しやすくするため閾値を1.0から0.45へ修正）
        self.v_rest = 0.0
        self.v_thresh = 0.45 
        self.v_reset = 0.0
        self.tau_m = 10.0
        
        # STDPのトレース（発火履歴：行列の代わりに1次元リストで管理）
        self.trace_in = [0.0 for _ in range(n_in)]
        self.trace_sensory = [0.0 for _ in range(n_sensory)]
        self.trace_cortex = [0.0 for _ in range(n_cortex)]
        self.tau_trace = 20.0
        
        # STDP学習率と境界パラメータ
        self.A_plus = 0.01  # LTP (Long-Term Potentiation)
        self.A_minus = 0.01 # LTD (Long-Term Depression)
        self.w_max = 2.0
        self.w_decay = 0.0001 # 恒常性維持（過剰なリカレントループを防ぐ微小減衰）
        
        # --- 結合と重みの初期化 ---
        
        # 1. Input -> Sensory (局所受容野: CNNライクな結合)
        self.W_in_sens = [[0.0 for _ in range(n_sensory)] for _ in range(n_in)]
        self.sensory_conn_pre = [[] for _ in range(n_sensory)] # 感覚野ノードが参照する入力インデックス
        self.sensory_conn_post = [[] for _ in range(n_in)]     # 入力ノードが接続する感覚野インデックス
        
        rf_size = 12 # 受容野のサイズ
        for j in range(n_sensory):
            # 201ノードを均等にカバーするストライド計算
            start_idx = int(j * (n_in - rf_size) / max(1, n_sensory - 1))
            for i in range(start_idx, min(n_in, start_idx + rf_size)):
                self.W_in_sens[i][j] = random.uniform(0.1, 0.5)
                self.sensory_conn_pre[j].append(i)
                self.sensory_conn_post[i].append(j)
                
        # 2. Sensory -> Cortex (全結合)
        self.W_sens_ctx = [[random.uniform(0.1, 0.5) for _ in range(n_cortex)] for _ in range(n_sensory)]
        
        # 3. Cortex -> Cortex (リカレント結合: 時系列学習用)
        # 自身への結合（i == j）は持たせない
        self.W_ctx_ctx = [[random.uniform(0.0, 0.3) if i != j else 0.0 for j in range(n_cortex)] for i in range(n_cortex)]

    def step(self, heat_data):
        # heat_data: 毎ステップ変化するNeuroFEMの熱分布
        spike_in = [False for _ in range(self.n_in)]
        
        # --- 1. 入力層の発火判定とトレース更新（ポアソンエンコーディング） ---
        for i in range(self.n_in):
            # 修正: 発火確率の底上げ（ゲイン2.0を乗算して入力信号を強化）
            p_fire = min(1.0, heat_data[i] * 2.0)
            if random.random() < p_fire:
                spike_in[i] = True
                self.trace_in[i] += 1.0 # プレシナプストレース上昇
                
        decay_factor = math.exp(-self.dt / self.tau_trace)
        for i in range(self.n_in):
            self.trace_in[i] *= decay_factor
            
        # --- 2. Sensory Layer の膜電位更新と発火判定 ---
        for j in range(self.n_sensory):
            # シナプス入力電流の計算（全探索を避け、接続のある受容野のみループ）
            I_syn = 0.0
            for i in self.sensory_conn_pre[j]:
                if spike_in[i]:
                    I_syn += self.W_in_sens[i][j]
                    
            # LIFモデルの膜電位積分
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
            # 感覚野からの順伝播入力
            for i in range(self.n_sensory):
                if self.spike_sensory[i]:
                    I_syn_ctx[j] += self.W_sens_ctx[i][j]
                    
            # 皮質野のリカレント入力（前ステップの発火状態を利用して時系列を捉える）
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

        # --- 4. 全層STDPによる自己組織化（行列計算に依存しない純粋なループ処理） ---
        self._update_stdp_in_sens(spike_in)
        self._update_stdp_sens_ctx()
        self._update_stdp_ctx_ctx()

    def _update_stdp_in_sens(self, spike_in):
        # Input -> Sensory のSTDP
        for j in range(self.n_sensory):
            if self.spike_sensory[j]: # ポストニューロン発火：LTP
                for i in self.sensory_conn_pre[j]:
                    self.W_in_sens[i][j] += self.A_plus * self.trace_in[i]
                    if self.W_in_sens[i][j] > self.w_max:
                        self.W_in_sens[i][j] = self.w_max
                        
        for i in range(self.n_in):
            if spike_in[i]: # プレニューロン発火：LTD
                for j in self.sensory_conn_post[i]:
                    self.W_in_sens[i][j] -= self.A_minus * self.trace_sensory[j]
                    if self.W_in_sens[i][j] < 0.0:
                        self.W_in_sens[i][j] = 0.0

    def _update_stdp_sens_ctx(self):
        # Sensory -> Cortex のSTDP
        for j in range(self.n_cortex):
            if self.spike_cortex[j]: # ポスト発火: LTP
                for i in range(self.n_sensory):
                    self.W_sens_ctx[i][j] += self.A_plus * self.trace_sensory[i]
                    if self.W_sens_ctx[i][j] > self.w_max:
                        self.W_sens_ctx[i][j] = self.w_max
                        
        for i in range(self.n_sensory):
            if self.spike_sensory[i]: # プレ発火: LTD
                for j in range(self.n_cortex):
                    self.W_sens_ctx[i][j] -= self.A_minus * self.trace_cortex[j]
                    if self.W_sens_ctx[i][j] < 0.0:
                        self.W_sens_ctx[i][j] = 0.0

    def _update_stdp_ctx_ctx(self):
        # Cortex -> Cortex (リカレント) のSTDP
        for j in range(self.n_cortex):
            if self.spike_cortex[j]:
                for i in range(self.n_cortex):
                    if i != j:
                        # LTP + 恒常性維持のための重み微小減衰
                        self.W_ctx_ctx[i][j] += self.A_plus * self.trace_cortex[i]
                        self.W_ctx_ctx[i][j] -= self.w_decay * self.W_ctx_ctx[i][j]
                        if self.W_ctx_ctx[i][j] > self.w_max:
                            self.W_ctx_ctx[i][j] = self.w_max
                            
        for i in range(self.n_cortex):
            if self.spike_cortex[i]:
                for j in range(self.n_cortex):
                    if i != j:
                        # LTD
                        self.W_ctx_ctx[i][j] -= self.A_minus * self.trace_cortex[j]
                        if self.W_ctx_ctx[i][j] < 0.0:
                            self.W_ctx_ctx[i][j] = 0.0