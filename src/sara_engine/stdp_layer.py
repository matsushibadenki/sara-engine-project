# src/sara_engine/stdp_layer.py
# title: STDP Enhanced Layer & Engine
# description: ロードマップに基づき、トレースベースのSTDP学習則と教師なし学習機能を実装したモジュール。

import numpy as np
import pickle
from typing import List, Tuple, Optional

class STDPLiquidLayer:
    """
    STDP (Spike-Timing-Dependent Plasticity) を実装したリザーバ層。
    生物学的な学習則により、シナプス結合強度を自己組織化します。
    """
    
    def __init__(self, input_size: int, hidden_size: int, 
                 decay: float, input_scale: float, rec_scale: float, 
                 density: float = 0.1,
                 stdp_enabled: bool = True):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.decay = decay
        self.stdp_enabled = stdp_enabled
        
        # スパース接続 (Adjacency List方式で行列演算を回避)
        self.in_indices: List[np.ndarray] = []
        self.in_weights: List[np.ndarray] = []
        self.rec_indices: List[np.ndarray] = []
        self.rec_weights: List[np.ndarray] = []
        
        # --- 初期化 ---
        # Input Weights
        fan_in = max(1, int(input_size * density))
        w_range_in = input_scale * np.sqrt(2.0 / fan_in)
        
        for i in range(input_size):
            n = int(hidden_size * density)
            if n > 0:
                idx = np.random.choice(hidden_size, n, replace=False).astype(np.int32)
                w = np.random.normal(0, w_range_in, n).astype(np.float32)
                self.in_indices.append(idx)
                self.in_weights.append(w)
            else:
                self.in_indices.append(np.array([], dtype=np.int32))
                self.in_weights.append(np.array([], dtype=np.float32))
        
        # Recurrent Weights
        rec_density = 0.12
        fan_in_rec = max(1, int(hidden_size * rec_density))
        w_range_rec = rec_scale / np.sqrt(fan_in_rec)
        
        for i in range(hidden_size):
            n = int(hidden_size * rec_density)
            if n > 0:
                idx = np.random.choice(hidden_size, n, replace=False).astype(np.int32)
                w = np.random.normal(0, w_range_rec, n).astype(np.float32)
                self.rec_indices.append(idx)
                self.rec_weights.append(w)
            else:
                self.rec_indices.append(np.array([], dtype=np.int32))
                self.rec_weights.append(np.array([], dtype=np.float32))
        
        # ニューロンの状態
        self.v = np.zeros(hidden_size, dtype=np.float32)
        self.refractory = np.zeros(hidden_size, dtype=np.float32)
        
        # 適応的閾値
        self.base_thresh = 0.7 if decay < 0.8 else 0.8
        self.thresh = np.ones(hidden_size, dtype=np.float32) * self.base_thresh
        self.target_rate = 0.03
        self.refractory_period = 2.0
        
        # --- STDP パラメータ (ロードマップ準拠) ---
        if self.stdp_enabled:
            self.a_plus = 0.008   # LTP (Long-Term Potentiation)
            self.a_minus = 0.009  # LTD (Long-Term Depression)
            self.tau_plus = 20.0  # ms
            self.tau_minus = 20.0 # ms
            
            # シナプストレース (発火の痕跡)
            self.trace_pre = np.zeros(input_size, dtype=np.float32)
            self.trace_post = np.zeros(hidden_size, dtype=np.float32)
            
            # リカレント用トレース
            self.trace_rec_pre = np.zeros(hidden_size, dtype=np.float32)
    
    def reset(self):
        """状態のリセット"""
        self.v.fill(0)
        self.refractory.fill(0)
        if self.stdp_enabled:
            self.trace_pre.fill(0)
            self.trace_post.fill(0)
            self.trace_rec_pre.fill(0)
    
    def update_homeostasis(self, activity_history: np.ndarray, steps: int):
        """ホメオスタシス：目標発火率に近づくように閾値を調整"""
        if steps == 0: return
        rate = activity_history / float(steps)
        diff = rate - self.target_rate
        # 発火しすぎなら閾値を上げ、しなさすぎなら下げる
        gain = 0.1 if np.max(np.abs(diff)) > 0.1 else 0.02
        self.thresh += gain * diff
        self.thresh = np.clip(self.thresh, self.base_thresh * 0.5, self.base_thresh * 5.0)
    
    def forward(self, active_inputs: List[int], prev_active_hidden: List[int], 
                dt: float = 1.0, learning: bool = True) -> List[int]:
        """
        前向き計算とSTDP学習
        
        Args:
            active_inputs: 入力層の発火インデックス
            prev_active_hidden: 直前の隠れ層の発火インデックス
            dt: 時間刻み
            learning: 学習を行うかどうか
        """
        # 1. 不応期の更新
        self.refractory = np.maximum(0, self.refractory - 1)
        
        # 2. 膜電位の自然減衰 (Leak)
        self.v *= self.decay
        
        # 3. STDPトレースの更新 (指数減衰)
        if self.stdp_enabled:
            decay_plus = np.exp(-dt / self.tau_plus)
            decay_minus = np.exp(-dt / self.tau_minus)
            self.trace_pre *= decay_plus
            self.trace_post *= decay_minus
            self.trace_rec_pre *= decay_plus
            
            # 入力が来た時点でのトレースを加算
            if active_inputs:
                self.trace_pre[active_inputs] += 1.0
            if prev_active_hidden:
                self.trace_rec_pre[prev_active_hidden] += 1.0

        # 4. シナプス入力の統合 (Sparse Operation)
        # Input -> Hidden
        for pre_id in active_inputs:
            if pre_id < len(self.in_indices):
                targets = self.in_indices[pre_id]
                ws = self.in_weights[pre_id]
                if len(targets) > 0:
                    self.v[targets] += ws
        
        # Recurrent (Hidden -> Hidden)
        for pre_h_id in prev_active_hidden:
            if pre_h_id < len(self.rec_indices):
                targets = self.rec_indices[pre_h_id]
                ws = self.rec_weights[pre_h_id]
                if len(targets) > 0:
                    self.v[targets] += ws
        
        # 5. 発火判定
        ready_mask = (self.v >= self.thresh) & (self.refractory <= 0)
        fired_indices = np.where(ready_mask)[0].tolist()
        
        if fired_indices:
            # リセットと不応期設定
            self.v[fired_indices] -= self.thresh[fired_indices] # Soft reset
            self.v = np.maximum(self.v, 0.0)
            self.refractory[fired_indices] = self.refractory_period
            
            # Postトレースの更新（発火直後）
            if self.stdp_enabled:
                self.trace_post[fired_indices] += 1.0

        # 6. STDP学習則の適用 (Weight Update)
        if self.stdp_enabled and learning:
            # Input Weights Update
            # LTP: Pre(trace) -> Post(fire)
            # LTD: Post(trace) -> Pre(fire) ... 今回はPre発火時のイベント駆動ではなく、Post発火時にまとめて近似計算
            
            if fired_indices:
                fired_arr = np.array(fired_indices, dtype=int)
                
                # 全入力ニューロンに対して更新するのは重いため、
                # 「今回発火したニューロン(Post)」に接続しているシナプスのみ更新
                
                # ここでは簡易的に、今回入力があったPreと、今回発火したPostの間で計算
                # LTP: Pre Trace High + Post Fired
                for pre_id in active_inputs: # Active Pre
                    if pre_id < len(self.in_indices):
                        targets = self.in_indices[pre_id]
                        # ターゲットのうち、今回発火したもの(Post)を強化
                        mask = np.isin(targets, fired_arr)
                        if np.any(mask):
                            self.in_weights[pre_id][mask] += self.a_plus * self.trace_pre[pre_id]
                            # 重みクリッピング
                            np.clip(self.in_weights[pre_id], -3.0, 3.0, out=self.in_weights[pre_id])

            # LTD: Post Trace High + Pre Fired (Inverse logic approximated)
            # 簡略化のため、Hebb則的な強化を中心に実装し、正規化で抑制を行うアプローチを採用
            # または、Postが発火したときに、Preのトレースが低いものを弱める
            
            # Recurrent Weights Update
            if fired_indices:
                 for pre_h_id in prev_active_hidden:
                    if pre_h_id < len(self.rec_indices):
                        targets = self.rec_indices[pre_h_id]
                        mask = np.isin(targets, fired_arr)
                        if np.any(mask):
                            # LTP
                            self.rec_weights[pre_h_id][mask] += self.a_plus * self.trace_rec_pre[pre_h_id]
                            np.clip(self.rec_weights[pre_h_id], -2.0, 2.0, out=self.rec_weights[pre_h_id])

        return fired_indices


class STDPSaraEngine:
    """
    ロードマップ Pillar 1 対応: STDP搭載型SARAエンジン
    教師なし事前学習（Pretraining）と教師あり微調整（Fine-tuning）をサポート。
    """
    
    def __init__(self, input_size: int, output_size: int, load_path: Optional[str] = None):
        if load_path:
            self.load_model(load_path)
            return

        self.input_size = input_size
        self.output_size = output_size
        
        # 3層のSTDPリザーバ
        self.reservoirs = [
            STDPLiquidLayer(input_size, 1200, decay=0.25, input_scale=1.2, rec_scale=1.3),
            STDPLiquidLayer(input_size, 1800, decay=0.5, input_scale=1.0, rec_scale=1.6),
            STDPLiquidLayer(input_size, 1800, decay=0.75, input_scale=0.7, rec_scale=1.8),
        ]
        
        self.offsets = [0, 1200, 3000]
        self.total_hidden = sum(r.hidden_size for r in self.reservoirs)
        
        # Readout Layer (Supervised)
        self.w_ho = []
        for _ in range(output_size):
            # He初期化
            limit = np.sqrt(2.0 / self.total_hidden)
            w = np.random.normal(0, limit, self.total_hidden).astype(np.float32)
            self.w_ho.append(w)
            
        self.o_v = np.zeros(output_size, dtype=np.float32)
        
        # 状態変数
        # 修正: 型ヒント追加
        self.prev_spikes: List[List[int]] = [[] for _ in self.reservoirs]
        self.layer_activity_counters = [np.zeros(r.hidden_size) for r in self.reservoirs]
        self.t = 0
        
        # 学習パラメータ
        self.lr = 0.002
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.m_ho = [np.zeros_like(w) for w in self.w_ho]
        self.v_ho = [np.zeros_like(w) for w in self.w_ho]

    def reset_state(self):
        for r in self.reservoirs: r.reset()
        self.prev_spikes = [[] for _ in self.reservoirs]
        self.o_v.fill(0)
        self.layer_activity_counters = [np.zeros(r.hidden_size) for r in self.reservoirs]

    def pretrain(self, spike_trains: List[List[List[int]]], epochs: int = 1):
        """
        教師なし事前学習を実行する。
        STDPのみを使って、リザーバ内の特徴抽出能力を高める。
        ラベル情報は使用しない。
        """
        print(f"Starting Unsupervised STDP Pretraining ({epochs} epochs)...")
        total = len(spike_trains)
        
        for epoch in range(epochs):
            for i, train_seq in enumerate(spike_trains):
                self.reset_state()
                steps = len(train_seq)
                
                for input_spikes in train_seq:
                    # 各層でSTDP有効
                    for j, r in enumerate(self.reservoirs):
                        local_spikes = r.forward(input_spikes, self.prev_spikes[j], learning=True)
                        self.prev_spikes[j] = local_spikes
                        
                        if local_spikes:
                            self.layer_activity_counters[j][local_spikes] += 1.0
                
                # ホメオスタシス更新
                for j, r in enumerate(self.reservoirs):
                    r.update_homeostasis(self.layer_activity_counters[j], steps)
                
                if (i+1) % 100 == 0:
                    print(f"  Pretrain Epoch {epoch+1}: {i+1}/{total} samples processed.", end='\r')
        print("\nPretraining Complete.")

    def train_step(self, spike_train: List[List[int]], target_label: int):
        """教師あり学習ステップ（Readout層のみ学習、リザーバは固定または微調整）"""
        self.reset_state()
        self.t += 1
        
        # Forward pass
        all_hidden_spikes_history = []
        steps = len(spike_train)
        
        # リザーバのSTDPは、教師ありフェーズではオフにするか、弱める（ここではオフ）
        for input_spikes in spike_train:
            current_step_spikes = []
            for j, r in enumerate(self.reservoirs):
                local_spikes = r.forward(input_spikes, self.prev_spikes[j], learning=False)
                self.prev_spikes[j] = local_spikes
                
                if local_spikes:
                    base = self.offsets[j]
                    current_step_spikes.extend([x + base for x in local_spikes])
            
            all_hidden_spikes_history.append(current_step_spikes)
            
            # Readout積分
            self.o_v *= 0.9 # Readout decay
            if current_step_spikes:
                for o in range(self.output_size):
                    # Sparse sum
                    self.o_v[o] += np.sum(self.w_ho[o][current_step_spikes]) * 0.1

        # Error Calculation (Margin Ranking Loss風)
        grad_accum = [np.zeros_like(w) for w in self.w_ho]
        
        # ターゲットは高く、他は低く
        err_target = 0.0
        if self.o_v[target_label] < 2.0:
            err_target = 2.0 - self.o_v[target_label]
        
        # 誤差逆伝播は使わない。Delta則をスパイク履歴全体に適用
        for t_spikes in all_hidden_spikes_history:
            if not t_spikes: continue
            
            # Target neuron update (LTP)
            if err_target > 0:
                 grad_accum[target_label][t_spikes] += err_target * 0.05
            
            # Other neurons update (LTD)
            for o in range(self.output_size):
                if o != target_label and self.o_v[o] > -0.5:
                    err_other = -0.5 - self.o_v[o]
                    grad_accum[o][t_spikes] += err_other * 0.05

        # Adam Update (Matrix free logic)
        lr_t = self.lr * (1.0 / (1.0 + 0.0001 * self.t))
        epsilon = 1e-8
        
        for o in range(self.output_size):
            grad = grad_accum[o]
            self.m_ho[o] = self.beta1 * self.m_ho[o] + (1 - self.beta1) * grad
            self.v_ho[o] = self.beta2 * self.v_ho[o] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m_ho[o] / (1 - self.beta1 ** min(self.t, 1000))
            v_hat = self.v_ho[o] / (1 - self.beta2 ** min(self.t, 1000))
            
            update = lr_t * m_hat / (np.sqrt(v_hat) + epsilon)
            self.w_ho[o] += update
            np.clip(self.w_ho[o], -5.0, 5.0, out=self.w_ho[o])

    def predict(self, spike_train: List[List[int]]) -> int:
        self.reset_state()
        potentials = np.zeros(self.output_size)
        
        for input_spikes in spike_train:
            step_spikes = []
            for j, r in enumerate(self.reservoirs):
                local_spikes = r.forward(input_spikes, self.prev_spikes[j], learning=False)
                self.prev_spikes[j] = local_spikes
                if local_spikes:
                    base = self.offsets[j]
                    step_spikes.extend([x + base for x in local_spikes])
            
            potentials *= 0.9
            if step_spikes:
                for o in range(self.output_size):
                    potentials[o] += np.sum(self.w_ho[o][step_spikes]) * 0.1
                    
        return int(np.argmax(potentials))

    def save_model(self, filepath: str):
        data = {
            'reservoirs': self.reservoirs,
            'w_ho': self.w_ho,
            'm_ho': self.m_ho,
            'v_ho': self.v_ho
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.reservoirs = data['reservoirs']
        self.w_ho = data['w_ho']
        self.m_ho = data['m_ho']
        self.v_ho = data['v_ho']
        self.total_hidden = sum(r.hidden_size for r in self.reservoirs)
        print(f"Model loaded from {filepath}")