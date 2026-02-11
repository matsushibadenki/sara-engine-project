# src/sara_engine/stdp_layer.py
# STDP (Spike-Timing-Dependent Plasticity) 実装
# 生物学的な学習則を追加したレイヤー

import numpy as np
from typing import List, Tuple

class STDPLiquidLayer:
    """
    STDP学習則を持つLiquid State Machine層
    
    STDP: スパイクのタイミングに依存した可塑性
    - プレシナプスの発火後にポストシナプスが発火 → LTP (強化)
    - ポストシナプスの発火後にプレシナプスが発火 → LTD (弱化)
    """
    
    def __init__(self, input_size: int, hidden_size: int, 
                 decay: float, input_scale: float, rec_scale: float, 
                 density: float = 0.1,
                 stdp_enabled: bool = True):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.decay = decay
        self.stdp_enabled = stdp_enabled
        
        # スパース接続の初期化
        self.in_indices = []
        self.in_weights = []
        self.rec_indices = []
        self.rec_weights = []
        
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
        
        # 閾値設定
        if decay < 0.4:
            self.base_thresh = 0.6
            self.target_rate = 0.025
            self.refractory_period = 2.5
        elif decay < 0.8:
            self.base_thresh = 0.7
            self.target_rate = 0.035
            self.refractory_period = 2.0
        else:
            self.base_thresh = 0.8
            self.target_rate = 0.025
            self.refractory_period = 1.5
        
        self.thresh = np.ones(hidden_size, dtype=np.float32) * self.base_thresh
        
        # STDPパラメータ
        if self.stdp_enabled:
            self.a_plus = 0.005   # LTP学習率
            self.a_minus = 0.006  # LTD学習率
            self.tau_plus = 20.0  # LTP時定数
            self.tau_minus = 20.0 # LTD時定数
            
            # スパイクトレース（プレ・ポストシナプスの履歴）
            self.trace_pre = np.zeros(input_size, dtype=np.float32)
            self.trace_post = np.zeros(hidden_size, dtype=np.float32)
            self.trace_rec = np.zeros(hidden_size, dtype=np.float32)
    
    def reset(self):
        """状態のリセット"""
        self.v.fill(0)
        self.refractory.fill(0)
        if self.stdp_enabled:
            self.trace_pre.fill(0)
            self.trace_post.fill(0)
            self.trace_rec.fill(0)
    
    def update_homeostasis(self, activity_history: np.ndarray, steps: int):
        """ホメオスタシス（発火率の調整）"""
        if steps == 0:
            return
        rate = activity_history / float(steps)
        diff = rate - self.target_rate
        gain = np.where(np.abs(diff) > 0.08, 0.2, 0.05)
        self.thresh += gain * diff
        self.thresh = np.clip(self.thresh, self.base_thresh * 0.4, self.base_thresh * 6.0)
    
    def update_stdp_traces(self, active_inputs: List[int], fired_indices: List[int], dt: float = 1.0):
        """STDPトレースの更新"""
        if not self.stdp_enabled:
            return
        
        # トレースの減衰
        self.trace_pre *= np.exp(-dt / self.tau_plus)
        self.trace_post *= np.exp(-dt / self.tau_minus)
        self.trace_rec *= np.exp(-dt / self.tau_minus)
        
        # 発火したニューロンのトレースを増加
        if active_inputs:
            self.trace_pre[active_inputs] += 1.0
        
        if fired_indices:
            self.trace_post[fired_indices] += 1.0
            self.trace_rec[fired_indices] += 1.0
    
    def apply_stdp(self, active_inputs: List[int], fired_indices: List[int]):
        """STDP学習則の適用"""
        if not self.stdp_enabled or not fired_indices:
            return
        
        # Input → Hidden の STDP
        for pre_id in active_inputs:
            if pre_id >= len(self.in_indices):
                continue
            
            targets = self.in_indices[pre_id]
            if len(targets) == 0:
                continue
            
            # LTP: プレが発火 → ポストが発火していた
            # ポストのトレースが高い = 最近発火していた
            ltp_update = self.a_plus * self.trace_post[targets]
            
            # LTD: ポストが発火 → プレが発火していた
            # プレのトレースが高い = 最近発火していた
            # (fired_indicesに含まれるターゲットのみ)
            ltd_mask = np.isin(targets, fired_indices)
            ltd_update = np.zeros_like(ltp_update)
            ltd_update[ltd_mask] = -self.a_minus * self.trace_pre[pre_id]
            
            # 重みの更新
            self.in_weights[pre_id] += ltp_update + ltd_update
            
            # 重みのクリッピング
            self.in_weights[pre_id] = np.clip(self.in_weights[pre_id], -3.0, 3.0)
        
        # Recurrent の STDP（簡易版）
        for post_id in fired_indices:
            if post_id >= len(self.rec_indices):
                continue
            
            targets = self.rec_indices[post_id]
            if len(targets) == 0:
                continue
            
            # リカレント接続のSTDP
            ltp_update = self.a_plus * 0.5 * self.trace_rec[targets]
            self.rec_weights[post_id] += ltp_update
            self.rec_weights[post_id] = np.clip(self.rec_weights[post_id], -2.0, 2.0)
    
    def forward(self, active_inputs: List[int], prev_active_hidden: List[int], 
                apply_stdp: bool = False) -> List[int]:
        """前向き計算"""
        # 不応期の減少
        self.refractory = np.maximum(0, self.refractory - 1)
        
        # 膜電位の減衰
        self.v *= self.decay
        
        # 入力からの電流
        for pre_id in active_inputs:
            if pre_id < len(self.in_indices):
                targets = self.in_indices[pre_id]
                ws = self.in_weights[pre_id]
                if len(targets) > 0:
                    self.v[targets] += ws
        
        # リカレント接続からの電流
        for pre_h_id in prev_active_hidden:
            if pre_h_id < len(self.rec_indices):
                targets = self.rec_indices[pre_h_id]
                ws = self.rec_weights[pre_h_id]
                if len(targets) > 0:
                    self.v[targets] += ws
        
        # 発火判定
        ready_mask = (self.v >= self.thresh) & (self.refractory <= 0)
        fired_indices = np.where(ready_mask)[0].tolist()
        
        # 発火後の処理
        if fired_indices:
            self.v[fired_indices] -= self.thresh[fired_indices]
            self.v = np.maximum(self.v, 0.0)
            self.refractory[fired_indices] = self.refractory_period
        
        # STDPの適用（オプション）
        if apply_stdp:
            self.apply_stdp(active_inputs, fired_indices)
        
        # トレースの更新
        if self.stdp_enabled:
            self.update_stdp_traces(active_inputs, fired_indices)
        
        return fired_indices


class STDPSaraEngine:
    """
    STDP学習を含むSARA Engine
    
    使用方法:
    1. 教師なし事前学習: pretrain() でSTDPのみで特徴抽出
    2. 教師あり微調整: train_step() で出力層を学習
    """
    
    def __init__(self, input_size: int, output_size: int, use_stdp: bool = True):
        self.input_size = input_size
        self.output_size = output_size
        self.use_stdp = use_stdp
        
        # STDPレイヤーの構築
        self.reservoirs = [
            STDPLiquidLayer(input_size, 1200, decay=0.25, input_scale=1.2, 
                           rec_scale=1.3, stdp_enabled=use_stdp),
            STDPLiquidLayer(input_size, 1800, decay=0.5, input_scale=1.0, 
                           rec_scale=1.6, stdp_enabled=use_stdp),
            STDPLiquidLayer(input_size, 1800, decay=0.75, input_scale=0.7, 
                           rec_scale=1.8, stdp_enabled=use_stdp),
            STDPLiquidLayer(input_size, 1200, decay=0.92, input_scale=0.5, 
                           rec_scale=2.2, stdp_enabled=use_stdp),
        ]
        
        self.total_hidden = sum(r.hidden_size for r in self.reservoirs)
        self.offsets = [0, 1200, 3000, 4800]
        
        # 出力層の重み
        self.w_ho = []
        self.m_ho = []
        self.v_ho = []
        
        for _ in range(output_size):
            limit = np.sqrt(2.0 / self.total_hidden)
            w = np.random.normal(0, limit, self.total_hidden).astype(np.float32)
            self.w_ho.append(w)
            self.m_ho.append(np.zeros(self.total_hidden, dtype=np.float32))
            self.v_ho.append(np.zeros(self.total_hidden, dtype=np.float32))
        
        self.o_v = np.zeros(output_size, dtype=np.float32)
        self.lr = 0.0015
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.o_decay = 0.88
        
        self.layer_activity_counters = [np.zeros(r.hidden_size, dtype=np.float32) 
                                        for r in self.reservoirs]
        self.prev_spikes = [[] for _ in self.reservoirs]
        self.t = 0
    
    def reset_state(self):
        """状態のリセット"""
        for r in self.reservoirs:
            r.reset()
        self.o_v.fill(0)
        for c in self.layer_activity_counters:
            c.fill(0)
        self.prev_spikes = [[] for _ in self.reservoirs]
    
    def pretrain_unsupervised(self, spike_trains: List[List[List[int]]], 
                             epochs: int = 3, verbose: bool = True):
        """
        教師なし事前学習（STDPのみ）
        
        Args:
            spike_trains: スパイク列のリスト
            epochs: エポック数
            verbose: 進捗表示
        """
        if not self.use_stdp:
            print("Warning: STDP is disabled. Skipping pretraining.")
            return
        
        if verbose:
            print(f"Starting unsupervised pretraining with STDP ({epochs} epochs)...")
        
        for epoch in range(epochs):
            total_samples = len(spike_trains)
            
            for i, spike_train in enumerate(spike_trains):
                self.reset_state()
                
                # STDPを有効にしてフォワードパス
                for input_spikes in spike_train:
                    for j, r in enumerate(self.reservoirs):
                        local_spikes = r.forward(input_spikes, self.prev_spikes[j], 
                                                apply_stdp=True)
                        self.prev_spikes[j] = local_spikes
                
                if verbose and (i + 1) % 100 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}: {i+1}/{total_samples} samples", 
                          end='\r')
            
            if verbose:
                print(f"  Epoch {epoch+1}/{epochs}: Complete                    ")
        
        if verbose:
            print("Unsupervised pretraining finished!")
    
    def train_step(self, spike_train: List[List[int]], target_label: int, 
                   dropout_rate: float = 0.08, use_stdp: bool = False):
        """
        教師あり学習ステップ
        
        Args:
            spike_train: 入力スパイク列
            target_label: ターゲットラベル
            dropout_rate: ドロップアウト率
            use_stdp: STDPを併用するか
        """
        self.reset_state()
        self.t += 1
        
        current_lr = self.lr * (1.0 / (1.0 + 0.00001 * self.t))
        grad_accumulator = [np.zeros_like(w) for w in self.w_ho]
        steps = len(spike_train)
        
        for input_spikes in spike_train:
            if dropout_rate > 0.0 and len(input_spikes) > 2:
                if np.random.random() < 0.5:
                    active_inputs = [idx for idx in input_spikes 
                                   if np.random.random() > dropout_rate]
                else:
                    active_inputs = input_spikes
            else:
                active_inputs = input_spikes
            
            all_hidden_spikes = []
            for i, r in enumerate(self.reservoirs):
                # STDPの適用はオプション
                local_spikes = r.forward(active_inputs, self.prev_spikes[i], 
                                        apply_stdp=use_stdp)
                self.prev_spikes[i] = local_spikes
                if local_spikes:
                    self.layer_activity_counters[i][local_spikes] += 1.0
                    base = self.offsets[i]
                    all_hidden_spikes.extend([idx + base for idx in local_spikes])
            
            self.o_v *= self.o_decay
            
            if not all_hidden_spikes:
                continue
            
            num_spikes = len(all_hidden_spikes)
            scale_factor = 12.0 / (num_spikes + 15.0)
            
            for o in range(self.output_size):
                current = np.sum(self.w_ho[o][all_hidden_spikes])
                self.o_v[o] += current * scale_factor
            
            if np.max(self.o_v) > 0:
                self.o_v -= 0.08 * np.mean(self.o_v)
            self.o_v = np.clip(self.o_v, -6.0, 6.0)
            
            errors = np.zeros(self.output_size, dtype=np.float32)
            if self.o_v[target_label] < 1.5:
                errors[target_label] = 1.5 - self.o_v[target_label]
            
            for o in range(self.output_size):
                if o != target_label and self.o_v[o] > -0.2:
                    errors[o] = -0.2 - self.o_v[o]
            
            for o in range(self.output_size):
                if abs(errors[o]) > 0.01:
                    grad_accumulator[o][all_hidden_spikes] += errors[o]
        
        # Adam更新
        for o in range(self.output_size):
            grad = grad_accumulator[o]
            self.m_ho[o] = self.beta1 * self.m_ho[o] + (1 - self.beta1) * grad
            self.v_ho[o] = self.beta2 * self.v_ho[o] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m_ho[o] / (1 - self.beta1 ** min(self.t, 1000))
            v_hat = self.v_ho[o] / (1 - self.beta2 ** min(self.t, 1000))
            
            self.w_ho[o] += current_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            np.clip(self.w_ho[o], -4.0, 4.0, out=self.w_ho[o])
        
        for i, r in enumerate(self.reservoirs):
            r.update_homeostasis(self.layer_activity_counters[i], steps)
    
    def predict(self, spike_train: List[List[int]]) -> int:
        """予測"""
        self.reset_state()
        total_potentials = np.zeros(self.output_size, dtype=np.float32)
        
        for input_spikes in spike_train:
            all_hidden_spikes = []
            for i, r in enumerate(self.reservoirs):
                local = r.forward(input_spikes, self.prev_spikes[i], apply_stdp=False)
                self.prev_spikes[i] = local
                base = self.offsets[i]
                all_hidden_spikes.extend([x + base for x in local])
            
            self.o_v *= self.o_decay
            if all_hidden_spikes:
                num_spikes = len(all_hidden_spikes)
                scale_factor = 12.0 / (num_spikes + 15.0)
                for o in range(self.output_size):
                    self.o_v[o] += np.sum(self.w_ho[o][all_hidden_spikes]) * scale_factor
            
            if np.max(self.o_v) > 0:
                self.o_v -= 0.08 * np.mean(self.o_v)
            total_potentials += self.o_v
        
        return int(np.argmax(total_potentials))