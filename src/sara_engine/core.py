# src/sara_engine/core.py
# Saraエンジン・コアロジック (Improved Version v74: Polished Diamond)
# プルーニングの最小化とマージン最大化（+2.0 vs -1.0）により、堅牢な識別境界を構築し95%突破を目指す

import numpy as np
import random
import pickle
from typing import List, Tuple, Dict, Optional, Union

class LiquidLayer:
    def __init__(self, input_size: int, hidden_size: int, decay: float, input_scale: float, rec_scale: float, density: float = 0.06):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.decay = decay
        
        self.in_indices: List[np.ndarray] = []
        self.in_weights: List[np.ndarray] = []
        self.rec_indices: List[np.ndarray] = []
        self.rec_weights: List[np.ndarray] = []
        
        # 最適化: 密度0.06 (0.05と0.08の中間)
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
        rec_density = 0.10
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

        self.v = np.zeros(hidden_size, dtype=np.float32)
        self.refractory = np.zeros(hidden_size, dtype=np.float32)
        
        # 閾値設定 (安定志向)
        if decay < 0.4:  
            self.base_thresh = 0.65
            self.target_rate = 0.03
            self.refractory_period = 2.5
        elif decay < 0.8:  
            self.base_thresh = 0.75
            self.target_rate = 0.03
            self.refractory_period = 2.0
        else:  
            self.base_thresh = 0.85
            self.target_rate = 0.02
            self.refractory_period = 1.5
            
        self.thresh = np.ones(hidden_size, dtype=np.float32) * self.base_thresh

    def reset(self):
        self.v.fill(0)
        self.refractory.fill(0)

    def update_homeostasis(self, activity_history: np.ndarray, steps: int):
        if steps == 0: return
        rate = activity_history / float(steps)
        diff = rate - self.target_rate
        gain = np.where(np.abs(diff) > 0.08, 0.1, 0.02)
        self.thresh += gain * diff
        self.thresh = np.clip(self.thresh, self.base_thresh * 0.5, self.base_thresh * 5.0)

    def forward(self, active_inputs: List[int], prev_active_hidden: List[int]) -> List[int]:
        self.refractory = np.maximum(0, self.refractory - 1)
        self.v *= self.decay
        
        # Vectorized Input Integration
        if active_inputs:
            for pre_id in active_inputs:
                if pre_id < len(self.in_indices):
                    targets = self.in_indices[pre_id]
                    if len(targets) > 0:
                        self.v[targets] += self.in_weights[pre_id]
        
        if prev_active_hidden:
            for pre_h_id in prev_active_hidden:
                if pre_h_id < len(self.rec_indices):
                    targets = self.rec_indices[pre_h_id]
                    if len(targets) > 0:
                        self.v[targets] += self.rec_weights[pre_h_id]
        
        ready_mask = (self.v >= self.thresh) & (self.refractory <= 0)
        fired_indices = np.where(ready_mask)[0]
        
        if len(fired_indices) > 0:
            self.v[fired_indices] -= self.thresh[fired_indices]
            self.v = np.maximum(self.v, 0.0)
            self.refractory[fired_indices] = self.refractory_period
            
        return fired_indices.tolist()

class SaraEngine:
    def __init__(self, input_size: int, output_size: int, load_path: Optional[str] = None):
        if load_path:
            self.load_model(load_path)
            return

        self.input_size = input_size
        self.output_size = output_size
        
        # L1の入力を強化 (1.2 -> 2.0) して初期反応をブースト
        self.reservoirs = [
            LiquidLayer(input_size, 1000, decay=0.25, input_scale=2.0, rec_scale=1.2, density=0.08),
            LiquidLayer(input_size, 1600, decay=0.50, input_scale=1.5, rec_scale=1.5, density=0.06),
            LiquidLayer(input_size, 1600, decay=0.75, input_scale=1.0, rec_scale=1.8, density=0.05),
            LiquidLayer(input_size, 1000, decay=0.92, input_scale=0.8, rec_scale=2.0, density=0.04),
        ]
        
        self.total_hidden = sum(r.hidden_size for r in self.reservoirs)
        self.offsets = [0, 1000, 2600, 4200]
        
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
        
        self.lr = 0.002
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        
        # 安定重視の出力減衰
        self.o_decay = 0.94
        
        self.layer_activity_counters = [np.zeros(r.hidden_size, dtype=np.float32) for r in self.reservoirs]
        self.prev_spikes = [[] for _ in self.reservoirs]
        self.t = 0

    def reset_state(self):
        for r in self.reservoirs: r.reset()
        self.o_v.fill(0)
        for c in self.layer_activity_counters: c.fill(0)
        self.prev_spikes = [[] for _ in self.reservoirs]

    def save_model(self, filepath: str):
        state = {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'reservoirs': self.reservoirs,
            'offsets': self.offsets,
            'w_ho': self.w_ho, 'm_ho': self.m_ho, 'v_ho': self.v_ho,
            'lr': self.lr, 't': self.t
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.input_size = state['input_size']
        self.output_size = state['output_size']
        self.reservoirs = state['reservoirs']
        self.offsets = state['offsets']
        
        if 'w_ho_a' in state: # 互換性維持
            self.w_ho = state['w_ho_a']
            self.m_ho = [np.zeros_like(w) for w in self.w_ho]
            self.v_ho = [np.zeros_like(w) for w in self.w_ho]
        else:
            self.w_ho = state['w_ho']
            self.m_ho = state['m_ho']
            self.v_ho = state['v_ho']

        self.lr = state.get('lr', 0.001)
        self.t = state.get('t', 0)
        self.total_hidden = sum(r.hidden_size for r in self.reservoirs)
        self.o_v = np.zeros(self.output_size, dtype=np.float32)
        self.layer_activity_counters = [np.zeros(r.hidden_size, dtype=np.float32) for r in self.reservoirs]
        self.prev_spikes = [[] for _ in self.reservoirs]
        print(f"Model loaded from {filepath}")

    def sleep_phase(self, epoch: int = 0, sample_size: int = 1000):
        """適応的プルーニング (One-time, Low-Damage)"""
        # Epoch 2の終了時のみ実行
        if epoch != 1:
            print("  [Sleep Phase] Skipping pruning (Preserving structure).")
            return
        
        # 修正: プルーニング率を2.5%に下げて、ダメージを最小限にする
        prune_rate = 0.025 
        
        print(f"  [Sleep Phase] Pruning {prune_rate*100:.1f}% (Conservative)...")
        pruned_total = 0
        total_weights = 0
        
        for o in range(self.output_size):
            weights = self.w_ho[o]
            total_weights += len(weights)
            weights *= 0.998
            
            abs_w = np.abs(weights)
            nonzero_w = abs_w[abs_w > 1e-6]
            if len(nonzero_w) > 0:
                threshold = np.percentile(nonzero_w, prune_rate * 100)
                mask = abs_w < threshold
                pruned_total += np.sum(mask)
                weights[mask] = 0.0
                if len(self.m_ho) > o:
                    self.m_ho[o][mask] = 0.0
                    self.v_ho[o][mask] = 0.0
            
            norm = np.linalg.norm(weights)
            if norm > 6.0: weights *= (6.0 / norm)
            self.w_ho[o] = weights
            
        print(f"  [Sleep Phase] Pruned {pruned_total} / {total_weights}.")

    def train_step(self, spike_train: List[List[int]], target_label: int, dropout_rate: float = 0.08):
        self.reset_state()
        self.t += 1
        
        # 学習率減衰
        current_lr = self.lr / (1.0 + 0.0005 * self.t)
        
        grad_accumulator = [np.zeros_like(w) for w in self.w_ho]
        
        for input_spikes in spike_train:
            if dropout_rate > 0.0 and len(input_spikes) > 3 and random.random() < 0.5:
                active_inputs = [idx for idx in input_spikes if random.random() > dropout_rate]
            else:
                active_inputs = input_spikes

            all_hidden_spikes = []
            for i, r in enumerate(self.reservoirs):
                local_spikes = r.forward(active_inputs, self.prev_spikes[i])
                self.prev_spikes[i] = local_spikes
                if local_spikes:
                    self.layer_activity_counters[i][local_spikes] += 1.0
                    base = self.offsets[i]
                    all_hidden_spikes.extend([idx + base for idx in local_spikes])
            
            self.o_v *= self.o_decay
            
            if not all_hidden_spikes:
                continue

            scale_factor = 14.0 / (len(all_hidden_spikes) + 12.0)
            
            for o in range(self.output_size):
                self.o_v[o] += np.sum(self.w_ho[o][all_hidden_spikes]) * scale_factor
            
            if np.max(self.o_v) > 0:
                self.o_v -= 0.05 * np.mean(self.o_v)
            self.o_v = np.clip(self.o_v, -6.0, 6.0)

            errors = np.zeros(self.output_size, dtype=np.float32)
            
            # 修正: マージン最大化 (+2.0)
            if self.o_v[target_label] < 2.0:
                errors[target_label] = 2.0 - self.o_v[target_label]
            
            # 修正: 間違いを強く叩く (-1.0)
            # これにより「正解(+2.0)」と「不正解(-1.0)」の差（マージン）を広げる
            for o in range(self.output_size):
                if o != target_label and self.o_v[o] > -1.0:
                    errors[o] = -1.0 - self.o_v[o]
            
            for o in range(self.output_size):
                if abs(errors[o]) > 0.01:
                    grad_accumulator[o][all_hidden_spikes] += errors[o]

        # Adam Update
        for o in range(self.output_size):
            self.m_ho[o] = self.beta1 * self.m_ho[o] + (1 - self.beta1) * grad_accumulator[o]
            self.v_ho[o] = self.beta2 * self.v_ho[o] + (1 - self.beta2) * (grad_accumulator[o] ** 2)
            
            m_hat = self.m_ho[o] / (1 - self.beta1 ** min(self.t, 1000))
            v_hat = self.v_ho[o] / (1 - self.beta2 ** min(self.t, 1000))
            
            self.w_ho[o] += current_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            np.clip(self.w_ho[o], -5.0, 5.0, out=self.w_ho[o])
        
        for i, r in enumerate(self.reservoirs):
            r.update_homeostasis(self.layer_activity_counters[i], len(spike_train))

    def predict(self, spike_train: List[List[int]]) -> int:
        self.reset_state()
        total_potentials = np.zeros(self.output_size, dtype=np.float32)
        
        for input_spikes in spike_train:
            all_hidden_spikes = []
            for i, r in enumerate(self.reservoirs):
                local = r.forward(input_spikes, self.prev_spikes[i])
                self.prev_spikes[i] = local
                base = self.offsets[i]
                all_hidden_spikes.extend([x + base for x in local])
            
            self.o_v *= self.o_decay
            if all_hidden_spikes:
                scale_factor = 14.0 / (len(all_hidden_spikes) + 12.0)
                for o in range(self.output_size):
                    self.o_v[o] += np.sum(self.w_ho[o][all_hidden_spikes]) * scale_factor
            
            if np.max(self.o_v) > 0:
                self.o_v -= 0.05 * np.mean(self.o_v)
            total_potentials += self.o_v
            
        return int(np.argmax(total_potentials))