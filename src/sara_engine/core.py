# src/sara_engine/core.py
# Saraエンジン・コアロジック (Improved Version v50)
# Vectorized Sparse Ops & WTA Readout Layer

import numpy as np
import random
import pickle
from typing import List, Tuple, Dict, Optional, Union

class LiquidLayer:
    def __init__(self, input_size: int, hidden_size: int, decay: float, input_scale: float, rec_scale: float, density: float = 0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.decay = decay
        
        self.in_indices: List[np.ndarray] = []
        self.in_weights: List[np.ndarray] = []
        self.rec_indices: List[np.ndarray] = []
        self.rec_weights: List[np.ndarray] = []
        
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

        self.v = np.zeros(hidden_size, dtype=np.float32)
        self.refractory = np.zeros(hidden_size, dtype=np.float32)
        
        if decay < 0.4:  
            self.base_thresh = 1.0
            self.target_rate = 0.025
            self.refractory_period = 2.0
        elif decay < 0.8:  
            self.base_thresh = 1.2
            self.target_rate = 0.035
            self.refractory_period = 2.0
        else:  
            self.base_thresh = 1.4
            self.target_rate = 0.025
            self.refractory_period = 1.5
            
        self.thresh = np.ones(hidden_size, dtype=np.float32) * self.base_thresh

    def reset(self):
        self.v.fill(0)
        self.refractory.fill(0)

    def update_homeostasis(self, activity_history: np.ndarray, steps: int):
        if steps == 0: return
        rate = activity_history / float(steps)
        diff = rate - self.target_rate
        gain = np.where(np.abs(diff) > 0.08, 0.2, 0.05)
        self.thresh += gain * diff
        self.thresh = np.clip(self.thresh, self.base_thresh * 0.5, self.base_thresh * 5.0)

    def forward(self, active_inputs: List[int], prev_active_hidden: List[int]) -> List[int]:
        self.refractory = np.maximum(0, self.refractory - 1)
        self.v *= self.decay
        
        # 修正: ベクトル化による高速化
        if active_inputs:
            all_targets = []
            all_weights = []
            for pre_id in active_inputs:
                if pre_id < len(self.in_indices):
                    targets = self.in_indices[pre_id]
                    ws = self.in_weights[pre_id]
                    if len(targets) > 0:
                        all_targets.append(targets)
                        all_weights.append(ws)
            
            if all_targets:
                combined_targets = np.concatenate(all_targets)
                combined_weights = np.concatenate(all_weights)
                np.add.at(self.v, combined_targets, combined_weights)
        
        if prev_active_hidden:
            all_targets = []
            all_weights = []
            for pre_h_id in prev_active_hidden:
                if pre_h_id < len(self.rec_indices):
                    targets = self.rec_indices[pre_h_id]
                    ws = self.rec_weights[pre_h_id]
                    if len(targets) > 0:
                        all_targets.append(targets)
                        all_weights.append(ws)
            
            if all_targets:
                combined_targets = np.concatenate(all_targets)
                combined_weights = np.concatenate(all_weights)
                np.add.at(self.v, combined_targets, combined_weights)
        
        ready_mask = (self.v >= self.thresh) & (self.refractory <= 0)
        fired_indices = np.where(ready_mask)[0]
        
        max_spikes = int(self.hidden_size * 0.20)
        if len(fired_indices) > max_spikes:
            np.random.shuffle(fired_indices)
            fired_indices = fired_indices[:max_spikes]
            self.thresh += 0.05
        
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
        
        self.reservoirs = [
            LiquidLayer(input_size, 1200, decay=0.25, input_scale=1.2, rec_scale=1.3),
            LiquidLayer(input_size, 1800, decay=0.5, input_scale=1.0, rec_scale=1.6),
            LiquidLayer(input_size, 1800, decay=0.75, input_scale=0.7, rec_scale=1.8),
            LiquidLayer(input_size, 1200, decay=0.92, input_scale=0.5, rec_scale=2.2),
        ]
        
        self.total_hidden = sum(r.hidden_size for r in self.reservoirs)
        self.offsets = [0, 1200, 3000, 4800]
        
        # 修正: 中間層 (Winner-Take-All) を追加
        self.intermediate_size = 500
        self.w_hi = []  # Hidden -> Intermediate
        self.w_io = []  # Intermediate -> Output
        
        # Hidden -> Intermediate
        for _ in range(self.intermediate_size):
            fan_in = 600
            idx = np.random.choice(self.total_hidden, fan_in, replace=False)
            w = np.random.normal(0, 0.05, fan_in).astype(np.float32)
            self.w_hi.append({'idx': idx, 'w': w})
        
        # Intermediate -> Output
        for _ in range(output_size):
            w = np.random.normal(0, 0.1, self.intermediate_size).astype(np.float32)
            self.w_io.append(w)
            
        self.intermediate_v = np.zeros(self.intermediate_size, dtype=np.float32)
        self.o_v = np.zeros(output_size, dtype=np.float32)
        
        self.lr = 0.002
        self.o_decay = 0.88
        
        self.layer_activity_counters = [np.zeros(r.hidden_size, dtype=np.float32) for r in self.reservoirs]
        self.prev_spikes: List[List[int]] = [[] for _ in self.reservoirs]
        self.t = 0

    def reset_state(self):
        for r in self.reservoirs: r.reset()
        self.o_v.fill(0)
        self.intermediate_v.fill(0)
        for c in self.layer_activity_counters: c.fill(0)
        self.prev_spikes = [[] for _ in self.reservoirs]

    def save_model(self, filepath: str):
        state = {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'reservoirs': self.reservoirs,
            'offsets': self.offsets,
            'w_hi': self.w_hi,
            'w_io': self.w_io,
            'lr': self.lr,
            't': self.t
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.input_size = state['input_size']
        self.output_size = state['output_size']
        self.reservoirs = state['reservoirs']
        self.offsets = state['offsets']
        self.w_hi = state['w_hi']
        self.w_io = state['w_io']
        self.lr = state.get('lr', 0.001)
        self.t = state.get('t', 0)
        
        self.total_hidden = sum(r.hidden_size for r in self.reservoirs)
        self.intermediate_size = len(self.w_io[0])
        self.intermediate_v = np.zeros(self.intermediate_size, dtype=np.float32)
        self.o_v = np.zeros(self.output_size, dtype=np.float32)
        self.layer_activity_counters = [np.zeros(r.hidden_size, dtype=np.float32) for r in self.reservoirs]
        self.prev_spikes = [[] for _ in self.reservoirs]
        print(f"Model loaded from {filepath}")

    def sleep_phase(self, epoch: int = 0, sample_size: int = 0):
        """
        睡眠フェーズ：記憶の整理とシナプスの刈り込みを実行 (WTA版)
        """
        prune_rate = 0.02
        if epoch >= 2: prune_rate = 0.01

        if sample_size >= 1000:
            if epoch == 0: prune_rate = 0.01
            elif epoch == 1: prune_rate = 0.005
            else: prune_rate = 0.001

        print(f"  [Sleep Phase] Consolidating memory (Auto-rate: {prune_rate*100:.2f}%)...")
        pruned_total = 0
        total_weights = 0
        
        # Prune Intermediate -> Output weights (w_io)
        for o in range(self.output_size):
            weights = self.w_io[o]
            total_weights += len(weights)
            weights *= 0.997
            
            abs_w = np.abs(weights)
            nonzero_w = abs_w[abs_w > 1e-6]
            if len(nonzero_w) > 0:
                threshold = np.percentile(nonzero_w, prune_rate * 100)
                mask = abs_w < threshold
                pruned_total += np.sum(mask)
                weights[mask] = 0.0
            
            norm = np.linalg.norm(weights)
            if norm > 6.0: weights *= (6.0 / norm)
            self.w_io[o] = weights
            
        print(f"  [Sleep Phase] Pruned {pruned_total} / {total_weights} connections.")

    def train_step(self, spike_train: List[List[int]], target_label: int, dropout_rate: float = 0.08):
        self.reset_state()
        self.t += 1
        steps = len(spike_train)
        
        for input_spikes in spike_train:
            if dropout_rate > 0.0 and len(input_spikes) > 2:
                if random.random() < 0.5:
                    active_inputs = [idx for idx in input_spikes if random.random() > dropout_rate]
                else:
                    active_inputs = input_spikes
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
            
            # 修正: Hidden -> Intermediate (Winner-Take-All)
            self.intermediate_v.fill(0)
            if all_hidden_spikes:
                for i in range(self.intermediate_size):
                    mapping = self.w_hi[i]
                    active_mask = np.isin(mapping['idx'], all_hidden_spikes)
                    if np.any(active_mask):
                        self.intermediate_v[i] = np.sum(mapping['w'][active_mask])
            
            # Top-K (Sparse)
            k = int(self.intermediate_size * 0.2)
            if k > 0:
                top_k_indices = np.argpartition(self.intermediate_v, -k)[-k:]
                intermediate_spikes = top_k_indices[self.intermediate_v[top_k_indices] > 0].tolist()
            else:
                intermediate_spikes = []

            # Intermediate -> Output
            self.o_v *= self.o_decay
            if intermediate_spikes:
                for o in range(self.output_size):
                    self.o_v[o] += np.sum(self.w_io[o][intermediate_spikes])
            
            # 学習 (Online Delta Rule)
            errors = np.zeros(self.output_size, dtype=np.float32)
            if self.o_v[target_label] < 1.5:
                errors[target_label] = 1.5 - self.o_v[target_label]
            
            for o in range(self.output_size):
                if o != target_label and self.o_v[o] > -0.2:
                    errors[o] = -0.2 - self.o_v[o]
            
            # Weight Update (w_io)
            if intermediate_spikes:
                for o in range(self.output_size):
                    if abs(errors[o]) > 0.01:
                        self.w_io[o][intermediate_spikes] += self.lr * errors[o]

        for i, r in enumerate(self.reservoirs):
            r.update_homeostasis(self.layer_activity_counters[i], steps)

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
            
            # Forward WTA
            self.intermediate_v.fill(0)
            if all_hidden_spikes:
                for i in range(self.intermediate_size):
                    mapping = self.w_hi[i]
                    active_mask = np.isin(mapping['idx'], all_hidden_spikes)
                    if np.any(active_mask):
                        self.intermediate_v[i] = np.sum(mapping['w'][active_mask])
            
            k = int(self.intermediate_size * 0.2)
            intermediate_spikes = []
            if k > 0:
                top_k_indices = np.argpartition(self.intermediate_v, -k)[-k:]
                intermediate_spikes = top_k_indices[self.intermediate_v[top_k_indices] > 0].tolist()

            self.o_v *= self.o_decay
            if intermediate_spikes:
                for o in range(self.output_size):
                    self.o_v[o] += np.sum(self.w_io[o][intermediate_spikes])
            
            total_potentials += self.o_v
            
        return int(np.argmax(total_potentials))