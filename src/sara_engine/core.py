# src/sara_engine/core.py
# Saraエンジン・コアロジック
# Liquid State MachineとSNNを用いた学習モデルのクラス定義および実装

import numpy as np
import random
import pickle
from typing import List, Tuple, Dict, Optional, Union

class LiquidLayer:
    def __init__(self, input_size: int, hidden_size: int, decay: float, input_scale: float, rec_scale: float, density: float = 0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.decay = decay
        
        self.in_indices = []
        self.in_weights = []
        self.rec_indices = []
        self.rec_weights = []
        
        # Input Weights
        fan_in = max(1, int(input_size * density))
        w_range_in = input_scale * np.sqrt(3.0 / fan_in)
        
        for i in range(input_size):
            n = int(hidden_size * density)
            if n > 0:
                idx = np.random.choice(hidden_size, n, replace=False).astype(np.int32)
                w = np.random.uniform(-w_range_in, w_range_in, n).astype(np.float32)
                self.in_indices.append(idx)
                self.in_weights.append(w)
            else:
                self.in_indices.append(np.array([], dtype=np.int32))
                self.in_weights.append(np.array([], dtype=np.float32))

        # Recurrent Weights
        rec_density = 0.1
        fan_in_rec = max(1, int(hidden_size * rec_density))
        w_range_rec = rec_scale / np.sqrt(fan_in_rec)
        
        for i in range(hidden_size):
            n = int(hidden_size * rec_density)
            if n > 0:
                idx = np.random.choice(hidden_size, n, replace=False).astype(np.int32)
                w = np.random.uniform(-w_range_rec, w_range_rec, n).astype(np.float32)
                self.rec_indices.append(idx)
                self.rec_weights.append(w)
            else:
                self.rec_indices.append(np.array([], dtype=np.int32))
                self.rec_weights.append(np.array([], dtype=np.float32))

        self.v = np.zeros(hidden_size, dtype=np.float32)
        self.refractory = np.zeros(hidden_size, dtype=np.float32)
        
        if decay < 0.4:  
            self.base_thresh = 0.8
            self.target_rate = 0.02
            self.refractory_period = 3.0
        elif decay < 0.8:  
            self.base_thresh = 0.8
            self.target_rate = 0.03
            self.refractory_period = 2.0
        else:  
            self.base_thresh = 1.0
            self.target_rate = 0.02
            self.refractory_period = 1.5
            
        self.thresh = np.ones(hidden_size, dtype=np.float32) * self.base_thresh

    def reset(self):
        self.v.fill(0)
        self.refractory.fill(0)

    def update_homeostasis(self, activity_history: np.ndarray, steps: int):
        rate = activity_history / float(steps)
        diff = rate - self.target_rate
        gain = np.where(np.abs(diff) > 0.05, 0.15, 0.03)
        self.thresh += gain * diff
        self.thresh = np.clip(self.thresh, self.base_thresh * 0.5, self.base_thresh * 5.0)

    def forward(self, active_inputs: List[int], prev_active_hidden: List[int]) -> List[int]:
        self.refractory = np.maximum(0, self.refractory - 1)
        self.v *= self.decay
        
        for pre_id in active_inputs:
            if pre_id < len(self.in_indices):
                targets = self.in_indices[pre_id]
                ws = self.in_weights[pre_id]
                if len(targets) > 0:
                    self.v[targets] += ws
                    
        for pre_h_id in prev_active_hidden:
            if pre_h_id < len(self.rec_indices):
                targets = self.rec_indices[pre_h_id]
                ws = self.rec_weights[pre_h_id]
                if len(targets) > 0:
                    self.v[targets] += ws
        
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
        
        self.reservoirs = [
            LiquidLayer(input_size, 1500, decay=0.3, input_scale=1.0, rec_scale=1.2), 
            LiquidLayer(input_size, 2000, decay=0.7, input_scale=0.8, rec_scale=1.5), 
            LiquidLayer(input_size, 1500, decay=0.95, input_scale=0.4, rec_scale=2.0),
        ]
        
        self.total_hidden = sum(r.hidden_size for r in self.reservoirs)
        self.offsets = [0, 1500, 3500]
        
        self.w_ho = []
        self.m_ho = []
        self.v_ho = []
        
        for _ in range(output_size):
            limit = np.sqrt(6.0 / (self.total_hidden + output_size))
            w = np.random.uniform(-limit, limit, self.total_hidden).astype(np.float32)
            self.w_ho.append(w)
            self.m_ho.append(np.zeros(self.total_hidden, dtype=np.float32))
            self.v_ho.append(np.zeros(self.total_hidden, dtype=np.float32))
            
        self.o_v = np.zeros(output_size, dtype=np.float32)
        
        self.lr = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.o_decay = 0.9
        
        self.layer_activity_counters = [np.zeros(r.hidden_size, dtype=np.float32) for r in self.reservoirs]
        self.prev_spikes = [[] for _ in self.reservoirs]

    def reset_state(self):
        for r in self.reservoirs:
            r.reset()
        self.o_v.fill(0)
        for c in self.layer_activity_counters: c.fill(0)
        self.prev_spikes = [[] for _ in self.reservoirs]

    def save_model(self, filepath: str):
        state = {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'reservoirs': self.reservoirs,
            'offsets': self.offsets,
            'w_ho': self.w_ho,
            'm_ho': self.m_ho,
            'v_ho': self.v_ho,
            'lr': self.lr
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
        self.w_ho = state['w_ho']
        self.m_ho = state['m_ho']
        self.v_ho = state['v_ho']
        self.lr = state.get('lr', 0.001)
        
        self.total_hidden = sum(r.hidden_size for r in self.reservoirs)
        self.o_v = np.zeros(self.output_size, dtype=np.float32)
        self.layer_activity_counters = [np.zeros(r.hidden_size, dtype=np.float32) for r in self.reservoirs]
        self.prev_spikes = [[] for _ in self.reservoirs]
        print(f"Model loaded from {filepath}")

    def sleep_phase(self, prune_rate: float = 0.05):
        print(f"  [Sleep Phase] Consolidating memory...")
        pruned_total = 0
        total_weights = 0
        
        for o in range(self.output_size):
            weights = self.w_ho[o]
            total_weights += len(weights)
            weights *= 0.995 
            
            abs_w = np.abs(weights)
            nonzero_w = abs_w[abs_w > 1e-6]
            if len(nonzero_w) > 0:
                threshold = np.percentile(nonzero_w, prune_rate * 100)
                mask = abs_w < threshold
                pruned_total += np.sum(mask)
                weights[mask] = 0.0
            
            norm = np.linalg.norm(weights)
            if norm > 5.0: weights *= (5.0 / norm)
            self.w_ho[o] = weights
            
        print(f"  [Sleep Phase] Pruned {pruned_total} / {total_weights} connections.")

    def train_step(self, spike_train: List[List[int]], target_label: int, dropout_rate: float = 0.1):
        self.reset_state()
        grad_accumulator = [np.zeros_like(w) for w in self.w_ho]
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
            
            self.o_v *= self.o_decay
            
            if not all_hidden_spikes:
                continue

            num_spikes = len(all_hidden_spikes)
            scale_factor = 10.0 / (num_spikes + 20.0)
            
            for o in range(self.output_size):
                current = np.sum(self.w_ho[o][all_hidden_spikes])
                self.o_v[o] += current * scale_factor
            
            if np.max(self.o_v) > 0:
                self.o_v -= 0.1 * np.mean(self.o_v)
            self.o_v = np.clip(self.o_v, -5.0, 5.0)

            errors = np.zeros(self.output_size, dtype=np.float32)
            if self.o_v[target_label] < 1.0:
                errors[target_label] = 1.0 - self.o_v[target_label]
            
            for o in range(self.output_size):
                if o != target_label and self.o_v[o] > -0.1:
                    errors[o] = -0.1 - self.o_v[o]
            
            for o in range(self.output_size):
                if abs(errors[o]) > 0.01:
                    grad_accumulator[o][all_hidden_spikes] += errors[o]

        for o in range(self.output_size):
            grad = grad_accumulator[o]
            self.m_ho[o] = self.beta1 * self.m_ho[o] + (1 - self.beta1) * grad
            self.v_ho[o] = self.beta2 * self.v_ho[o] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m_ho[o]
            v_hat = self.v_ho[o]
            self.w_ho[o] += self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            np.clip(self.w_ho[o], -3.0, 3.0, out=self.w_ho[o])
        
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
            
            self.o_v *= self.o_decay
            if all_hidden_spikes:
                num_spikes = len(all_hidden_spikes)
                scale_factor = 10.0 / (num_spikes + 20.0)
                for o in range(self.output_size):
                    self.o_v[o] += np.sum(self.w_ho[o][all_hidden_spikes]) * scale_factor
            
            if np.max(self.o_v) > 0:
                self.o_v -= 0.1 * np.mean(self.o_v)
            total_potentials += self.o_v
            
        return int(np.argmax(total_potentials))