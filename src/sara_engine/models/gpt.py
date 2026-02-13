_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/gpt.py",
    "//": "タイトル: SARA-Transformer Model (v7: Deep Context)",
    "//": "目的: L3層の入力感度を極限まで下げ、記憶の保持（Echo）を優先する。"
}

import numpy as np
import pickle
import random
from collections import OrderedDict
from typing import List, Dict, Tuple, Any, Optional

from ..core.layers import DynamicLiquidLayer
from ..core.attention import SpikeAttention
from ..memory.sdr import SDREncoder

class SaraGPT:
    """
    SARA-Transformer v7: Deep Context Architecture
    """
    def __init__(self, sdr_size: int = 1024):
        self.sdr_size = sdr_size
        self.encoder = SDREncoder(sdr_size, density=0.02)
        
        self.h_size = 2000
        self.total_hidden = self.h_size * 3
        
        # --- Layer Configuration ---
        # L1: 感覚野。入力に敏感。
        self.l1 = DynamicLiquidLayer(sdr_size, self.h_size, decay=0.5, density=0.08, 
                                     input_scale=2.5, rec_scale=0.5, feedback_scale=0.1)
        
        # L2: 連合野。バランス型。
        self.l2 = DynamicLiquidLayer(self.h_size, self.h_size, decay=0.8, density=0.08, 
                                     input_scale=1.0, rec_scale=1.0, feedback_scale=0.3)
        
        # L3: 海馬/長期記憶バッファ。
        # input_scale=0.1: 新しい入力("is")の影響をほとんど受けない。
        # decay=0.999: 一度入った情報("sky")を絶対に忘れない。
        self.l3 = DynamicLiquidLayer(self.h_size, self.h_size, decay=0.999, density=0.08, 
                                     input_scale=0.1, rec_scale=2.0, feedback_scale=0.0)
        
        self.layers = [self.l1, self.l2, self.l3]
        self.offsets = [0, self.h_size, self.h_size * 2]
        self.prev_spikes: List[List[int]] = [[], [], []]
        
        self.attention = SpikeAttention(input_size=self.h_size, hidden_size=800, 
                                      memory_size=100, num_heads=4)
        self.attention_active = True
        
        # Readout
        self.readout_weights: List[Dict[str, Any]] = []
        for _ in range(sdr_size):
            fan_in = 1000 # 接続を増やしてL3の微弱な信号も拾えるようにする
            idx = np.random.choice(self.total_hidden, fan_in, replace=False)
            w = np.random.normal(0, 0.001, fan_in).astype(np.float32)
            self.readout_weights.append({'idx': idx, 'w': w})
        
        self.readout_v = np.zeros(sdr_size, dtype=np.float32)
        self.readout_decay = 0.8
        self.readout_thresh = 0.5
        self.readout_refractory = np.zeros(sdr_size, dtype=np.float32)

    def reset_state(self):
        for layer in self.layers: layer.reset()
        self.attention.reset()
        self.prev_spikes = [[], [], []]
        self.readout_v.fill(0)
        self.readout_refractory.fill(0)

    def forward_step(self, input_sdr: List[int], training: bool = False, 
                    force_output: bool = False, target_sdr: Optional[List[int]] = None) -> Tuple[List[int], List[int]]:
        
        # --- Forward Pass ---
        spikes_1 = self.l1.forward_with_feedback(input_sdr, self.prev_spikes[0], 
                                                 feedback_active=self.prev_spikes[1], learning=training)
        spikes_2 = self.l2.forward_with_feedback(spikes_1, self.prev_spikes[1], 
                                                 feedback_active=self.prev_spikes[2], learning=training)
        
        attn_signal = []
        if self.attention_active:
            attn_signal = self.attention.compute(spikes_2)
            if len(spikes_2) > 0:
                self.attention.update_memory(spikes_2)
        
        spikes_3 = self.l3.forward_with_feedback(spikes_2, self.prev_spikes[2], 
                                                 feedback_active=[], learning=training,
                                                 attention_signal=attn_signal)
        
        self.prev_spikes = [spikes_1, spikes_2, spikes_3]
        
        all_spikes = []
        all_spikes.extend(spikes_1)
        all_spikes.extend([x + self.offsets[1] for x in spikes_2])
        all_spikes.extend([x + self.offsets[2] for x in spikes_3])
        
        # --- Readout Dynamics ---
        self.readout_v *= self.readout_decay
        self.readout_refractory = np.maximum(0, self.readout_refractory - 1)
        
        if not training:
            noise = np.random.normal(0, 0.01, self.sdr_size).astype(np.float32)
            self.readout_v += noise

        if len(all_spikes) > 0:
            for out_idx in range(self.sdr_size):
                if self.readout_refractory[out_idx] > 0: continue
                mapping = self.readout_weights[out_idx]
                indices = mapping['idx']
                weights = mapping['w']
                active_mask = np.isin(indices, all_spikes)
                if np.any(active_mask):
                    self.readout_v[out_idx] += np.sum(weights[active_mask])
        
        fired_mask = (self.readout_v > self.readout_thresh) & (self.readout_refractory <= 0)
        predicted_sdr = []
        
        if np.any(fired_mask):
            candidates = np.where(fired_mask)[0]
            potentials = self.readout_v[candidates]
            n_predict = max(1, int(self.sdr_size * 0.02))
            if len(candidates) > n_predict:
                top_indices = np.argsort(potentials)[-n_predict:]
                predicted_sdr = candidates[top_indices].tolist()
            else:
                predicted_sdr = candidates.tolist()
            
            self.readout_v[predicted_sdr] = 0
            self.readout_refractory[predicted_sdr] = 2.0 
        
        if len(predicted_sdr) == 0 and force_output:
            if np.max(self.readout_v) > -999:
                best_idx = np.argmax(self.readout_v)
                predicted_sdr = [best_idx]
                self.readout_v[best_idx] = 0
                self.readout_refractory[best_idx] = 2.0

        # --- Readout Learning (Competitive Delta Rule) ---
        actual_target = target_sdr if target_sdr is not None else input_sdr
        
        if training and len(all_spikes) > 0 and len(actual_target) > 0:
            learning_rate = 0.3
            target_set = set(actual_target)
            
            # Positive (Target Forcing)
            for out_idx in actual_target:
                mapping = self.readout_weights[out_idx]
                indices = mapping['idx']
                weights = mapping['w']
                active_mask = np.isin(indices, all_spikes)
                if np.any(active_mask):
                    weights[active_mask] += learning_rate
                    np.clip(weights, -5.0, 5.0, out=weights)

            # Negative (Inhibition)
            for pred_idx in predicted_sdr:
                if pred_idx not in target_set:
                    mapping = self.readout_weights[pred_idx]
                    indices = mapping['idx']
                    weights = mapping['w']
                    active_mask = np.isin(indices, all_spikes)
                    if np.any(active_mask):
                        weights[active_mask] -= learning_rate * 3.0
                        np.clip(weights, -5.0, 5.0, out=weights)
        
        return predicted_sdr, all_spikes

    def save_model(self, filepath: str):
        state = {
            'version': 'sara_transformer_v7',
            'sdr_size': self.sdr_size,
            'readout_weights': self.readout_weights,
            'encoder_cache': dict(self.encoder.cache),
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.readout_weights = state['readout_weights']
        if 'encoder_cache' in state: self.encoder.cache = state['encoder_cache']
        self.reset_state()
        print(f"Model loaded from {filepath}")