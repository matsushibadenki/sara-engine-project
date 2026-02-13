import numpy as np
import pickle
import random
from collections import OrderedDict
from typing import List, Dict, Tuple, Any

from ..core.layers import DynamicLiquidLayer
from ..core.attention import SpikeAttention
from ..memory.sdr import SDREncoder

class SaraGPT:
    """v49: SARA-GPT (Generic SNN Model)"""
    def __init__(self, sdr_size: int = 1024):
        self.sdr_size = sdr_size
        self.encoder = SDREncoder(sdr_size, density=0.02)
        
        self.h_size = 2000
        self.total_hidden = self.h_size * 3
        
        self.l1 = DynamicLiquidLayer(sdr_size, self.h_size, decay=0.3, density=0.08, 
                                     input_scale=2.0, rec_scale=1.0, feedback_scale=0.3)
        self.l2 = DynamicLiquidLayer(self.h_size, self.h_size, decay=0.6, density=0.08, 
                                     input_scale=1.5, rec_scale=1.2, feedback_scale=0.5)
        self.l3 = DynamicLiquidLayer(self.h_size, self.h_size, decay=0.92, density=0.08, 
                                     input_scale=1.2, rec_scale=1.5, feedback_scale=0.0)
        
        self.layers = [self.l1, self.l2, self.l3]
        self.offsets = [0, self.h_size, self.h_size * 2]
        self.prev_spikes: List[List[int]] = [[], [], []]
        
        self.attention = SpikeAttention(input_size=self.h_size, hidden_size=500, memory_size=60)
        self.attention_active = True
        
        self.episodic_memory: List[List[str]] = []
        self.max_episodic_size = 150

        self.readout_weights: List[Dict[str, Any]] = []
        for _ in range(sdr_size):
            fan_in = 600
            idx = np.random.choice(self.total_hidden, fan_in, replace=False)
            w = np.random.normal(0, 0.05, fan_in).astype(np.float32)
            self.readout_weights.append({'idx': idx, 'w': w})
        
        self.readout_v = np.zeros(sdr_size, dtype=np.float32)
        self.readout_decay = 0.85
        self.readout_thresh = 0.5
        self.readout_refractory = np.zeros(sdr_size, dtype=np.float32)
        self.lr = 0.05

    def reset_state(self):
        for layer in self.layers: layer.reset()
        self.attention.reset()
        self.prev_spikes = [[], [], []]
        self.readout_v.fill(0)
        self.readout_refractory.fill(0)

    def forward_step(self, input_sdr: List[int], training: bool = False, 
                    force_output: bool = False) -> Tuple[List[int], List[int]]:
        
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
            
            if not training:
                self.readout_v[predicted_sdr] = 0
                self.readout_refractory[predicted_sdr] = 3.0
        
        if len(predicted_sdr) == 0 and force_output:
            if np.max(self.readout_v) > -999:
                best_idx = np.argmax(self.readout_v)
                predicted_sdr = [best_idx]
                if not training:
                    self.readout_v[best_idx] = 0
                    self.readout_refractory[best_idx] = 3.0
        
        return predicted_sdr, all_spikes

    def save_model(self, filepath: str):
        state = {
            'version': 'v49',
            'sdr_size': self.sdr_size,
            'readout_weights': self.readout_weights,
            'encoder_cache': dict(self.encoder.cache),
            'attn_w_q': self.attention.w_query,
            'attn_w_k': self.attention.w_key,
            'attn_w_v': self.attention.w_value
        }
        # Pythonモード時のみ内部状態を保存
        if not self.l1.use_rust:
             state['layers'] = [{'in_idx': l.in_indices, 'in_w': l.in_weights, 
                        'rec_idx': l.rec_indices, 'rec_w': l.rec_weights, 
                        'dynamic_thresh': l.dynamic_thresh, 'feedback_w': l.feedback_weights} 
                       for l in self.layers]
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        if 'layers' in state and not self.l1.use_rust:
            saved_layers = state['layers']
            for i, layer_data in enumerate(saved_layers):
                target_layer = self.layers[i]
                target_layer.in_indices = layer_data['in_idx']
                target_layer.in_weights = layer_data['in_w']
                target_layer.rec_indices = layer_data['rec_idx']
                target_layer.rec_weights = layer_data['rec_w']
                if 'dynamic_thresh' in layer_data: target_layer.dynamic_thresh = layer_data['dynamic_thresh']
                if 'feedback_w' in layer_data: target_layer.feedback_weights = layer_data['feedback_w']
        self.readout_weights = state['readout_weights']
        if 'encoder_cache' in state: self.encoder.cache = OrderedDict(state['encoder_cache'])
        if 'attn_w_q' in state:
            self.attention.w_query = state['attn_w_q']
            self.attention.w_key = state['attn_w_k']
            self.attention.w_value = state['attn_w_v']
        self.reset_state()
        print(f"Model loaded from {filepath}")