# src/sara_engine/sara_gpt_core.py
# SARA-GPT Core Logic (v43: Echo Training & Normalized Weights)

import numpy as np
import hashlib
import pickle
import random
from typing import List, Dict, Tuple, Optional
from .core import LiquidLayer 

class SDREncoder:
    """v42: 完全決定論的SDRエンコーダー (変更なし)"""
    def __init__(self, input_size: int, density: float = 0.02):
        self.input_size = input_size
        self.density = density
        self.cache: Dict[str, List[int]] = {}

    def encode(self, text: str) -> List[int]:
        if text in self.cache: return self.cache[text]
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        target_n = int(self.input_size * self.density)
        indices = rng.choice(self.input_size, target_n, replace=False)
        indices.sort()
        result = indices.tolist()
        self.cache[text] = result
        return result

    def decode(self, sdr: List[int], candidates: List[str]) -> str:
        if not sdr: return ""
        best_word = "<unk>"
        best_overlap = -1
        sdr_set = set(sdr)
        for word in candidates:
            target_sdr = self.encode(word)
            overlap = len(sdr_set.intersection(target_sdr))
            if overlap > best_overlap:
                best_overlap = overlap
                best_word = word
        return best_word

class KWtaLiquidLayer(LiquidLayer):
    """v43: K-WTA Liquid Layer"""
    def __init__(self, input_size: int, hidden_size: int, decay: float, 
                 density: float = 0.1, input_scale: float = 1.0, 
                 rec_scale: float = 1.0, feedback_scale: float = 0.5):
        
        self.size = hidden_size
        self.decay = decay
        
        self.in_indices = []
        self.in_weights = []
        self.rec_indices = []
        self.rec_weights = []
        
        # Init weights
        for i in range(input_size):
            n = int(hidden_size * density)
            if n > 0:
                idx = np.random.choice(hidden_size, n, replace=False).astype(np.int32)
                w = np.random.uniform(-input_scale, input_scale, n).astype(np.float32)
                self.in_indices.append(idx)
                self.in_weights.append(w)
            else:
                self.in_indices.append(np.array([], dtype=np.int32))
                self.in_weights.append(np.array([], dtype=np.float32))
        
        rec_density = 0.15
        for i in range(hidden_size):
            n = int(hidden_size * rec_density)
            if n > 0:
                idx = np.random.choice(hidden_size, n, replace=False).astype(np.int32)
                w = np.random.uniform(-rec_scale, rec_scale, n).astype(np.float32)
                self.rec_indices.append(idx)
                self.rec_weights.append(w)
            else:
                self.rec_indices.append(np.array([], dtype=np.int32))
                self.rec_weights.append(np.array([], dtype=np.float32))
        
        self.v = np.zeros(hidden_size, dtype=np.float32)
        self.refractory = np.zeros(hidden_size, dtype=np.float32)
        self.dynamic_thresh = np.ones(hidden_size, dtype=np.float32) * 0.5
        
        self.stp_gains = [np.ones(len(w), dtype=np.float32) for w in self.rec_weights]
        self.facilitation_rate = 0.1 # 0.25 -> 0.1 (Stabilize)
        self.max_gain = 2.0 # 3.5 -> 2.0 (Stabilize)
        self.recovery_rate = 0.05
        self.target_rate = 0.05 # Target 5% activity
        self.homeostasis_rate = 0.005
        
        self.feedback_scale = feedback_scale
        self.feedback_weights = []
        rng = np.random.RandomState(42)
        for i in range(hidden_size):
            targets = rng.choice(hidden_size, int(hidden_size * 0.05), replace=False)
            self.feedback_weights.append(targets)
        
        self.spike_trace = np.zeros(hidden_size, dtype=np.float32)
        self.trace_decay = 0.9
        self.stdp_lr = 0.005
        self.k_winners = int(hidden_size * 0.05) # Keep sparsity high (5%)

    def forward_with_feedback(self, active_inputs: List[int], 
                             prev_active_hidden: List[int], 
                             feedback_active: List[int] = [], 
                             learning: bool = False) -> List[int]:
        
        self.refractory = np.maximum(0, self.refractory - 1)
        self.v *= self.decay
        
        # Input
        for pre_id in active_inputs:
            if pre_id < len(self.in_indices):
                targets = self.in_indices[pre_id]
                ws = self.in_weights[pre_id]
                if len(targets) > 0: self.v[targets] += ws
        
        # Recurrent
        for pre_h_id in prev_active_hidden:
            if pre_h_id < len(self.rec_indices):
                targets = self.rec_indices[pre_h_id]
                ws = self.rec_weights[pre_h_id]
                if pre_h_id < len(self.stp_gains):
                    gains = self.stp_gains[pre_h_id]
                    if len(targets) > 0:
                        self.v[targets] += ws * gains
                        gains += self.facilitation_rate
                        np.clip(gains, 1.0, self.max_gain, out=gains)
        
        # Feedback
        if len(feedback_active) > 0:
            for fb_id in feedback_active:
                if fb_id < len(self.feedback_weights):
                    targets = self.feedback_weights[fb_id]
                    self.v[targets] += self.feedback_scale
        
        # STP Recovery
        for i in range(len(self.stp_gains)):
            if len(self.stp_gains[i]) > 0:
                self.stp_gains[i] += (1.0 - self.stp_gains[i]) * self.recovery_rate
        
        # Fire (K-WTA)
        candidates_mask = (self.v >= self.dynamic_thresh) & (self.refractory <= 0)
        candidates_indices = np.where(candidates_mask)[0]
        
        fired_indices = []
        if len(candidates_indices) > 0:
            if len(candidates_indices) > self.k_winners:
                potentials = self.v[candidates_indices]
                top_k_local_indices = np.argsort(potentials)[-self.k_winners:]
                fired_indices = candidates_indices[top_k_local_indices].tolist()
            else:
                fired_indices = candidates_indices.tolist()
            
            fired_arr = np.array(fired_indices, dtype=int)
            self.v[fired_arr] = 0.0
            self.refractory[fired_arr] = 2.0
            self.dynamic_thresh[fired_arr] += self.homeostasis_rate * 2.0
        
        # Homeostasis
        not_fired_mask = np.ones(self.size, dtype=bool)
        if len(fired_indices) > 0:
            not_fired_mask[np.array(fired_indices, dtype=int)] = False
        
        self.dynamic_thresh[not_fired_mask] -= self.homeostasis_rate * self.target_rate
        np.clip(self.dynamic_thresh, 0.1, 5.0, out=self.dynamic_thresh)
        
        # STDP
        if learning and len(fired_indices) > 0:
            fired_arr = np.array(fired_indices, dtype=int)
            for pre_id in prev_active_hidden:
                if pre_id < len(self.rec_indices):
                    targets = self.rec_indices[pre_id]
                    weights = self.rec_weights[pre_id]
                    mask = np.isin(targets, fired_arr)
                    if np.any(mask):
                        weights[mask] += self.stdp_lr
                        np.clip(weights, -2.0, 2.0, out=weights)
        
        # Trace
        self.spike_trace *= self.trace_decay
        if len(fired_indices) > 0:
            self.spike_trace[np.array(fired_indices, dtype=int)] += 1.0
        
        return fired_indices

    def prune_synapses(self, threshold: float = 0.01):
        for i in range(len(self.rec_weights)):
            weights = self.rec_weights[i]
            weights[np.abs(weights) < threshold] = 0.0

    def reset(self):
        self.v.fill(0)
        self.refractory.fill(0)
        self.stp_gains = [np.ones(len(w), dtype=np.float32) for w in self.rec_weights]
        self.spike_trace.fill(0)

    def relax(self, steps=10):
        for _ in range(steps):
            self.v *= self.decay
            self.refractory = np.maximum(0, self.refractory - 1)
            for i in range(len(self.stp_gains)):
                if len(self.stp_gains[i]) > 0:
                    self.stp_gains[i] += (1.0 - self.stp_gains[i]) * 0.01

class SaraGPT:
    """v43: Echo Training & Stabilized Readout"""
    def __init__(self, sdr_size: int = 1024):
        self.sdr_size = sdr_size
        self.encoder = SDREncoder(sdr_size, density=0.02)
        
        self.h_size = 2000
        self.total_hidden = self.h_size * 3
        
        self.l1 = KWtaLiquidLayer(sdr_size, self.h_size, decay=0.3, density=0.05, 
                                  input_scale=3.0, rec_scale=1.2, feedback_scale=0.5)
        self.l2 = KWtaLiquidLayer(self.h_size, self.h_size, decay=0.6, density=0.05, 
                                  input_scale=2.0, rec_scale=1.5, feedback_scale=0.8)
        self.l3 = KWtaLiquidLayer(self.h_size, self.h_size, decay=0.95, density=0.05, 
                                  input_scale=1.5, rec_scale=1.8, feedback_scale=0.0)
        
        self.layers = [self.l1, self.l2, self.l3]
        self.offsets = [0, self.h_size, self.h_size * 2]
        self.prev_spikes = [[], [], []]
        
        self.episodic_memory = []
        self.max_episodic_size = 150

        # Readout: Lower random init
        self.readout_weights = []
        for _ in range(sdr_size):
            fan_in = 600
            idx = np.random.choice(self.total_hidden, fan_in, replace=False)
            w = np.random.rand(fan_in).astype(np.float32) * 0.05 # Lower init (0.2 -> 0.05)
            self.readout_weights.append({'idx': idx, 'w': w})
        
        self.readout_v = np.zeros(sdr_size, dtype=np.float32)
        self.readout_decay = 0.90 # Faster decay to avoid accumulation
        self.readout_thresh = 0.6 # Higher threshold
        self.readout_refractory = np.zeros(sdr_size, dtype=np.float32)
        
        self.lr = 0.05 # Lower base LR

    def reset_state(self):
        for layer in self.layers:
            layer.reset()
        self.prev_spikes = [[], [], []]
        self.readout_v.fill(0)
        self.readout_refractory.fill(0)

    def relax(self, steps: int = 20):
        for layer in self.layers:
            layer.relax(steps)
        self.readout_v *= (self.readout_decay ** steps)
        self.readout_refractory = np.maximum(0, self.readout_refractory - steps)

    def save_model(self, filepath: str):
        state = {
            'version': 'v43',
            'sdr_size': self.sdr_size,
            'layers': [
                {
                    'in_idx': l.in_indices, 'in_w': l.in_weights,
                    'rec_idx': l.rec_indices, 'rec_w': l.rec_weights,
                    'dynamic_thresh': l.dynamic_thresh,
                    'feedback_w': l.feedback_weights
                }
                for l in self.layers
            ],
            'readout_weights': self.readout_weights,
            'encoder_cache': self.encoder.cache
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        saved_layers = state['layers']
        for i, layer_data in enumerate(saved_layers):
            target_layer = self.layers[i]
            target_layer.in_indices = layer_data['in_idx']
            target_layer.in_weights = layer_data['in_w']
            target_layer.rec_indices = layer_data['rec_idx']
            target_layer.rec_weights = layer_data['rec_w']
            target_layer.stp_gains = [np.ones(len(w), dtype=np.float32) 
                                     for w in target_layer.rec_weights]
            if 'dynamic_thresh' in layer_data:
                target_layer.dynamic_thresh = layer_data['dynamic_thresh']
            if 'feedback_w' in layer_data:
                target_layer.feedback_weights = layer_data['feedback_w']

        self.readout_weights = state['readout_weights']
        if 'encoder_cache' in state:
            self.encoder.cache = state['encoder_cache']
        
        self.reset_state()
        print(f"Model loaded from {filepath}")

    def forward_step(self, input_sdr: List[int], training: bool = False, 
                    force_output: bool = False) -> Tuple[List[int], List[int]]:
        
        spikes_1 = self.l1.forward_with_feedback(input_sdr, self.prev_spikes[0], 
                                                 feedback_active=self.prev_spikes[1], 
                                                 learning=False)
        spikes_2 = self.l2.forward_with_feedback(spikes_1, self.prev_spikes[1], 
                                                 feedback_active=self.prev_spikes[2], 
                                                 learning=False)
        spikes_3 = self.l3.forward_with_feedback(spikes_2, self.prev_spikes[2], 
                                                 feedback_active=[], 
                                                 learning=False)
        
        self.prev_spikes = [spikes_1, spikes_2, spikes_3]
        
        all_spikes = []
        all_spikes.extend(spikes_1)
        all_spikes.extend([x + self.offsets[1] for x in spikes_2])
        all_spikes.extend([x + self.offsets[2] for x in spikes_3])
        
        self.readout_v *= self.readout_decay
        self.readout_refractory = np.maximum(0, self.readout_refractory - 1)
        
        # Noise (reduced)
        if not training:
            noise = np.random.normal(0, 0.02, self.sdr_size).astype(np.float32)
            self.readout_v += noise

        if len(all_spikes) > 0:
            all_spikes_set = set(all_spikes)
            for out_idx in range(self.sdr_size):
                if self.readout_refractory[out_idx] > 0: continue
                
                mapping = self.readout_weights[out_idx]
                indices = mapping['idx']
                weights = mapping['w']
                
                active_mask = np.isin(indices, list(all_spikes_set))
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
                self.readout_refractory[predicted_sdr] = 3.0 # Increase ref
        
        if len(predicted_sdr) == 0 and force_output:
            if np.max(self.readout_v) > -999:
                best_idx = np.argmax(self.readout_v)
                predicted_sdr = [best_idx]
                if not training:
                    self.readout_v[best_idx] = 0
                    self.readout_refractory[best_idx] = 3.0
        
        return predicted_sdr, all_spikes

    def train_sequence(self, text_sequence: List[str]):
        self.reset_state()
        self._learn_tokens(text_sequence + ["<eos>"], boost=True)

    def listen(self, text: str, online_learning: bool = True):
        words = text.split()
        if not words: return
        self.episodic_memory.append(words)
        if len(self.episodic_memory) > self.max_episodic_size:
            self.episodic_memory.pop(0)
        
        if online_learning:
            for _ in range(5):
                self._learn_tokens(words, reset=False, boost=True)
        else:
            for w in words:
                sdr = self.encoder.encode(w)
                self.forward_step(sdr, training=False)

    def dream(self, cycles: int = 10):
        if not self.episodic_memory: return
        for _ in range(cycles):
            episode = random.choice(self.episodic_memory)
            self._learn_tokens(episode, reset=True, boost=True)
        for layer in self.layers:
            layer.prune_synapses()

    def think(self, length: int = 20, vocabulary: List[str] = [], trigger_text: str = "") -> str:
        self.readout_refractory.fill(0)
        
        if trigger_text:
            triggers = trigger_text.split()
            for w in triggers[-3:]:
                sdr = self.encoder.encode(w)
                self.forward_step(sdr, training=False, force_output=False)
        
        generated = []
        empty_sdr = []
        search_vocab = vocabulary + ["<eos>"] if "<eos>" not in vocabulary else vocabulary
        recent_window = []
        
        for i in range(length):
            # v43: Echo Inference is naturally handled here
            predicted_sdr, _ = self.forward_step(empty_sdr, training=False, force_output=True)
            next_word = self.encoder.decode(predicted_sdr, search_vocab)
            
            if i == 0 and next_word == "<eos>":
                candidates = [w for w in vocabulary if w != "<eos>"]
                if candidates: next_word = np.random.choice(candidates)
            
            if len(recent_window) >= 2: recent_window.pop(0)
            if next_word in recent_window:
                candidates = [w for w in vocabulary if w not in recent_window and w != "<eos>"]
                if candidates: next_word = np.random.choice(candidates[:3])
            
            recent_window.append(next_word)
            
            if next_word == "" or next_word == "<unk>":
                if len(vocabulary) > 0:
                    next_word = np.random.choice(vocabulary)
                    empty_sdr = self.encoder.encode(next_word)
                else: break
            else:
                empty_sdr = self.encoder.encode(next_word)
            
            if next_word == "<eos>": break
            generated.append(next_word)
        
        return " ".join(generated)

    def _learn_tokens(self, tokens: List[str], reset: bool = False, boost: bool = False):
        """v43: Echo Training Implementation"""
        if reset: self.reset_state()
        
        sdr_sequence = [self.encoder.encode(w) for w in tokens]
        if tokens[-1] != "<eos>": sdr_sequence.append(self.encoder.encode("<eos>"))
        
        base_lr = 0.5 if boost else self.lr # Moderate LR
        
        for t in range(len(sdr_sequence) - 1):
            input_sdr = sdr_sequence[t]
            target_sdr = sdr_sequence[t + 1]
            
            # 1. Stimulate with Input
            self.forward_step(input_sdr, training=True, force_output=False)
            
            # 2. Echo Step (Empty Input) - This state maps to target
            _, active_spikes = self.forward_step([], training=True, force_output=False)
            
            if len(active_spikes) == 0: continue
            
            target_set = set(target_sdr)
            active_spikes_set = set(active_spikes)
            
            for out_idx in range(self.sdr_size):
                mapping = self.readout_weights[out_idx]
                indices = mapping['idx']
                weights = mapping['w']
                
                active_mask = np.isin(indices, list(active_spikes_set))
                if not np.any(active_mask): continue
                
                if out_idx in target_set:
                    # Positive Update (Controlled)
                    delta = base_lr * 0.05 # Very small steps
                    weights[active_mask] += delta
                else:
                    # Negative Update
                    delta = base_lr * -0.01 
                    weights[active_mask] += delta
                
                # Clip to small range (Normalize)
                np.clip(weights, 0.0, 1.0, out=weights)