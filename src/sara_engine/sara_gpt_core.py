# src/sara_engine/sara_gpt_core.py
# SARA-GPT Core Logic (v40: Orthogonal Encoder & Criticality Tuning)

import numpy as np
import hashlib
import pickle
import random
from typing import List, Dict, Tuple, Optional
from .core import LiquidLayer 

class SDREncoder:
    """v40: Orthogonal Semantic SDR"""
    def __init__(self, input_size: int, density: float = 0.02):
        self.input_size = input_size
        self.density = density
        self.cache: Dict[str, List[int]] = {}

    def encode(self, text: str) -> List[int]:
        if text in self.cache: return self.cache[text]
        
        # v40: 短い単語や特殊トークンは「完全ハッシュ」で直交化させる
        # これにより "who" と "hot" のような偶然の重なり（N-gram衝突）を防ぐ
        if len(text) <= 3 or text in ["<eos>", "<unk>"]:
            hash_obj = hashlib.sha256(text.encode())
            seed = int(hash_obj.hexdigest(), 16) % (2**32)
            rng = np.random.RandomState(seed)
            target_n = int(self.input_size * self.density)
            indices = rng.choice(self.input_size, target_n, replace=False)
            indices.sort()
            result = indices.tolist()
            self.cache[text] = result
            return result

        # 長い単語はこれまで通りN-gramで意味の近さを表現する
        grams = []
        padded = "^" + text + "$"
        for n in [2, 3]:
            if len(padded) < n: continue
            for i in range(len(padded) - n + 1):
                grams.append(padded[i:i+n])
        if not grams: grams = [text]

        active_indices = set()
        bits_per_gram = 15 
        for g in grams:
            g_hash = int(hashlib.sha256(g.encode()).hexdigest(), 16)
            g_rng = np.random.RandomState(g_hash % (2**32))
            bits = g_rng.choice(self.input_size, bits_per_gram, replace=False)
            active_indices.update(bits)

        result = sorted(list(active_indices))
        target_n = int(self.input_size * self.density)
        if len(result) > target_n:
            text_hash = int(hashlib.sha256(text.encode()).hexdigest(), 16)
            t_rng = np.random.RandomState(text_hash % (2**32))
            result = sorted(t_rng.choice(result, target_n, replace=False).tolist())
        elif len(result) < target_n:
             text_hash = int(hashlib.sha256(text.encode()).hexdigest(), 16)
             t_rng = np.random.RandomState(text_hash % (2**32))
             remaining = [i for i in range(self.input_size) if i not in active_indices]
             added = t_rng.choice(remaining, target_n - len(result), replace=False)
             result = sorted(list(active_indices) + added.tolist())

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
    """v30: K-Winner-Take-All"""
    def __init__(self, input_size: int, hidden_size: int, decay: float, density: float = 0.1, input_scale: float = 1.0, rec_scale: float = 1.0, feedback_scale: float = 0.5):
        super().__init__(input_size, hidden_size, decay, input_scale=input_scale, rec_scale=rec_scale, density=density)
        
        self.size = hidden_size
        self.stp_gains = [np.ones(len(w), dtype=np.float32) for w in self.rec_weights]
        self.facilitation_rate = 0.2
        self.max_gain = 3.0
        self.recovery_rate = 0.05
        self.target_rate = 0.05 
        self.homeostasis_rate = 0.005
        self.dynamic_thresh = np.copy(self.thresh)

        # Feedback
        self.feedback_scale = feedback_scale
        self.feedback_weights = []
        rng = np.random.RandomState(42)
        for i in range(hidden_size):
             targets = rng.choice(hidden_size, int(hidden_size * 0.05), replace=False)
             self.feedback_weights.append(targets)

        # STDP
        self.spike_trace = np.zeros(hidden_size, dtype=np.float32)
        self.trace_decay = 0.9 
        self.stdp_lr = 0.005

        # K-WTA (Relaxed to 10%)
        self.k_winners = int(hidden_size * 0.10) 

    def forward_with_feedback(self, active_inputs: List[int], prev_active_hidden: List[int], feedback_active: List[int] = [], learning: bool = False) -> List[int]:
        self.refractory = np.maximum(0, self.refractory - 1)
        self.v *= self.decay
        
        for pre_id in active_inputs:
            if pre_id < len(self.in_indices):
                targets = self.in_indices[pre_id]
                ws = self.in_weights[pre_id]
                if len(targets) > 0: self.v[targets] += ws

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

        if len(feedback_active) > 0:
            for fb_id in feedback_active:
                if fb_id < len(self.feedback_weights):
                    targets = self.feedback_weights[fb_id]
                    self.v[targets] += self.feedback_scale

        for i in range(len(self.stp_gains)):
             if len(self.stp_gains[i]) > 0:
                self.stp_gains[i] += (1.0 - self.stp_gains[i]) * self.recovery_rate

        candidates_mask = (self.v >= self.dynamic_thresh) & (self.refractory <= 0)
        candidates_indices = np.where(candidates_mask)[0]
        
        fired_indices = []
        if len(candidates_indices) > 0:
            if len(candidates_indices) > self.k_winners:
                potentials = self.v[candidates_indices]
                top_k_local_indices = np.argsort(potentials)[-self.k_winners:]
                fired_indices_np = candidates_indices[top_k_local_indices]
                fired_indices = fired_indices_np.tolist()
            else:
                fired_indices = candidates_indices.tolist()
            
            fired_idx_arr = np.array(fired_indices, dtype=int)
            self.dynamic_thresh[fired_idx_arr] += self.homeostasis_rate * (1.0 - self.target_rate)
            self.v[fired_idx_arr] = 0.0 
            self.refractory[fired_idx_arr] = 4.0 

        not_fired_mask = np.ones(self.size, dtype=bool)
        if len(fired_indices) > 0:
            not_fired_mask[np.array(fired_indices, dtype=int)] = False
        
        self.dynamic_thresh[not_fired_mask] -= self.homeostasis_rate * self.target_rate
        np.clip(self.dynamic_thresh, 0.1, 5.0, out=self.dynamic_thresh)

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

        self.spike_trace *= self.trace_decay
        if len(fired_indices) > 0:
             self.spike_trace[np.array(fired_indices, dtype=int)] += 1.0

        return fired_indices

    def prune_synapses(self, threshold: float = 0.01):
        for i in range(len(self.rec_weights)):
            weights = self.rec_weights[i]
            weights[np.abs(weights) < threshold] = 0.0

    def reset(self):
        super().reset()
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
    def __init__(self, sdr_size: int = 1024):
        self.sdr_size = sdr_size
        self.encoder = SDREncoder(sdr_size, density=0.02)
        
        self.h_size = 2000
        self.total_hidden = self.h_size * 3
        
        # v40: Criticality Tuning (Edge of Chaos)
        # rec_scale を 3.0 から 1.3~1.5 へ下げる。カオス回避と安定化のため。
        # input_scale は高めに維持して「入力への反応」を良くする。
        self.l1 = KWtaLiquidLayer(sdr_size, self.h_size, decay=0.3, density=0.05, input_scale=5.0, rec_scale=1.3, feedback_scale=0.5)
        self.l2 = KWtaLiquidLayer(self.h_size, self.h_size, decay=0.6, density=0.05, input_scale=2.0, rec_scale=1.4, feedback_scale=0.8)
        self.l3 = KWtaLiquidLayer(self.h_size, self.h_size, decay=0.95, density=0.05, input_scale=1.5, rec_scale=1.5, feedback_scale=0.0)
        
        self.layers = [self.l1, self.l2, self.l3]
        self.offsets = [0, self.h_size, self.h_size * 2]
        self.prev_spikes = [[], [], []]
        
        self.episodic_memory = []
        self.max_episodic_size = 100

        self.readout_weights = []
        self.eligibility_traces = [] 
        for _ in range(sdr_size):
            fan_in = 400
            idx = np.random.choice(self.total_hidden, fan_in, replace=False)
            w = np.random.rand(fan_in).astype(np.float32) * 0.1
            self.readout_weights.append({'idx': idx, 'w': w})
            self.eligibility_traces.append(np.zeros(fan_in, dtype=np.float32))
        
        self.readout_v = np.zeros(sdr_size, dtype=np.float32)
        self.readout_decay = 0.95 
        self.readout_thresh = 0.5 
        self.readout_refractory = np.zeros(sdr_size, dtype=np.float32)
        
        self.lr = 0.05 

    def reset_state(self):
        for layer in self.layers:
            layer.reset()
        self.prev_spikes = [[], [], []]
        self.readout_v.fill(0)
        self.readout_refractory.fill(0)
        for trace in self.eligibility_traces:
            trace.fill(0)

    def relax(self, steps: int = 20):
        for layer in self.layers:
            layer.relax(steps)
        self.readout_v *= (self.readout_decay ** steps)
        self.readout_refractory = np.maximum(0, self.readout_refractory - steps)

    def save_model(self, filepath: str):
        state = {
            'version': 'v40',
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
            'readout_weights': self.readout_weights
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        if state.get('sdr_size') != self.sdr_size:
            print(f"Warning: SDR size mismatch.")

        saved_layers = state['layers']
        for i, layer_data in enumerate(saved_layers):
            target_layer = self.layers[i]
            target_layer.in_indices = layer_data['in_idx']
            target_layer.in_weights = layer_data['in_w']
            target_layer.rec_indices = layer_data['rec_idx']
            target_layer.rec_weights = layer_data['rec_w']
            target_layer.stp_gains = [np.ones(len(w), dtype=np.float32) for w in target_layer.rec_weights]
            if 'dynamic_thresh' in layer_data: target_layer.dynamic_thresh = layer_data['dynamic_thresh']
            if 'feedback_w' in layer_data: target_layer.feedback_weights = layer_data['feedback_w']

        self.readout_weights = state['readout_weights']
        self.reset_state()
        print(f"Model loaded from {filepath}")

    def forward_step(self, input_sdr: List[int], training: bool = False, force_output: bool = False, internal_learning: bool = False) -> Tuple[List[int], List[int]]:
        spikes_1 = self.l1.forward_with_feedback(input_sdr, self.prev_spikes[0], feedback_active=self.prev_spikes[1], learning=internal_learning)
        spikes_2 = self.l2.forward_with_feedback(spikes_1, self.prev_spikes[1], feedback_active=self.prev_spikes[2], learning=internal_learning)
        spikes_3 = self.l3.forward_with_feedback(spikes_2, self.prev_spikes[2], feedback_active=[], learning=internal_learning)
        
        self.prev_spikes = [spikes_1, spikes_2, spikes_3]
        
        all_spikes = []
        all_spikes.extend(spikes_1)
        all_spikes.extend([x + self.offsets[1] for x in spikes_2])
        all_spikes.extend([x + self.offsets[2] for x in spikes_3])
        
        self.readout_v *= self.readout_decay
        self.readout_refractory = np.maximum(0, self.readout_refractory - 1)
        
        if not training:
            noise = np.random.normal(0, 0.05, self.sdr_size).astype(np.float32)
            self.readout_v += noise

        current_inputs = np.zeros(self.sdr_size, dtype=np.float32)
        if len(all_spikes) > 0:
            for out_idx in range(self.sdr_size):
                if self.readout_refractory[out_idx] > 0: continue
                mapping = self.readout_weights[out_idx]
                indices = mapping['idx']
                weights = mapping['w']
                traces = self.eligibility_traces[out_idx]
                
                traces *= 0.9 
                mask = np.isin(indices, all_spikes)
                traces[mask] += 1.0 
                
                if np.any(mask):
                    current_inputs[out_idx] = np.sum(weights[mask])
        
        self.readout_v += current_inputs
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
                self.readout_refractory[predicted_sdr] = 2.0
        
        if len(predicted_sdr) == 0 and force_output:
            if np.max(self.readout_v) > -999:
                best_idx = np.argmax(self.readout_v)
                predicted_sdr = [best_idx]
                if not training:
                    self.readout_v[best_idx] = 0
                    self.readout_refractory[best_idx] = 2.0
            
        return predicted_sdr, all_spikes

    def train_sequence(self, text_sequence: List[str]):
        self.reset_state()
        self._learn_tokens(text_sequence + ["<eos>"], boost=False)

    def listen(self, text: str, online_learning: bool = True):
        words = text.split()
        if not words: return
        self.episodic_memory.append(words)
        if len(self.episodic_memory) > self.max_episodic_size: self.episodic_memory.pop(0)

        if online_learning:
            for _ in range(2):
                self._learn_tokens(words, reset=False, boost=True, stdp=True)
        else:
            for w in words:
                sdr = self.encoder.encode(w)
                self.forward_step(sdr, training=False)

    def dream(self, cycles: int = 5):
        if not self.episodic_memory: return
        for _ in range(cycles):
            episode = random.choice(self.episodic_memory)
            self._learn_tokens(episode, reset=True, boost=True, stdp=True)
        for layer in self.layers:
            layer.prune_synapses()

    def think(self, length: int = 20, vocabulary: List[str] = [], trigger_text: str = "") -> str:
        self.readout_refractory.fill(0)
        
        if trigger_text:
            triggers = trigger_text.split()
            for w in triggers[-2:]:
                sdr = self.encoder.encode(w)
                self.forward_step(sdr, training=False, force_output=False)

        generated = []
        empty_sdr = [] 
        search_vocab = vocabulary + ["<eos>"] if "<eos>" not in vocabulary else vocabulary

        for i in range(length):
            predicted_sdr, _ = self.forward_step(empty_sdr, training=False, force_output=True)
            next_word = self.encoder.decode(predicted_sdr, search_vocab)
            
            if i == 0 and next_word == "<eos>":
                 candidates = [w for w in vocabulary if w != "<eos>"]
                 if candidates:
                     next_word = np.random.choice(candidates)
            
            if next_word == "" or next_word == "<unk>":
                 if len(vocabulary) > 0:
                     next_word = np.random.choice(vocabulary)
                     empty_sdr = self.encoder.encode(next_word)
                 else:
                     break
            else:
                 empty_sdr = self.encoder.encode(next_word)

            if next_word == "<eos>":
                break
            
            generated.append(next_word)

        return " ".join(generated)

    def _learn_tokens(self, tokens: List[str], reset: bool = False, boost: bool = False, stdp: bool = False):
        if reset: self.reset_state()
        sdr_sequence = [self.encoder.encode(w) for w in tokens]
        if tokens[-1] != "<eos>": sdr_sequence.append(self.encoder.encode("<eos>"))
        
        for t in range(len(sdr_sequence) - 1):
            input_sdr = sdr_sequence[t]
            target_sdr = sdr_sequence[t+1]
            predicted_sdr, active_spikes = self.forward_step(input_sdr, training=True, force_output=True, internal_learning=stdp)
            
            base_lr = 0.5 if boost else self.lr
            target_set = set(target_sdr)
            
            for out_idx in range(self.sdr_size):
                mapping = self.readout_weights[out_idx]
                traces = self.eligibility_traces[out_idx]
                if np.max(traces) < 0.01: continue
                
                dopamine = 0.0
                if out_idx in target_set: dopamine = 3.0 
                elif out_idx in predicted_sdr: dopamine = -0.5 
                
                if dopamine != 0.0:
                    delta = base_lr * dopamine * traces
                    mapping['w'] += delta
                    np.clip(mapping['w'], 0.0, 5.0, out=mapping['w'])

            if t % 10 == 0:
                decay = 0.995
                for out_idx in range(self.sdr_size):
                     self.readout_weights[out_idx]['w'] *= decay