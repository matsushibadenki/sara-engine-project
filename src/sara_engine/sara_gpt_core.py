# src/sara_engine/sara_gpt_core.py
# SARA-GPT Core Logic (v18: Repetition Penalty & Stable Learning)

import numpy as np
import hashlib
import pickle
import random
from typing import List, Dict, Tuple, Optional
from .core import LiquidLayer 

class SDREncoder:
    def __init__(self, input_size: int, density: float = 0.02):
        self.input_size = input_size
        self.density = density
        self.cache: Dict[str, List[int]] = {}

    def encode(self, text: str) -> List[int]:
        if text in self.cache:
            return self.cache[text]

        hash_obj = hashlib.sha256(text.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        
        n_active = max(1, int(self.input_size * self.density))
        indices = rng.choice(self.input_size, n_active, replace=False)
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

class AttentionLiquidLayer(LiquidLayer):
    def __init__(self, input_size: int, hidden_size: int, decay: float, density: float = 0.1, input_scale: float = 1.0, rec_scale: float = 1.0):
        super().__init__(input_size, hidden_size, decay, input_scale=input_scale, rec_scale=rec_scale, density=density)
        self.stp_gains = [np.ones_like(w) for w in self.rec_weights]
        self.facilitation_rate = 0.1 
        self.recovery_rate = 0.1

    def forward_with_attention(self, active_inputs: List[int], prev_active_hidden: List[int]) -> List[int]:
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
                
                # STP Gain
                if pre_h_id < len(self.stp_gains):
                    gains = self.stp_gains[pre_h_id]
                else:
                    gains = 1.0
                
                if len(targets) > 0:
                    self.v[targets] += ws * gains
                    if pre_h_id < len(self.stp_gains):
                        self.stp_gains[pre_h_id] = np.minimum(gains + self.facilitation_rate, 2.0)

        # Gain Decay
        for i in range(len(self.stp_gains)):
             if len(self.stp_gains[i]) > 0:
                self.stp_gains[i] += (1.0 - self.stp_gains[i]) * self.recovery_rate

        ready_mask = (self.v >= self.thresh) & (self.refractory <= 0)
        fired_indices = np.where(ready_mask)[0]
        
        if len(fired_indices) > 0:
            self.v[fired_indices] -= self.thresh[fired_indices]
            self.refractory[fired_indices] = self.refractory_period
            
        return fired_indices.tolist()

    def reset(self):
        super().reset()
        self.stp_gains = [np.ones_like(w) for w in self.rec_weights]

    def relax(self, steps=10):
        for _ in range(steps):
            self.v *= self.decay
            self.refractory = np.maximum(0, self.refractory - 1)
            for i in range(len(self.stp_gains)):
                if len(self.stp_gains[i]) > 0:
                    self.stp_gains[i] += (1.0 - self.stp_gains[i]) * 0.02

class SaraGPT:
    def __init__(self, sdr_size: int = 1024):
        self.sdr_size = sdr_size
        self.encoder = SDREncoder(sdr_size, density=0.02)
        
        self.h_size = 2000
        self.total_hidden = self.h_size * 3
        
        self.l1 = AttentionLiquidLayer(sdr_size, self.h_size, decay=0.3, density=0.05, input_scale=3.0, rec_scale=1.2)
        self.l2 = AttentionLiquidLayer(self.h_size, self.h_size, decay=0.6, density=0.05, input_scale=2.0, rec_scale=1.5)
        self.l3 = AttentionLiquidLayer(self.h_size, self.h_size, decay=0.95, density=0.05, input_scale=1.5, rec_scale=2.0)
        
        self.layers = [self.l1, self.l2, self.l3]
        self.offsets = [0, self.h_size, self.h_size * 2]
        self.prev_spikes = [[], [], []]
        
        self.readout_weights = []
        for _ in range(sdr_size):
            fan_in = 400
            idx = np.random.choice(self.total_hidden, fan_in, replace=False)
            w = np.zeros(fan_in, dtype=np.float32)
            self.readout_weights.append({'idx': idx, 'w': w})
        
        self.readout_refractory = np.zeros(sdr_size, dtype=np.float32)
        self.lr = 0.05 

    def reset_state(self):
        for layer in self.layers:
            layer.reset()
        self.prev_spikes = [[], [], []]
        self.readout_refractory.fill(0)

    def relax(self, steps: int = 20):
        for layer in self.layers:
            layer.relax(steps)
        self.readout_refractory = np.maximum(0, self.readout_refractory - steps)

    def save_model(self, filepath: str):
        state = {
            'version': 'v18',
            'sdr_size': self.sdr_size,
            'layers': [
                {'in_idx': l.in_indices, 'in_w': l.in_weights, 'rec_idx': l.rec_indices, 'rec_w': l.rec_weights}
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
            target_layer.stp_gains = [np.ones_like(w) for w in target_layer.rec_weights]

        self.readout_weights = state['readout_weights']
        self.reset_state()
        print(f"Model loaded from {filepath}")

    def forward_step(self, input_sdr: List[int], training: bool = False, force_output: bool = False) -> Tuple[List[int], List[int]]:
        spikes_1 = self.l1.forward_with_attention(input_sdr, self.prev_spikes[0])
        spikes_2 = self.l2.forward_with_attention(spikes_1, self.prev_spikes[1])
        spikes_3 = self.l3.forward_with_attention(spikes_2, self.prev_spikes[2])
        
        self.prev_spikes = [spikes_1, spikes_2, spikes_3]
        
        all_spikes = []
        all_spikes.extend(spikes_1)
        all_spikes.extend([x + self.offsets[1] for x in spikes_2])
        all_spikes.extend([x + self.offsets[2] for x in spikes_3])
        
        predicted_potentials = np.zeros(self.sdr_size, dtype=np.float32)
        
        if len(all_spikes) > 0:
            for out_idx in range(self.sdr_size):
                mapping = self.readout_weights[out_idx]
                indices = mapping['idx']
                weights = mapping['w']
                
                mask = np.isin(indices, all_spikes)
                if np.any(mask):
                    predicted_potentials[out_idx] = np.sum(weights[mask])

        is_refractory = self.readout_refractory > 0
        predicted_potentials[is_refractory] = -999.0
        self.readout_refractory = np.maximum(0, self.readout_refractory - 1)

        threshold = 0.5
        fired_mask = predicted_potentials > threshold
        
        predicted_sdr = []
        
        if not np.any(fired_mask) and force_output:
            if np.max(predicted_potentials) > -100:
                best_idx = np.argmax(predicted_potentials)
                predicted_sdr = [best_idx]
        elif np.any(fired_mask):
            candidates = np.where(fired_mask)[0]
            potentials = predicted_potentials[candidates]
            n_predict = max(1, int(self.sdr_size * 0.02)) 
            if len(candidates) > n_predict:
                 top_indices = np.argsort(potentials)[-n_predict:]
                 predicted_sdr = candidates[top_indices].tolist()
            else:
                 predicted_sdr = candidates.tolist()
        
        if not training and len(predicted_sdr) > 0:
            self.readout_refractory[predicted_sdr] = 2.0
            
        return predicted_sdr, all_spikes

    def train_sequence(self, text_sequence: List[str]):
        self.reset_state()
        self._learn_tokens(text_sequence + ["<eos>"], boost=False)

    def listen(self, text: str, online_learning: bool = True):
        words = text.split()
        if not words: return
        
        if online_learning:
            # v18: リハーサル回数を5回 -> 2回に減らし、過学習(ループ)を防ぐ
            for _ in range(2):
                self._learn_tokens(words, reset=False, boost=True)
        else:
            for w in words:
                sdr = self.encoder.encode(w)
                self.forward_step(sdr, training=False)

    def think(self, length: int = 20, vocabulary: List[str] = [], trigger_text: str = "") -> str:
        generated = []
        empty_sdr = [] 
        search_vocab = vocabulary + ["<eos>"] if "<eos>" not in vocabulary else vocabulary

        for i in range(length):
            predicted_sdr, _ = self.forward_step(empty_sdr, training=False, force_output=True)
            
            next_word = self.encoder.decode(predicted_sdr, search_vocab)
            
            # --- v18: Repetition Penalty (N-gram Blocking) ---
            # 直前の2単語と同じ単語が出たら、強制的に他の候補に変える
            is_repetitive = False
            if len(generated) >= 1 and generated[-1] == next_word:
                is_repetitive = True
            if len(generated) >= 2 and generated[-2] == next_word:
                is_repetitive = True
            
            if is_repetitive:
                # 罰則：ランダムな他の単語、あるいはトリガーから選ぶ
                if trigger_text and random.random() < 0.5:
                     triggers = trigger_text.split()
                     if triggers:
                         next_word = random.choice(triggers)
                else:
                     # 語彙リストからランダム (ただし現在の単語は除く)
                     candidates = [w for w in vocabulary if w != next_word]
                     if candidates:
                         next_word = random.choice(candidates)

            # --- End Repetition Penalty ---

            if next_word == "<eos>":
                if len(generated) == 0 and trigger_text:
                    triggers = trigger_text.split()
                    if triggers:
                        trigger_word = triggers[-1]
                        if trigger_word in vocabulary:
                            empty_sdr = self.encoder.encode(trigger_word)
                            continue
                break

            if next_word == "":
                 if len(vocabulary) > 0:
                     next_word = np.random.choice(vocabulary)
                 else:
                     break
            
            generated.append(next_word)
            empty_sdr = self.encoder.encode(next_word)

        return " ".join(generated)

    def _learn_tokens(self, tokens: List[str], reset: bool = False, boost: bool = False):
        if reset: self.reset_state()
        sdr_sequence = [self.encoder.encode(w) for w in tokens]
        
        if tokens[-1] != "<eos>":
             sdr_sequence.append(self.encoder.encode("<eos>"))
        
        for t in range(len(sdr_sequence) - 1):
            input_sdr = sdr_sequence[t]
            target_sdr = sdr_sequence[t+1]
            
            predicted_sdr, active_spikes = self.forward_step(input_sdr, training=True, force_output=True)
            if len(active_spikes) == 0: continue

            target_set = set(target_sdr)
            base_lr = 0.5 if boost else self.lr
            
            for out_idx in target_sdr:
                mapping = self.readout_weights[out_idx]
                mask = np.isin(mapping['idx'], active_spikes)
                mapping['w'][mask] += base_lr * (5.0 - mapping['w'][mask])

            for out_idx in predicted_sdr:
                if out_idx not in target_set:
                    mapping = self.readout_weights[out_idx]
                    mask = np.isin(mapping['idx'], active_spikes)
                    mapping['w'][mask] -= base_lr * 1.5 * mapping['w'][mask]

            if t % 10 == 0:
                decay = 0.995
                for out_idx in range(self.sdr_size):
                     self.readout_weights[out_idx]['w'] *= decay