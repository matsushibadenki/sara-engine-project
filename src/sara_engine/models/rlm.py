import numpy as np
import pickle
import re
from typing import List, Dict, Tuple, Any, Optional
from collections import deque, OrderedDict
import sys
import os

from ..core.layers import DynamicLiquidLayer
from ..core.attention import SpikeAttention
from ..memory.sdr import SDREncoder

# LTM Import
try:
    from ..memory.ltm import SparseMemoryStore
except ImportError:
    pass

# --- Components ---
class WorkingMemory:
    def __init__(self, capacity: int = 10, memory_size: int = 500):
        self.capacity = capacity
        self.memory_size = memory_size
        self.buffer: deque = deque(maxlen=capacity)
        self.wm_neurons = np.zeros(memory_size, dtype=np.float32)
        self.wm_decay = 0.95
        
    def store(self, pattern: List[int], importance: float = 1.0):
        self.buffer.append({'pattern': pattern[:100], 'importance': importance, 'age': 0})
        for idx in pattern:
            if idx < self.memory_size:
                self.wm_neurons[idx] = min(self.wm_neurons[idx] + importance, 3.0)
    
    def get_context_spikes(self, threshold: float = 0.3) -> List[int]:
        active = np.where(self.wm_neurons > threshold)[0]
        return active.tolist()
    
    def update(self):
        self.wm_neurons *= self.wm_decay
        for item in self.buffer: item['age'] += 1
    
    def reset(self):
        self.buffer.clear()
        self.wm_neurons.fill(0)

class StateNeuronGroup:
    def __init__(self, num_states: int = 5):
        self.num_states = num_states
        self.state_names = ["INIT", "SEARCH", "READ", "EXTRACT", "DONE"]
        self.activations = np.zeros(num_states, dtype=np.float32)
        self.transition_matrix = np.eye(num_states, dtype=np.float32) * 0.5
        self.transition_matrix += np.random.uniform(0, 0.1, (num_states, num_states))
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        self.transition_matrix /= row_sums
        self.current_state = 0 
        self.activations[0] = 1.0
        
    def update(self, input_strength: float = 0.0, context_strength: float = 0.0):
        probs = self.transition_matrix[self.current_state].copy()
        if input_strength > 0.1 and self.get_state_name() in ["INIT", "DONE"]:
            probs[self.state_names.index("SEARCH")] += 0.2
        noise = np.random.normal(0, 0.05, self.num_states)
        probs = np.maximum(0, probs + noise)
        if probs.sum() > 0: probs /= probs.sum()
        else: probs[self.current_state] = 1.0
        next_state = int(np.argmax(probs))
        self.activations.fill(0)
        self.activations[next_state] = 1.0
        self.current_state = next_state
        
    def learn_transition(self, prev_state_idx: int, current_state_idx: int, reward: float):
        self.transition_matrix[prev_state_idx, current_state_idx] += 0.1 * reward
        self.transition_matrix = np.maximum(0, self.transition_matrix)
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.transition_matrix /= row_sums

    def get_state_name(self) -> str: return self.state_names[self.current_state]
    def get_state_index(self, name: str) -> int: return self.state_names.index(name) if name in self.state_names else -1
    def set_state(self, state_name: str):
        if state_name in self.state_names:
            idx = self.state_names.index(state_name)
            self.activations.fill(0)
            self.activations[idx] = 1.0
            self.current_state = idx

# --- Stateful Brain ---
class StatefulSaraGPT:
    def __init__(self, sdr_size: int = 1024):
        self.sdr_size = sdr_size
        self.encoder = SDREncoder(sdr_size, density=0.02)
        self.h_size = 2000
        self.total_hidden = self.h_size * 3
        
        self.l1 = DynamicLiquidLayer(sdr_size, self.h_size, decay=0.3, density=0.08, input_scale=2.0)
        self.l2 = DynamicLiquidLayer(self.h_size, self.h_size, decay=0.6, density=0.08, input_scale=1.5)
        self.l3 = DynamicLiquidLayer(self.h_size, self.h_size, decay=0.92, density=0.08, input_scale=1.2)
        
        self.layers = [self.l1, self.l2, self.l3]
        self.offsets = [0, self.h_size, self.h_size * 2]
        self.prev_spikes: List[List[int]] = [[], [], []]
        
        self.attention = SpikeAttention(input_size=self.h_size, hidden_size=500, memory_size=60)
        self.attention_active = True
        
        self.working_memory = WorkingMemory(capacity=10, memory_size=500)
        self.state_neurons = StateNeuronGroup(num_states=5)
        self.state_readout_weights: Dict[str, List[Dict[str, Any]]] = {}
        
        for state in self.state_neurons.state_names:
            weights_list = []
            for _ in range(sdr_size):
                fan_in = 600
                max_idx = self.total_hidden + 500 
                idx = np.random.choice(max_idx, fan_in, replace=False)
                w = np.random.normal(0, 0.05, fan_in).astype(np.float32)
                weights_list.append({'idx': idx, 'w': w})
            self.state_readout_weights[state] = weights_list
            
        self.readout_v = np.zeros(sdr_size, dtype=np.float32)
        self.readout_decay = 0.85
        self.readout_thresh = 0.5
        self.readout_refractory = np.zeros(sdr_size, dtype=np.float32)
        self.step_counter = 0

    def reset_state(self):
        for layer in self.layers: layer.reset()
        self.attention.reset()
        self.working_memory.reset()
        self.state_neurons.set_state("INIT")
        self.prev_spikes = [[], [], []]
        self.readout_v.fill(0)
        self.readout_refractory.fill(0)
        self.step_counter = 0

    def forward_step(self, input_sdr: List[int], training: bool = False, 
                    force_output: bool = False) -> Tuple[List[int], List[int], str]:
        
        context_spikes = self.working_memory.get_context_spikes(threshold=0.4)
        self.state_neurons.update(input_strength=len(input_sdr)/100.0, context_strength=len(context_spikes)/100.0)
        current_state_name = self.state_neurons.get_state_name()
        
        combined_input = list(set(input_sdr + context_spikes))
        
        spikes_1 = self.l1.forward_with_feedback(combined_input, self.prev_spikes[0], feedback_active=self.prev_spikes[1], learning=training)
        spikes_2 = self.l2.forward_with_feedback(spikes_1, self.prev_spikes[1], feedback_active=self.prev_spikes[2], learning=training)
        
        attn_signal = []
        if self.attention_active:
            attn_signal = self.attention.compute(spikes_2)
            if len(spikes_2) > 0: self.attention.update_memory(spikes_2)
        
        spikes_3 = self.l3.forward_with_feedback(spikes_2, self.prev_spikes[2], feedback_active=[], learning=training, attention_signal=attn_signal)
        
        self.prev_spikes = [spikes_1, spikes_2, spikes_3]
        
        all_hidden_spikes = []
        all_hidden_spikes.extend(spikes_1)
        all_hidden_spikes.extend([x + self.offsets[1] for x in spikes_2])
        all_hidden_spikes.extend([x + self.offsets[2] for x in spikes_3])
        
        wm_offset = self.total_hidden
        wm_readout_spikes = [x + wm_offset for x in context_spikes]
        readout_input_spikes = all_hidden_spikes + wm_readout_spikes
        
        self.readout_v *= self.readout_decay
        self.readout_refractory = np.maximum(0, self.readout_refractory - 1)
        
        if not training:
            noise = np.random.normal(0, 0.01, self.sdr_size).astype(np.float32)
            self.readout_v += noise

        if len(readout_input_spikes) > 0:
            current_weights = self.state_readout_weights[current_state_name]
            for out_idx in range(self.sdr_size):
                if self.readout_refractory[out_idx] > 0: continue
                mapping = current_weights[out_idx]
                indices = mapping['idx']
                weights = mapping['w']
                active_mask = np.isin(indices, readout_input_spikes)
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
        
        if len(predicted_sdr) > 0:
            importance = 1.0
            if current_state_name == "SEARCH": importance = 1.5
            if current_state_name == "EXTRACT": importance = 2.0
            self.working_memory.store(predicted_sdr, importance)
            
        self.working_memory.update()
        self.step_counter += 1
        
        return predicted_sdr, all_hidden_spikes, current_state_name

    def reinforce(self, trajectory: List[Dict[str, Any]], final_reward: float):
        gamma = 0.9
        running_reward = final_reward
        for step_data in reversed(trajectory):
            state_name = step_data['state']
            next_state_name = step_data['next_state']
            step_reward = step_data.get('reward', 0.0)
            running_reward = running_reward + step_reward
            s_idx = self.state_neurons.get_state_index(state_name)
            ns_idx = self.state_neurons.get_state_index(next_state_name)
            if s_idx >= 0 and ns_idx >= 0:
                self.state_neurons.learn_transition(s_idx, ns_idx, running_reward)
            running_reward *= gamma

    def save_model(self, filepath: str):
        state = {
            'version': 'stateful_v2_rl',
            'sdr_size': self.sdr_size,
            'state_readout_weights': self.state_readout_weights,
            'encoder_cache': dict(self.encoder.cache),
            'attn_w_q': self.attention.w_query, 'attn_w_k': self.attention.w_key, 'attn_w_v': self.attention.w_value,
            'transition_matrix': self.state_neurons.transition_matrix
        }
        with open(filepath, 'wb') as f: pickle.dump(state, f)
        print(f"Stateful Model saved to {filepath}")

    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f: state = pickle.load(f)
        self.state_readout_weights = state['state_readout_weights']
        if 'encoder_cache' in state: self.encoder.cache = OrderedDict(state['encoder_cache'])
        if 'transition_matrix' in state: self.state_neurons.transition_matrix = state['transition_matrix']
        print(f"Stateful Model loaded from {filepath}")

# --- Agent ---
class StatefulRLMAgent:
    def __init__(self, model_path: Optional[str] = None):
        self.brain = StatefulSaraGPT(sdr_size=1024)
        if model_path and os.path.exists(model_path):
            self.brain.load_model(model_path)
        else:
            print("Warning: No model loaded.")
        
        self.ltm: Optional[SparseMemoryStore] = None
        try:
            from ..memory.ltm import SparseMemoryStore
            self.ltm = SparseMemoryStore(filepath="sara_ltm.pkl")
            print("LTM Module initialized.")
        except ImportError:
            print("LTM Module not found.")
            
    def solve(self, query: str, document: str = "", train_rl: bool = True) -> str:
        self.brain.reset_state()
        self.brain.working_memory.reset()
        
        q_words = query.split()
        for w in q_words:
            self.brain.working_memory.store(self.brain.encoder.encode(w), importance=2.0)
            
        chunks = [document[i:i+100] for i in range(0, len(document), 100)] if document else []
        max_steps = 20
        current_chunk_idx = 0
        found_info = ""
        
        trajectory: List[Dict[str, Any]] = []
        prev_state_name = "INIT"
        self.brain.state_neurons.set_state("INIT")
        
        if query.startswith("MEMORIZE:"):
            content_to_save = query.replace("MEMORIZE:", "").strip()
            if self.ltm:
                sdr = self.brain.encoder.encode(content_to_save)
                self.ltm.add(sdr, content_to_save)
                return f"Memorized: {content_to_save}"
            return "LTM not available."

        if query.startswith("RECALL:"):
            search_query = query.replace("RECALL:", "").strip()
            if self.ltm:
                sdr = self.brain.encoder.encode(search_query)
                results = self.ltm.search(sdr)
                if results:
                    best = results[0]
                    return f"Recalled: {best['content']} (Score: {best['score']:.2f})"
                return "Nothing found in LTM."
            return "LTM not available."

        for step in range(max_steps):
            if chunks and current_chunk_idx < len(chunks):
                visual_input_text = chunks[current_chunk_idx][:10]
            else:
                visual_input_text = "END"
            
            input_sdr = self.brain.encoder.encode(visual_input_text)
            predicted_sdr, _, state_name = self.brain.forward_step(input_sdr)
            
            if step > 0:
                trajectory.append({'state': prev_state_name, 'next_state': state_name, 'step': step})
            prev_state_name = state_name
            
            if state_name == "SEARCH":
                if step == 1 and self.ltm:
                    query_sdr = self.brain.encoder.encode(query)
                    ltm_results = self.ltm.search(query_sdr, threshold=0.1)
                    if ltm_results:
                        found_info = ltm_results[0]['content']
                        self.brain.state_neurons.set_state("DONE")
                        break
                current_text = chunks[current_chunk_idx] if chunks else ""
                if "code" in current_text or "password" in current_text or "is" in current_text: pass 
                else:
                    if chunks: current_chunk_idx = (current_chunk_idx + 1) % len(chunks)
            
            elif state_name == "READ":
                if chunks and current_chunk_idx < len(chunks):
                    content = chunks[current_chunk_idx]
                    for word in content.split(): self.brain.forward_step(self.brain.encoder.encode(word), training=False)
            
            elif state_name == "EXTRACT":
                if chunks and current_chunk_idx < len(chunks):
                    chunk_text = chunks[current_chunk_idx]
                    candidates = re.findall(r'\b[A-Z0-9\-]{2,}\b', chunk_text)
                    query_upper = query.upper().split()
                    candidates = [c for c in candidates if c not in query_upper]
                    is_matches = re.findall(r'\bis\s+([A-Za-z0-9\-]+)', chunk_text, re.IGNORECASE)
                    candidates.extend(is_matches)
                    if candidates:
                        found_info = candidates[0]
                        self.brain.state_neurons.set_state("DONE")
                    else:
                        current_chunk_idx = (current_chunk_idx + 1) % len(chunks)
                else:
                    current_chunk_idx = 0
                    self.brain.state_neurons.set_state("SEARCH")
            
            elif state_name == "DONE": break
            
            if step > 8 and state_name == "INIT":
                trajectory.append({'state': 'INIT', 'next_state': 'SEARCH', 'reward': -0.05, 'step': step})
                self.brain.state_neurons.set_state("SEARCH")
                prev_state_name = "SEARCH"

        if train_rl:
            reward = 0.0
            if found_info and found_info not in query: reward = 1.0
            else: reward = -0.5
            self.brain.reinforce(trajectory, reward)
        return found_info