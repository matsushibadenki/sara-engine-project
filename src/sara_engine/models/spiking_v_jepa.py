# Directory Path: src/sara_engine/models/spiking_v_jepa.py
# English Title: Spiking Video Joint Embedding Predictive Architecture
# Purpose/Content: Implementation of V-JEPA using Spiking Neural Networks. 
# It handles spatiotemporal prediction by masking segments of a spike stream and 
# predicting them using recurrent predictive coding and local STDP.

import random
from typing import Dict, List

class SpikingVJEPA:
    """
    Spiking Video Joint Embedding Predictive Architecture (V-JEPA).
    Focuses on temporal prediction of latent SDR streams using recurrence.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dim: int,
        learning_rate: float = 0.05,
        mask_ratio: float = 0.3
    ):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.mask_ratio = mask_ratio
        
        # Encoder: input -> embed
        self.encoder_synapses = self._init_synapses(input_dim, embed_dim)
        
        # Recurrent Predictor: embed + hidden -> hidden
        # Predicts next hidden state
        self.predictor_synapses = self._init_synapses(embed_dim + hidden_dim, hidden_dim)
        
        # Readout: hidden -> embed (Predicts masked embedding)
        self.readout_synapses = self._init_synapses(hidden_dim, embed_dim)
        
        self.potentials_enc = [0.0] * embed_dim
        self.potentials_pred = [0.0] * hidden_dim
        self.potentials_read = [0.0] * embed_dim
        
        self.current_hidden_spikes: List[int] = []
        self.threshold = 1.0
        self.w_max = 5.0

    def _init_synapses(self, in_dim: int, out_dim: int) -> List[Dict[int, float]]:
        synapses: List[Dict[int, float]] = []
        for _ in range(out_dim):
            connections: Dict[int, float] = {}
            for i in range(in_dim):
                if random.random() < 0.2:
                    connections[i] = random.uniform(0.1, 0.5)
            synapses.append(connections)
        return synapses

    def reset_state(self):
        """Resets membrane potentials and recurrent state."""
        self.potentials_enc = [0.0] * self.embed_dim
        self.potentials_pred = [0.0] * self.hidden_dim
        self.potentials_read = [0.0] * self.embed_dim
        self.current_hidden_spikes = []

    def encode(self, input_spikes: List[int]) -> List[int]:
        """Encodes current frame spikes into embedding spikes."""
        out_spikes = [0] * self.embed_dim
        for j in range(self.embed_dim):
            self.potentials_enc[j] *= 0.9
            for i in input_spikes:
                if i in self.encoder_synapses[j]:
                    self.potentials_enc[j] += self.encoder_synapses[j][i]
            
            if self.potentials_enc[j] >= self.threshold:
                out_spikes[j] = 1
                self.potentials_enc[j] = 0.0
        return [i for i, s in enumerate(out_spikes) if s == 1]

    def predict_next(self, current_embed_spikes: List[int]) -> List[int]:
        """Recurrent prediction of the next latent state."""
        # Combine current embedding and previous hidden state
        combined_input = current_embed_spikes + [i + self.embed_dim for i in self.current_hidden_spikes]
        
        # Forward to hidden
        next_hidden = [0] * self.hidden_dim
        for j in range(self.hidden_dim):
            self.potentials_pred[j] *= 0.9
            for i in combined_input:
                if i in self.predictor_synapses[j]:
                    self.potentials_pred[j] += self.predictor_synapses[j][i]
            
            if self.potentials_pred[j] >= self.threshold:
                next_hidden[j] = 1
                self.potentials_pred[j] = 0.0
        
        self.current_hidden_spikes = [i for i, s in enumerate(next_hidden) if s == 1]
        
        # Predict embedding from hidden
        predicted_embed = [0] * self.embed_dim
        for j in range(self.embed_dim):
            temp_pot = 0.0
            for i in self.current_hidden_spikes:
                if i in self.readout_synapses[j]:
                    temp_pot += self.readout_synapses[j][i]
            if temp_pot >= self.threshold:
                predicted_embed[j] = 1
        
        return [i for i, s in enumerate(predicted_embed) if s == 1]

    def update_stdp(self, pre_spikes: List[int], post_spikes: List[int], synapses: List[Dict[int, float]], signal: float):
        """Local STDP update."""
        pre_set = set(pre_spikes)
        for j in post_spikes:
            current_synapses = synapses[j]
            for i in list(current_synapses.keys()):
                if i in pre_set:
                    delta = self.learning_rate * signal * (self.w_max - current_synapses[i])
                    current_synapses[i] += delta
                else:
                    current_synapses[i] -= (self.learning_rate * 0.05 * current_synapses[i])
                
                if current_synapses[i] < 0.01:
                    del current_synapses[i]
                elif current_synapses[i] > self.w_max:
                    current_synapses[i] = self.w_max

    def step(self, video_stream: List[List[int]], learning: bool = True) -> float:
        """
        Process a sequence of spike patterns (frames).
        Implements temporal masking and predictive learning.
        """
        if not video_stream:
            return 0.0
            
        self.reset_state()
        total_surprise = 0.0
        
        for t in range(len(video_stream) - 1):
            current_frame = video_stream[t]
            next_frame = video_stream[t+1]
            
            # 1. Encode current
            current_embed = self.encode(current_frame)
            
            # 2. Predict next embedding
            predicted_embed = self.predict_next(current_embed)
            
            # 3. Encode actual next (target)
            actual_next_embed = self.encode(next_frame)
            
            # 4. Compare and Learn
            target_set = set(actual_next_embed)
            pred_set = set(predicted_embed)
            
            if target_set:
                intersect = len(target_set.intersection(pred_set))
                accuracy = intersect / len(target_set) if len(target_set) > 0 else 0.0
                surprise_signal = (accuracy * 2.0) - 1.0 # Reward if accurate
                
                if learning:
                    # Update Predictor (Embed + Recurrent -> Hidden)
                    combined_pre = current_embed + [i + self.embed_dim for i in self.current_hidden_spikes]
                    self.update_stdp(combined_pre, self.current_hidden_spikes, self.predictor_synapses, surprise_signal)
                    
                    # Update Readout (Hidden -> Predicted Embed)
                    self.update_stdp(self.current_hidden_spikes, actual_next_embed, self.readout_synapses, surprise_signal)
                    
                    # Update Encoder (Frame -> Embed)
                    self.update_stdp(current_frame, current_embed, self.encoder_synapses, surprise_signal)
                
                total_surprise += (1.0 - accuracy)
                
        return total_surprise / max(1, len(video_stream) - 1)
