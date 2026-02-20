# filepath: src/sara_engine/core/spike_attention.py
# title: スパイク自己注意機構
# description: スパイクのタイミング一致(coincidence)を利用し、行列演算を排したAttention機構。WTA(Winner-Take-All)でSoftmaxを代替する。

import random

class SpikingSelfAttention:
    def __init__(self, sdr_size, num_heads=4):
        self.sdr_size = sdr_size
        self.num_heads = num_heads
        
        # Sparse synaptic connections instead of dense matrices
        self.W_q = [{} for _ in range(sdr_size)]
        self.W_k = [{} for _ in range(sdr_size)]
        self.W_v = [{} for _ in range(sdr_size)]
        
        self._init_sparse_weights(self.W_q)
        self._init_sparse_weights(self.W_k)
        self._init_sparse_weights(self.W_v)
        
        self.spike_history_k = [] 
        self.spike_history_v = [] 

    def _init_sparse_weights(self, weights, density=0.05):
        # Initialize with sparse random connections
        for i in range(self.sdr_size):
            num_connections = int(self.sdr_size * density)
            targets = random.sample(range(self.sdr_size), num_connections)
            for t in targets:
                weights[i][t] = random.uniform(0.1, 0.5)

    def _propagate(self, active_inputs, weights):
        potentials = [0.0] * self.sdr_size
        for pre_id in active_inputs:
            for post_id, w in weights[pre_id].items():
                potentials[post_id] += w
        
        # Generate spikes (threshold = 1.0)
        spikes = [i for i, p in enumerate(potentials) if p > 1.0]
        return spikes

    def _spike_coincidence(self, spikes_a, set_b):
        # Spike timing coincidence (replaces dot product)
        overlap = 0
        for s in spikes_a:
            if s in set_b:
                overlap += 1
        return overlap

    def forward(self, current_spikes):
        # 1. Generate Query, Key, Value spikes
        q_spikes = self._propagate(current_spikes, self.W_q)
        k_spikes = self._propagate(current_spikes, self.W_k)
        v_spikes = self._propagate(current_spikes, self.W_v)
        
        self.spike_history_k.append(set(k_spikes))
        self.spike_history_v.append(v_spikes)
        
        # 2. Compute coincidence scores
        coincidence_scores = []
        for past_k in self.spike_history_k:
            score = self._spike_coincidence(q_spikes, past_k)
            coincidence_scores.append(score)
            
        if not coincidence_scores:
            return []
            
        # 3. Spiking Softmax / Winner-Take-All (WTA)
        max_score = max(coincidence_scores)
        winners = [i for i, score in enumerate(coincidence_scores) if score >= max_score - 1 and score > 0]
        
        # 4. Weighted sum of Values using membrane accumulation
        output_potentials = [0.0] * self.sdr_size
        for w_idx in winners:
            for v_spike in self.spike_history_v[w_idx]:
                output_potentials[v_spike] += 1.0
                
        # Generate final attention spikes
        attended_spikes = [i for i, p in enumerate(output_potentials) if p > len(winners) * 0.5]
        return attended_spikes