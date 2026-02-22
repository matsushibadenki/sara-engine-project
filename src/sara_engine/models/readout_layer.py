_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/readout_layer.py",
    "//": "ファイルの日本語タイトル: スパイク読み出し層 (Pure PA-I SSTDP)",
    "//": "ファイルの目的や内容: バグの温床となっていたスパースEMAを廃止し、純粋なMulti-Class PA-Iアルゴリズムによるマージン最大化に特化。クラスごとのバイアスと上下限クリッピングにより、強力な線形分離平面を形成する。"
}

import random
from typing import List, Dict, Optional

class SpikeReadoutLayer:
    def __init__(self, d_model: int, vocab_size: int, learning_rate: float = 0.01):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        
        self.W: List[Dict[int, float]] = [{} for _ in range(d_model)]
        self.b: List[float] = [0.0] * vocab_size
        
        self.refractory_states: Dict[int, int] = {}
        self.refractory_duration = 2
        
        self.w_max = 10.0
        self.w_min = -10.0 

    def forward(self, spikes: List[int], target_token: Optional[int] = None, learning: bool = True) -> int:
        for tid in list(self.refractory_states.keys()):
            self.refractory_states[tid] -= 1
            if self.refractory_states[tid] <= 0:
                del self.refractory_states[tid]

        potentials = list(self.b)
        for s in spikes:
            if s < len(self.W):
                for token_id, weight in self.W[s].items():
                    potentials[token_id] += weight
                
        if not learning:
            for tid in self.refractory_states:
                if tid < self.vocab_size:
                    potentials[tid] = -9999.0 
                
        predicted_token = 0
        max_potential = -9999.0
        
        for i in range(self.vocab_size):
            if potentials[i] > max_potential:
                max_potential = potentials[i]
                predicted_token = i
        
        if max_potential <= -9000.0 and len(spikes) == 0:
            predicted_token = random.randint(0, self.vocab_size - 1)
            
        if not learning:
            self.refractory_states[predicted_token] = self.refractory_duration

        if learning and target_token is not None and len(spikes) > 0:
            norm_sq = float(len(spikes))
            updates = {}
            
            C = self.learning_rate * 10.0
            target_margin = 15.0
            
            for i in range(self.vocab_size):
                if i != target_token:
                    loss_i = target_margin - (potentials[target_token] - potentials[i])
                    if loss_i > 0:
                        tau_i = min(C, loss_i / (2.0 * norm_sq + 1.0))
                        updates[i] = tau_i
            
            if updates:
                tau_sum = sum(updates.values())
                
                self.b[target_token] += tau_sum
                for wrong_token, tau_i in updates.items():
                    self.b[wrong_token] -= tau_i
                
                for s in spikes:
                    if s < len(self.W):
                        current_w_target = self.W[s].get(target_token, 0.0)
                        new_w_target = current_w_target + tau_sum
                        if new_w_target > self.w_max:
                            new_w_target = self.w_max
                        self.W[s][target_token] = new_w_target
                        
                        to_delete = []
                        for wrong_token, tau_i in updates.items():
                            current_w_wrong = self.W[s].get(wrong_token, 0.0)
                            new_w_wrong = current_w_wrong - tau_i
                            
                            if new_w_wrong < self.w_min:
                                new_w_wrong = self.w_min
                                
                            if -0.01 < new_w_wrong < 0.01:
                                to_delete.append(wrong_token)
                            else:
                                self.W[s][wrong_token] = new_w_wrong
                                
                        for tid in to_delete:
                            if tid in self.W[s]:
                                del self.W[s][tid]

        return predicted_token