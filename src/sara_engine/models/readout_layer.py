_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/readout_layer.py",
    "//": "タイトル: スパイク読み出し層 (強LTD付き)",
    "//": "目的: 頻出トークンによる予測のハイジャックを防ぐため、誤予測時のペナルティを大幅に強化する。"
}

import random
from typing import List, Dict, Optional

class SpikeReadoutLayer:
    def __init__(self, d_model: int, vocab_size: int, learning_rate: float = 0.05):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        
        self.W: List[Dict[int, float]] = [{} for _ in range(d_model)]
        
        self.refractory_states: Dict[int, int] = {}
        self.refractory_duration = 2

    def forward(self, spikes: List[int], target_token: Optional[int] = None, learning: bool = True) -> int:
        for tid in list(self.refractory_states.keys()):
            self.refractory_states[tid] -= 1
            if self.refractory_states[tid] <= 0:
                del self.refractory_states[tid]

        potentials: Dict[int, float] = {}
        for s in spikes:
            for token_id, weight in self.W[s].items():
                if token_id not in potentials:
                    potentials[token_id] = 0.0
                potentials[token_id] += weight
                
        if not learning:
            for tid in self.refractory_states:
                if tid in potentials:
                    potentials[tid] = -9999.0 
                
        predicted_token = 0
        max_potential = -1.0
        
        if potentials:
            for token_id, p in potentials.items():
                if p > max_potential:
                    max_potential = p
                    predicted_token = token_id
        else:
            predicted_token = random.randint(0, self.vocab_size - 1)
            
        if not learning:
            self.refractory_states[predicted_token] = self.refractory_duration

        if learning and target_token is not None:
            for s in spikes:
                if target_token not in self.W[s]:
                    self.W[s][target_token] = 0.01 
                
                self.W[s][target_token] += self.learning_rate
                
                if self.W[s][target_token] > 1.0:
                    self.W[s][target_token] = 1.0
                    
                # 誤予測したトークンに対して強烈なLTD(減衰)ペナルティを与える
                if predicted_token != target_token and predicted_token in self.W[s]:
                    self.W[s][predicted_token] -= self.learning_rate * 0.5  # 0.05から0.5に強化
                    if self.W[s][predicted_token] <= 0.0:
                        del self.W[s][predicted_token]
                        
        return predicted_token