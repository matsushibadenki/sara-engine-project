_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/transformer.py",
    "//": "タイトル: 高精度整数演算型 Spike Transformer (Veto修正版)",
    "//": "目的: 桁溢れによる自己ループを防ぐため、完全な自己抑制(Veto)を導入し遷移を成功させる。"
}

import json
import numpy as np
from typing import List, Dict
from .attention import SpikeAttention

class SpikePositionalEncoding:
    def __init__(self, d_model: int, max_len: int = 1000, density: float = 0.1):
        self.d_model = d_model
        self.max_len = max_len
        self.pos_spikes = []
        rng = np.random.RandomState(42)
        num_active = max(1, int(d_model * density))
        for _ in range(max_len):
            spikes = rng.choice(d_model, num_active, replace=False).tolist()
            self.pos_spikes.append(spikes)

    def get_spikes(self, pos: int) -> List[int]:
        if pos < self.max_len:
            return self.pos_spikes[pos]
        return self.pos_spikes[-1]

class IntPlasticSpikeFFN:
    """
    10ビット精度の固定小数点ベース・リザバーFFN。
    SCALE = 1024 (ビットシフト >> 10) を用いる。
    """
    SCALE = 1024  
    SHIFT = 10
    
    def __init__(self, d_model: int, d_ff: int, learning_rate: float = 0.5):
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.lr_int = int(learning_rate * self.SCALE)
        self.ltd_int = int(learning_rate * 0.25 * self.SCALE)
        self.max_w = 5 * self.SCALE
        self.thresh_h = int(0.5 * self.SCALE)
        self.thresh_o = int(0.2 * self.SCALE)
        
        rng = np.random.RandomState(42)
        self.w_up: List[Dict[int, int]] = [{} for _ in range(d_model)]
        self.w_ctx: List[Dict[int, int]] = [{} for _ in range(d_model)]
        
        for i in range(d_model):
            for t in rng.choice(d_ff, max(1, int(d_ff * 0.15)), replace=False):
                self.w_up[i][t] = int(rng.uniform(0.5, 1.0) * self.SCALE)
            for t in rng.choice(d_ff, max(1, int(d_ff * 0.15)), replace=False):
                self.w_ctx[i][t] = int(rng.uniform(0.2, 0.6) * self.SCALE)
                
        self.w_down: List[Dict[int, int]] = [{} for _ in range(d_ff)]
        self.trace_pre = np.zeros(d_model, dtype=np.int32)
        self.fired_h_prev: List[int] = []

    def get_hidden_state(self, valid_in: List[int]) -> List[int]:
        v_h = np.zeros(self.d_ff, dtype=np.int32)
        
        for i in valid_in:
            for tgt, w in self.w_up[i].items():
                v_h[tgt] += w
                
        for i in range(self.d_model):
            if self.trace_pre[i] > 0:
                for tgt, w in self.w_ctx[i].items():
                    v_h[tgt] += (w * self.trace_pre[i]) >> self.SHIFT
                    
        fired_h = np.where(v_h >= self.thresh_h)[0].tolist()
        k_h = max(1, int(self.d_ff * 0.05))
        if len(fired_h) > k_h:
            fired_h = np.argsort(v_h)[-k_h:].tolist()
        return fired_h

    def compute(self, input_spikes: List[int], learning: bool = False) -> List[int]:
        valid_in = [i for i in input_spikes if i < self.d_model]
        fired_h = self.get_hidden_state(valid_in)
        
        if learning:
            if self.fired_h_prev:
                for h in self.fired_h_prev:
                    for tgt in valid_in:
                        self.w_down[h][tgt] = self.w_down[h].get(tgt, 0) + self.lr_int
                        if self.w_down[h][tgt] > self.max_w:
                            self.w_down[h][tgt] = self.max_w
                            
                    for tgt in list(self.w_down[h].keys()):
                        if tgt not in valid_in:
                            self.w_down[h][tgt] -= self.ltd_int
                            if self.w_down[h][tgt] < 0:
                                self.w_down[h][tgt] = 0

            self.fired_h_prev = fired_h
            
            self.trace_pre >>= 1
            for i in valid_in:
                self.trace_pre[i] += self.SCALE
                
            return valid_in

        else: 
            v_o = np.zeros(self.d_model, dtype=np.int32)
            for h in fired_h:
                for tgt, w in self.w_down[h].items():
                    v_o[tgt] += w
                    
            # --- 厳格な自己抑制 (Absolute Veto) ---
            # 減算ではなく、現在の入力スパイクを完全に0にして出力を阻止する
            for i in valid_in:
                v_o[i] = 0
                
            fired_o = np.where(v_o >= self.thresh_o)[0].tolist()
            k_o = max(1, int(self.d_model * 0.05))
            if len(fired_o) > k_o:
                fired_o = np.argsort(v_o)[-k_o:].tolist()
                
            self.fired_h_prev = fired_h
            self.trace_pre >>= 1
            for i in valid_in:
                self.trace_pre[i] += self.SCALE
                
            return fired_o

    def reset(self):
        self.trace_pre.fill(0)
        self.fired_h_prev = []

    def get_state(self) -> dict:
        return {
            "w_down": [{str(k): int(v) for k, v in layer.items()} for layer in self.w_down]
        }

    def set_state(self, state: dict):
        self.w_down = [{int(k): int(v) for k, v in layer.items()} for layer in state["w_down"]]

class PlasticTransformerBlock:
    def __init__(self, d_model: int, num_heads: int, memory_size: int = 50):
        self.d_model = d_model
        self.attention = SpikeAttention(d_model, d_model, memory_size=memory_size, num_heads=num_heads)
        self.ffn = IntPlasticSpikeFFN(d_model, d_model * 4)
        self.pos_encoder = SpikePositionalEncoding(d_model)

    def compute(self, input_spikes: List[int], pos: int, learning: bool = False) -> List[int]:
        pos_spikes = self.pos_encoder.get_spikes(pos)
        x = list(set(input_spikes) | set(pos_spikes))
        attn_out = self.attention.compute(x)
        x_ctx = list(set(x) | set(attn_out))
        return self.ffn.compute(x_ctx, learning=learning)

    def generate_next(self, current_spikes: List[int], pos: int) -> List[int]:
        eval_pos = pos - 1 if pos > 0 else 0
        pos_spikes = self.pos_encoder.get_spikes(eval_pos)
        valid_in = [s for s in current_spikes if s < self.d_model]
        x = list(set(valid_in) | set(pos_spikes))
        attn_out = self.attention.compute(x)
        x_ctx = list(set(x) | set(attn_out))
        return self.ffn.compute(x_ctx, learning=False)

    def reset(self):
        self.attention.reset()
        self.ffn.reset()

    def fit(self, sequence_spikes: List[List[int]], epochs: int = 200, verbose: bool = True):
        if verbose:
            print(f"Training started for {epochs} epochs (Fixed-Point 10-bit Mode)...")
        for epoch in range(epochs):
            self.reset()
            for pos, spikes in enumerate(sequence_spikes):
                self.compute(spikes, pos=pos, learning=True)
        if verbose:
            print("Training finished.")

    def predict(self, initial_spikes: List[int], steps: int = 3) -> List[List[int]]:
        self.reset()
        current_spikes = initial_spikes
        generated = []
        for t in range(1, steps + 1):
            next_spikes = self.generate_next(current_spikes, pos=t)
            generated.append(next_spikes)
            current_spikes = next_spikes
        return generated

    def save(self, filepath: str):
        state = {
            "d_model": self.d_model,
            "ffn_state": self.ffn.get_state()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4)

    def load(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        if state["d_model"] != self.d_model:
            raise ValueError(f"d_model mismatch: expected {self.d_model}, got {state['d_model']}")
        self.ffn.set_state(state["ffn_state"])