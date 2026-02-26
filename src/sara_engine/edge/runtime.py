_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/edge/runtime.py",
    "//": "ファイルの日本語タイトル: Sara-Edge 軽量ランタイム",
    "//": "ファイルの目的や内容: エッジデバイス向けに最適化された推論専用エンジン。学習用のクラス階層やオーバーヘッドをすべて排除し、最小限のメモリでテキスト生成を実行する。"
}

import json
import random
import operator
from typing import List, Dict

class SaraEdgeRuntime:
    """
    Ultra-lightweight inference runtime for Edge devices (Raspberry Pi, Microcontrollers).
    Runs without the heavy nn.Module architecture and backpropagation structures.
    """
    def __init__(self, model_path: str):
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.context_length = data.get("context_length", 64)
        self.embed_dim = data.get("embed_dim", 64)
        self.total_readout_size = data.get("total_readout_size", 8192 + 64)
        
        # Keys in JSON are converted to strings, cast them back to ints
        self.readout_synapses = []
        for syn_dict in data.get("readout_synapses", []):
            converted = {int(k): float(v) for k, v in syn_dict.items()}
            self.readout_synapses.append(converted)
            
        self.reservoir_size = self.total_readout_size - self.embed_dim
        self.delay_buffer: List[int] = []

    def reset_state(self):
        self.delay_buffer.clear()

    def _get_sdr(self, delay: int, tok: int) -> List[int]:
        seed_val = (delay * 73856093) ^ (tok * 19349663) ^ 42
        random.seed(seed_val)
        spikes = random.sample(range(self.reservoir_size), 20)
        random.seed()
        return spikes

    def forward_step(self, token_id: int, refractory_tokens: List[int] = None) -> int:
        self.delay_buffer.insert(0, token_id)
        if len(self.delay_buffer) > self.context_length:
            self.delay_buffer.pop()

        res_spikes = set()
        for delay, tok in enumerate(self.delay_buffer):
            res_spikes.update(self._get_sdr(delay, tok))
        
        out_potentials: Dict[int, float] = {}
        for s in res_spikes:
            if s < self.total_readout_size:
                for v_idx, w in self.readout_synapses[s].items():
                    out_potentials[v_idx] = out_potentials.get(v_idx, 0.0) + w

        # Apply biological refractory penalty to prevent repetition
        if refractory_tokens:
            decay_factor = 0.4
            for r_tok in reversed(refractory_tokens):
                if r_tok in out_potentials:
                    out_potentials[r_tok] *= decay_factor
                decay_factor += 0.15
                if decay_factor > 1.0:
                    decay_factor = 1.0

        if out_potentials:
            max_val = max(out_potentials.values())
            if max_val > 0.1:
                return max(out_potentials.items(), key=operator.itemgetter(1))[0]
                
        return 32 # Fallback to Space character

    def generate(self, text: str, max_length: int = 50) -> str:
        input_ids = [ord(c) for c in text]
        self.reset_state()

        first_pred = 32
        for token_id in input_ids:
            first_pred = self.forward_step(token_id)

        generated_chars = []
        current_token = first_pred
        refractory_buffer = []

        for _ in range(max_length):
            if current_token == 0:
                break

            try:
                char = chr(current_token) if current_token >= 32 else ""
            except ValueError:
                char = ""

            generated_chars.append(char)
            refractory_buffer.append(current_token)
            if len(refractory_buffer) > 6:
                refractory_buffer.pop(0)

            current_token = self.forward_step(current_token, refractory_tokens=refractory_buffer)

        return text + "".join(generated_chars)