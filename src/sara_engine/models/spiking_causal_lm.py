# ファイルメタ情報
_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/spiking_causal_lm.py",
    "//": "タイトル: スパイキング因果言語モデル (Spiking Causal LM)",
    "//": "目的: SpikeTransformerModelへの変更を適応させ、シーケンス入力に対するループ処理を追加する。"
}

import random
from typing import List, Dict
from sara_engine.core.transformer import SpikeTransformerModel

class SpikeReadoutLayer:
    def __init__(self, d_model: int, vocab_size: int):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.weights: Dict[int, Dict[int, float]] = {i: {} for i in range(vocab_size)}
        
    def forward(self, spikes: List[int]) -> List[float]:
        scores = [0.0] * self.vocab_size
        for token_id in range(self.vocab_size):
            token_w = self.weights[token_id]
            score = 0.0
            for s in spikes:
                if s in token_w:
                    score += token_w[s]
            scores[token_id] = score
        return scores

    def learn(self, spikes: List[int], target_token: int):
        if not spikes:
            return

        scores = self.forward(spikes)

        target_w = self.weights.setdefault(target_token, {})
        for s in spikes:
            target_w[s] = min(1.0, target_w.get(s, 0.0) + 0.2)
                
        for token_id in range(self.vocab_size):
            if token_id != target_token and scores[token_id] > 0.0:
                other_w = self.weights[token_id]
                for s in spikes:
                    if s in other_w:
                        other_w[s] = max(0.0, other_w[s] - 0.05)
                        if other_w[s] == 0.0:
                            del other_w[s]


class SpikingCausalLM:
    def __init__(self, vocab_size: int, d_model: int = 1024, num_layers: int = 2, num_heads: int = 4):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.transformer = SpikeTransformerModel(num_layers=num_layers, embed_dim=d_model, hidden_dim=d_model*4)
        self.readout = SpikeReadoutLayer(d_model, vocab_size)

    def _compute_sequence(self, input_ids: List[int], learning: bool = True) -> List[List[int]]:
        """Transforms a sequence of tokens into a sequence of hidden spike states."""
        hidden_states = []
        for token_id in input_ids:
            # トークンIDを単一のスパイク入力として扱う
            spikes = self.transformer.forward([token_id % self.d_model], learning=learning)
            hidden_states.append(spikes)
        return hidden_states

    def train_step(self, input_ids: List[int], update_backbone: bool = True):
        if len(input_ids) < 2:
            return
            
        hidden_states = self._compute_sequence(input_ids[:-1], learning=update_backbone)
        
        for i, spikes in enumerate(hidden_states):
            target_token = input_ids[i + 1]
            self.readout.learn(spikes, target_token)

    def generate(self, input_ids: List[int], max_new_tokens: int = 20, top_k: int = 3) -> List[int]:
        generated = list(input_ids)
        
        for _ in range(max_new_tokens):
            hidden_states = self._compute_sequence(generated, learning=False)
            last_state = hidden_states[-1]
            
            scores = self.readout.forward(last_state)
            
            recent_tokens = set(generated[-4:])
            for t in recent_tokens:
                if t < len(scores):
                    scores[t] *= 0.1
            
            if len(scores) > 2:
                scores[0] = -1.0
                scores[1] = -1.0
                scores[2] = -1.0
            
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            top_scores = [scores[i] for i in top_indices if scores[i] > 0]
            valid_indices = [top_indices[i] for i in range(len(top_scores))]
            
            sum_scores = sum(top_scores)
            
            if sum_scores <= 0.0 or not valid_indices:
                valid_tokens = [i for i in range(4, self.vocab_size)]
                if len(generated) > len(input_ids) + 3:
                    valid_tokens.append(3)
                next_token = random.choice(valid_tokens) if valid_tokens else 3
            else:
                probs = [s / sum_scores for s in top_scores]
                r = random.random()
                cumulative = 0.0
                next_token = valid_indices[0]
                for idx, prob in zip(valid_indices, probs):
                    cumulative += prob
                    if r <= cumulative:
                        next_token = idx
                        break
                        
            generated.append(next_token)
            
            if next_token == 3:
                break
                
        return generated