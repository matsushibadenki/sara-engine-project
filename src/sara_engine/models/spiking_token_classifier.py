_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/spiking_token_classifier.py",
    "//": "ファイルの日本語タイトル: スパイキング・トークン分類器",
    "//": "ファイルの目的や内容: nn.SNNModuleを継承し、内部状態をstate_dictで管理するようにリファクタリング。"
}

import json
import os
import pickle
from typing import List, Dict, Optional, Any

from sara_engine import nn

class SNNTokenClassifierConfig:
    def __init__(self, vocab_size: int = 256, num_classes: int = 4, context_length: int = 32):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.context_length = context_length
        self.reservoir_size = vocab_size * context_length

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "num_classes": self.num_classes,
            "context_length": self.context_length,
            "reservoir_size": self.reservoir_size
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SNNTokenClassifierConfig':
        return cls(**data)


class SpikingTokenClassifier(nn.SNNModule):
    def __init__(self, config: SNNTokenClassifierConfig):
        super().__init__()
        self.config = config
        self.class_synapses: List[Dict[int, float]] = [{} for _ in range(config.num_classes)]
        self.register_state("class_synapses")
        self.class_potentials = [0.0] * config.num_classes

    def reset_state(self) -> None:
        super().reset_state()
        self.class_potentials = [0.0] * self.config.num_classes

    # mypy対応: target_classesにOptionalを追加
    def forward(self, token_ids: List[int], learning: bool = False, target_classes: Optional[List[int]] = None) -> List[int]:
        self.reset_state()
        predictions: List[int] = []
        delay_buffer: List[int] = []
        
        for step, tok in enumerate(token_ids):
            delay_buffer.insert(0, tok)
            if len(delay_buffer) > self.config.context_length:
                delay_buffer.pop()

            self.class_potentials = [0.0] * self.config.num_classes

            for delay, d_tok in enumerate(delay_buffer):
                r_idx = delay * self.config.vocab_size + d_tok
                for c_id in range(1, self.config.num_classes):
                    self.class_potentials[c_id] += self.class_synapses[c_id].get(r_idx, 0.0)

            pred_class = 0
            max_pot = -999.0
            best_class = 0
            
            if self.config.num_classes > 1:
                pots = self.class_potentials[1:]
                max_pot = max(pots)
                best_class = pots.index(max_pot) + 1

            threshold = 24.0
            if max_pot > threshold:
                pred_class = best_class

            predictions.append(pred_class)

            if learning and target_classes is not None and step < len(target_classes):
                target_c = target_classes[step]
                
                for delay, d_tok in enumerate(delay_buffer):
                    r_idx = delay * self.config.vocab_size + d_tok
                    decay = 0.5 ** delay
                    
                    if target_c != 0:
                        curr_w = self.class_synapses[target_c].get(r_idx, 0.0)
                        max_w = 30.0 * decay
                        self.class_synapses[target_c][r_idx] = min(max_w, curr_w + 6.0 * decay)
                    
                    if pred_class != target_c and pred_class != 0:
                        wrong_w = self.class_synapses[pred_class].get(r_idx, 0.0)
                        min_w = -25.0 * decay
                        self.class_synapses[pred_class][r_idx] = max(min_w, wrong_w - 8.0 * decay)

        return predictions

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=4)
            
        state_path = os.path.join(save_directory, "model_state.pkl")
        with open(state_path, "wb") as f:
            pickle.dump(self.state_dict(), f)

    @classmethod
    def from_pretrained(cls, save_directory: str) -> 'SpikingTokenClassifier':
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = SNNTokenClassifierConfig.from_dict(json.load(f))
            
        model = cls(config)
        state_path = os.path.join(save_directory, "model_state.pkl")
        if os.path.exists(state_path):
            with open(state_path, "rb") as f:
                state = pickle.load(f)
            model.load_state_dict(state)
        else:
            old_weights_path = os.path.join(save_directory, "token_synapses.json")
            if os.path.exists(old_weights_path):
                with open(old_weights_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    model.class_synapses = [{int(k): float(v) for k, v in c.items()} for c in loaded]
                    
        return model