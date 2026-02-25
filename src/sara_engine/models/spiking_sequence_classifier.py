_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/spiking_sequence_classifier.py",
    "//": "ファイルの日本語タイトル: スパイキング・シーケンス分類器",
    "//": "ファイルの目的や内容: nn.SNNModuleを継承し、内部状態をstate_dictで管理するようにリファクタリング。"
}

import json
import os
import random
import pickle
from typing import List, Dict, Optional

from sara_engine import nn

class SNNSequenceClassifierConfig:
    def __init__(self, vocab_size: int = 256, num_classes: int = 2, reservoir_size: int = 2048):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.reservoir_size = reservoir_size

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "num_classes": self.num_classes,
            "reservoir_size": self.reservoir_size
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class SpikingSequenceClassifier(nn.SNNModule):
    def __init__(self, config: SNNSequenceClassifierConfig):
        super().__init__()
        self.config = config
        self.context_length = 32
        self.sdr_map = {}
        
        random.seed(42)
        for delay in range(self.context_length):
            for tok in range(config.vocab_size):
                self.sdr_map[(delay, tok)] = random.sample(range(config.reservoir_size), 2)
        random.seed()
        
        # 学習可能なシナプス重みを状態として登録
        self.class_synapses: List[Dict[int, float]] = [{} for _ in range(config.num_classes)]
        self.register_state("class_synapses")
        
        self.active_reservoir_neurons = set()
        self.delay_buffer: List[int] = []

    def reset_state(self):
        super().reset_state()
        self.active_reservoir_neurons.clear()
        self.delay_buffer.clear()

    def forward(self, token_ids: List[int], learning: bool = False, target_class: Optional[int] = None) -> int:
        self.reset_state()
        
        class_potentials = [0.0] * self.config.num_classes
        self.delay_buffer = []
        
        for tok in token_ids:
            self.delay_buffer.insert(0, tok)
            if len(self.delay_buffer) > self.context_length:
                self.delay_buffer.pop()

            for c_id in range(self.config.num_classes):
                class_potentials[c_id] *= 0.99
                
            for delay, d_tok in enumerate(self.delay_buffer):
                if (delay, d_tok) in self.sdr_map:
                    self.active_reservoir_neurons.update(self.sdr_map[(delay, d_tok)])
                    for r_idx in self.sdr_map[(delay, d_tok)]:
                        for c_id in range(self.config.num_classes):
                            class_potentials[c_id] += self.class_synapses[c_id].get(r_idx, 0.0)
                        
        predicted_class = 0
        if max(class_potentials) > 0.0 or learning:
            predicted_class = class_potentials.index(max(class_potentials)) if max(class_potentials) > 0 else random.randint(0, self.config.num_classes - 1)

        if learning and target_class is not None:
            if predicted_class != target_class:
                for r_idx in self.active_reservoir_neurons:
                    # LTP
                    current_w = self.class_synapses[target_class].get(r_idx, 0.0)
                    self.class_synapses[target_class][r_idx] = min(15.0, current_w + 1.0)
                    
                    # LTD
                    wrong_w = self.class_synapses[predicted_class].get(r_idx, 0.0)
                    if wrong_w > 0:
                        self.class_synapses[predicted_class][r_idx] = max(0.0, wrong_w - 0.5)
                        if self.class_synapses[predicted_class][r_idx] <= 0.0:
                            del self.class_synapses[predicted_class][r_idx]
                            
        return predicted_class

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=4)
            
        state_path = os.path.join(save_directory, "model_state.pkl")
        with open(state_path, "wb") as f:
            pickle.dump(self.state_dict(), f)

    @classmethod
    def from_pretrained(cls, save_directory: str):
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = SNNSequenceClassifierConfig.from_dict(json.load(f))
            
        model = cls(config)
        state_path = os.path.join(save_directory, "model_state.pkl")
        if os.path.exists(state_path):
            with open(state_path, "rb") as f:
                state = pickle.load(f)
            model.load_state_dict(state)
        else:
            old_weights_path = os.path.join(save_directory, "classifier_synapses.json")
            if os.path.exists(old_weights_path):
                with open(old_weights_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    model.class_synapses = [{int(k): float(v) for k, v in c.items()} for c in loaded]
                    
        return model