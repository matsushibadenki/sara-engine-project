# src/sara_engine/models/spiking_sequence_classifier.py
_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/spiking_sequence_classifier.py",
    "//": "ファイルの日本語タイトル: スパイキング・シーケンス分類器",
    "//": "ファイルの目的や内容: LIFモデルの減衰率を0.99に調整し、文字単位トークンとなる日本語などの長い系列でも文頭の特徴を忘れないように改善。"
}

import json
import os
import random
from typing import List, Dict, Optional

class SNNSequenceClassifierConfig:
    def __init__(self, vocab_size: int = 256, num_classes: int = 2, reservoir_size: int = 1024):
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


class SpikingSequenceClassifier:
    def __init__(self, config: SNNSequenceClassifierConfig):
        self.config = config
        self.sdr_map = {}
        random.seed(42)
        for tok in range(config.vocab_size):
            self.sdr_map[tok] = random.sample(range(config.reservoir_size), 10)
        random.seed()
        
        self.class_synapses: List[Dict[int, float]] = [{} for _ in range(config.num_classes)]
        self.active_reservoir_neurons: set = set()

    def reset_state(self):
        self.active_reservoir_neurons.clear()

    def forward(self, token_ids: List[int], learning: bool = False, target_class: Optional[int] = None) -> int:
        self.reset_state()
        
        # 生物学的な漏れ積分発火(Leaky Integrate-and-Fire)の電位
        class_potentials = [0.0] * self.config.num_classes
        
        for tok in token_ids:
            # 1. Leak (減衰) - 0.95から0.99へ変更。文字単位の長いトークン列でも短期記憶を維持する。
            for c_id in range(self.config.num_classes):
                class_potentials[c_id] *= 0.99
                
            # 2. Integrate (統合)
            if tok in self.sdr_map:
                self.active_reservoir_neurons.update(self.sdr_map[tok])
                for r_idx in self.sdr_map[tok]:
                    for c_id in range(self.config.num_classes):
                        class_potentials[c_id] += self.class_synapses[c_id].get(r_idx, 0.0)
                        
        # 3. Fire (Winner-Takes-All)
        predicted_class = 0
        if max(class_potentials) > 0.0 or learning:
            predicted_class = class_potentials.index(max(class_potentials)) if max(class_potentials) > 0 else random.randint(0, self.config.num_classes - 1)

        # Error-Driven Reward-modulated STDP Learning
        if learning and target_class is not None:
            if predicted_class != target_class:
                for r_idx in self.active_reservoir_neurons:
                    # LTP: 正解クラスのシナプスを強化
                    current_w = self.class_synapses[target_class].get(r_idx, 0.0)
                    self.class_synapses[target_class][r_idx] = min(10.0, current_w + 1.0)
                    
                    # LTD: 誤予測クラスのシナプスを減衰
                    wrong_w = self.class_synapses[predicted_class].get(r_idx, 0.0)
                    if wrong_w > 0:
                        self.class_synapses[predicted_class][r_idx] = max(0.0, wrong_w - 0.5)
                        if self.class_synapses[predicted_class][r_idx] <= 0.0:
                            del self.class_synapses[predicted_class][r_idx]
                            
        return predicted_class

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=4)
            
        weights_path = os.path.join(save_directory, "classifier_synapses.json")
        with open(weights_path, "w", encoding="utf-8") as f:
            serializable_synapses = [
                {str(k): v for k, v in class_dict.items()} 
                for class_dict in self.class_synapses
            ]
            json.dump(serializable_synapses, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: str):
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = SNNSequenceClassifierConfig.from_dict(json.load(f))
            
        model = cls(config)
        weights_path = os.path.join(save_directory, "classifier_synapses.json")
        if os.path.exists(weights_path):
            with open(weights_path, "r", encoding="utf-8") as f:
                loaded_synapses = json.load(f)
                model.class_synapses = [
                    {int(k): float(v) for k, v in class_dict.items()} 
                    for class_dict in loaded_synapses
                ]
        return model