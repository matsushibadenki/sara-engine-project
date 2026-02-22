# src/sara_engine/models/spiking_sequence_classifier.py
_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/spiking_sequence_classifier.py",
    "//": "ファイルの日本語タイトル: スパイキング・シーケンス分類器",
    "//": "ファイルの目的や内容: UTF-8の日本語(1文字3バイト)を処理できるよう、時間遅延(Delay)を32に拡張。"
}

import json
import os
import random
from typing import List, Dict, Optional

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


class SpikingSequenceClassifier:
    def __init__(self, config: SNNSequenceClassifierConfig):
        self.config = config
        # 日本語（3バイト/文字）を考慮し、過去32バイト（約10文字分）の文脈を記憶
        self.context_length = 32
        self.sdr_map = {}
        
        random.seed(42)
        for delay in range(self.context_length):
            for tok in range(config.vocab_size):
                # 空間が広がったため、発火スパイク数を少し減らしてノイズを抑える
                self.sdr_map[(delay, tok)] = random.sample(range(config.reservoir_size), 2)
        random.seed()
        
        self.class_synapses: List[Dict[int, float]] = [{} for _ in range(config.num_classes)]
        self.active_reservoir_neurons: set = set()

    def reset_state(self):
        self.active_reservoir_neurons.clear()

    def forward(self, token_ids: List[int], learning: bool = False, target_class: Optional[int] = None) -> int:
        self.reset_state()
        
        class_potentials = [0.0] * self.config.num_classes
        delay_buffer: List[int] = []
        
        for tok in token_ids:
            # 短期記憶バッファの更新 (FIFO)
            delay_buffer.insert(0, tok)
            if len(delay_buffer) > self.context_length:
                delay_buffer.pop()

            # 1. Leak (減衰)
            for c_id in range(self.config.num_classes):
                class_potentials[c_id] *= 0.99
                
            # 2. Integrate (統合)
            for delay, d_tok in enumerate(delay_buffer):
                if (delay, d_tok) in self.sdr_map:
                    self.active_reservoir_neurons.update(self.sdr_map[(delay, d_tok)])
                    for r_idx in self.sdr_map[(delay, d_tok)]:
                        for c_id in range(self.config.num_classes):
                            class_potentials[c_id] += self.class_synapses[c_id].get(r_idx, 0.0)
                        
        # 3. Fire (Winner-Takes-All)
        predicted_class = 0
        if max(class_potentials) > 0.0 or learning:
            predicted_class = class_potentials.index(max(class_potentials)) if max(class_potentials) > 0 else random.randint(0, self.config.num_classes - 1)

        # 4. Error-Driven Reward-modulated STDP Learning
        if learning and target_class is not None:
            if predicted_class != target_class:
                for r_idx in self.active_reservoir_neurons:
                    # LTP: 正解クラスのシナプスを強化
                    current_w = self.class_synapses[target_class].get(r_idx, 0.0)
                    self.class_synapses[target_class][r_idx] = min(15.0, current_w + 1.0)
                    
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