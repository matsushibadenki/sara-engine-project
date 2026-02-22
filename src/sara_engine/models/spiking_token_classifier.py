_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/spiking_token_classifier.py",
    "//": "ファイルの日本語タイトル: スパイキング・トークン分類器",
    "//": "ファイルの目的や内容: STDPの超・急峻な時間窓(0.5^delay)を導入。遠い過去のプレフィックスへの過剰依存を断ち切り、直近のバイトのみで厳密な単語境界を自己組織化する。"
}

import json
import os
from typing import List, Dict

class SNNTokenClassifierConfig:
    def __init__(self, vocab_size: int = 256, num_classes: int = 4, context_length: int = 32):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.context_length = context_length
        self.reservoir_size = vocab_size * context_length

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "num_classes": self.num_classes,
            "context_length": self.context_length,
            "reservoir_size": self.reservoir_size
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class SpikingTokenClassifier:
    def __init__(self, config: SNNTokenClassifierConfig):
        self.config = config
        self.class_synapses: List[Dict[int, float]] = [{} for _ in range(config.num_classes)]
        self.class_potentials = [0.0] * config.num_classes

    def reset_state(self):
        self.class_potentials = [0.0] * self.config.num_classes

    def forward(self, token_ids: List[int], learning: bool = False, target_classes: List[int] = None) -> List[int]:
        self.reset_state()
        predictions = []
        delay_buffer: List[int] = []
        
        for step, tok in enumerate(token_ids):
            delay_buffer.insert(0, tok)
            if len(delay_buffer) > self.config.context_length:
                delay_buffer.pop()

            self.class_potentials = [0.0] * self.config.num_classes

            # 2. Integrate (時間的コンテキストの空間的評価)
            for delay, d_tok in enumerate(delay_buffer):
                r_idx = delay * self.config.vocab_size + d_tok
                for c_id in range(1, self.config.num_classes):
                    self.class_potentials[c_id] += self.class_synapses[c_id].get(r_idx, 0.0)

            # 3. Fire (閾値判定)
            pred_class = 0
            max_pot = -999.0
            best_class = 0
            
            if self.config.num_classes > 1:
                pots = self.class_potentials[1:]
                max_pot = max(pots)
                best_class = pots.index(max_pot) + 1

            # STDPのシャープな減衰に合わせて閾値を24.0に設定
            threshold = 24.0
            if max_pot > threshold:
                pred_class = best_class

            predictions.append(pred_class)

            # 4. Reward-modulated STDP Learning
            if learning and target_classes is not None and step < len(target_classes):
                target_c = target_classes[step]
                
                for delay, d_tok in enumerate(delay_buffer):
                    r_idx = delay * self.config.vocab_size + d_tok
                    
                    # 【最重要】急峻な時間窓(Sharp Time Window)。数文字前からの余韻を完全にゼロにする
                    decay = 0.5 ** delay
                    
                    # LTP: 正解エンティティの強化
                    if target_c != 0:
                        curr_w = self.class_synapses[target_c].get(r_idx, 0.0)
                        max_w = 30.0 * decay
                        self.class_synapses[target_c][r_idx] = min(max_w, curr_w + 6.0 * decay)
                    
                    # LTD: 誤予測(エンティティの滲み、助詞の巻き込み)の強烈な抑制
                    if pred_class != target_c and pred_class != 0:
                        wrong_w = self.class_synapses[pred_class].get(r_idx, 0.0)
                        min_w = -25.0 * decay
                        self.class_synapses[pred_class][r_idx] = max(min_w, wrong_w - 8.0 * decay)

        return predictions

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=4)
        with open(os.path.join(save_directory, "token_synapses.json"), "w", encoding="utf-8") as f:
            serializable = [{str(k): v for k, v in c_dict.items()} for c_dict in self.class_synapses]
            json.dump(serializable, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: str):
        with open(os.path.join(save_directory, "config.json"), "r", encoding="utf-8") as f:
            config = SNNTokenClassifierConfig.from_dict(json.load(f))
        model = cls(config)
        weights_path = os.path.join(save_directory, "token_synapses.json")
        if os.path.exists(weights_path):
            with open(weights_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                model.class_synapses = [{int(k): float(v) for k, v in c_dict.items()} for c_dict in loaded]
        return model