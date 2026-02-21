_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/spiking_image_classifier.py",
    "//": "ファイルの日本語タイトル: スパイキング画像分類モデル",
    "//": "ファイルの目的や内容: 画像のピクセル強度をスパイクに変換(Rate Coding)し、LIFニューロンと報酬変調型STDPを用いてパターンを学習・分類する。"
}

import json
import os
import random
from typing import List, Dict, Optional

class SNNImageClassifierConfig:
    def __init__(self, input_dim: int, num_classes: int, time_steps: int = 10, leak_rate: float = 0.9):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.time_steps = time_steps
        self.leak_rate = leak_rate

    def to_dict(self):
        return {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "time_steps": self.time_steps,
            "leak_rate": self.leak_rate
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class SpikingImageClassifier:
    """
    Bio-inspired Image Classification Model.
    Uses Rate Coding (brighter pixels spike more frequently) and Leaky Integrate-and-Fire (LIF) neurons.
    Learns via Error-Driven Reward-Modulated STDP. No convolutions or matrix multiplications.
    """
    def __init__(self, config: SNNImageClassifierConfig):
        self.config = config
        # synapses[class_id][pixel_idx] = weight
        self.class_synapses: List[Dict[int, float]] = [{} for _ in range(config.num_classes)]

    def forward(self, pixels: List[float], learning: bool = False, target_class: Optional[int] = None) -> int:
        """
        pixels: A flat list of normalized pixel intensities (0.0 to 1.0).
        """
        class_potentials = [0.0] * self.config.num_classes
        spiked_pixels = set()

        # Simulate over biological time steps
        for _ in range(self.config.time_steps):
            # 1. Leak (Membrane potential decay)
            for c_id in range(self.config.num_classes):
                class_potentials[c_id] *= self.config.leak_rate

            # 2. Retinal Rate Coding & Integrate
            for p_idx, intensity in enumerate(pixels):
                # Brighter pixels have a higher probability of emitting a spike
                if random.random() < intensity:
                    spiked_pixels.add(p_idx)
                    for c_id in range(self.config.num_classes):
                        weight = self.class_synapses[c_id].get(p_idx, 0.1) # small initial weight
                        class_potentials[c_id] += weight

        # 3. Fire (Winner-Takes-All)
        predicted_class = 0
        if max(class_potentials) > 0.0 or learning:
            predicted_class = class_potentials.index(max(class_potentials)) if max(class_potentials) > 0 else random.randint(0, self.config.num_classes - 1)

        # 4. Error-Driven STDP Learning
        if learning and target_class is not None:
            if predicted_class != target_class:
                for p_idx in spiked_pixels:
                    # LTP: Strengthen active synapses to the correct class
                    current_w = self.class_synapses[target_class].get(p_idx, 0.1)
                    self.class_synapses[target_class][p_idx] = min(5.0, current_w + 0.5)
                    
                    # LTD: Weaken active synapses to the wrongly predicted class
                    wrong_w = self.class_synapses[predicted_class].get(p_idx, 0.1)
                    if wrong_w > 0:
                        self.class_synapses[predicted_class][p_idx] = max(0.0, wrong_w - 0.5)

        return predicted_class

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=4)
            
        weights_path = os.path.join(save_directory, "image_classifier_synapses.json")
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
            config = SNNImageClassifierConfig.from_dict(json.load(f))
            
        model = cls(config)
        weights_path = os.path.join(save_directory, "image_classifier_synapses.json")
        if os.path.exists(weights_path):
            with open(weights_path, "r", encoding="utf-8") as f:
                loaded_synapses = json.load(f)
                model.class_synapses = [
                    {int(k): float(v) for k, v in class_dict.items()} 
                    for class_dict in loaded_synapses
                ]
        return model