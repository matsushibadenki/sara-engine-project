_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/spiking_image_classifier.py",
    "//": "ファイルの日本語タイトル: スパイキング・画像分類器",
    "//": "ファイルの目的や内容: 画像のピクセル強度をスパイクに変換し、行列演算(CNN等)なしで視覚パターンをSTDP分類する。"
}

import json
import os
import random
from typing import List, Dict, Optional

class SNNImageClassifierConfig:
    def __init__(self, input_size: int = 64, num_classes: int = 2):
        self.input_size = input_size  # 例: 8x8画像 = 64ピクセル
        self.num_classes = num_classes

    def to_dict(self):
        return {
            "input_size": self.input_size,
            "num_classes": self.num_classes
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

class SpikingImageClassifier:
    def __init__(self, config: SNNImageClassifierConfig):
        self.config = config
        # クラスごとのシナプス重み辞書（行列の代替）
        self.class_synapses: List[Dict[int, float]] = [{} for _ in range(config.num_classes)]

        # 初期の微小な生物学的揺らぎ（ランダムシナプス）を形成
        random.seed(42)
        for c in range(config.num_classes):
            for i in range(config.input_size):
                self.class_synapses[c][i] = random.uniform(0.1, 0.5)
        random.seed()

    def forward(self, pixel_intensities: List[float], learning: bool = False, target_class: Optional[int] = None) -> int:
        # 1. Rate Coding (発火コーディング)
        # ピクセル強度(0.0~1.0)が閾値を超えた場所を「スパイクが発火したニューロン」とみなす
        active_pixels = [i for i, intensity in enumerate(pixel_intensities) if intensity > 0.5]

        # 2. Integrate (空間的発火の統合)
        class_potentials = [0.0] * self.config.num_classes
        for c_id in range(self.config.num_classes):
            for p_idx in active_pixels:
                class_potentials[c_id] += self.class_synapses[c_id].get(p_idx, 0.0)

        # 3. Fire (Winner-Takes-Allによる発火クラスの決定)
        predicted_class = 0
        if max(class_potentials) > 0:
            predicted_class = class_potentials.index(max(class_potentials))
        else:
            predicted_class = random.randint(0, self.config.num_classes - 1)

        # 4. 報酬変調STDPによる局所学習 (行列の逆伝播を代替)
        if learning and target_class is not None:
            if predicted_class != target_class:
                for p_idx in active_pixels:
                    # LTP: 正解クラスのシナプスを強化
                    current_w = self.class_synapses[target_class].get(p_idx, 0.0)
                    self.class_synapses[target_class][p_idx] = min(5.0, current_w + 0.3)

                    # LTD: 誤って発火したクラスのシナプスを減衰
                    wrong_w = self.class_synapses[predicted_class].get(p_idx, 0.0)
                    if wrong_w > 0:
                        self.class_synapses[predicted_class][p_idx] = max(0.0, wrong_w - 0.3)

        return predicted_class

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=4)
            
        with open(os.path.join(save_directory, "vision_synapses.json"), "w", encoding="utf-8") as f:
            json.dump(self.class_synapses, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: str):
        with open(os.path.join(save_directory, "config.json"), "r", encoding="utf-8") as f:
            config = SNNImageClassifierConfig.from_dict(json.load(f))
            
        model = cls(config)
        synapses_path = os.path.join(save_directory, "vision_synapses.json")
        if os.path.exists(synapses_path):
            with open(synapses_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                model.class_synapses = [{int(k): float(v) for k, v in d.items()} for d in loaded]
        return model