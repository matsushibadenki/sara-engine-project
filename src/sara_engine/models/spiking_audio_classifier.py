_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/spiking_audio_classifier.py",
    "//": "ファイルの日本語タイトル: スパイキング・音声分類器",
    "//": "ファイルの目的や内容: フーリエ変換(FFT)や行列演算を一切使わず、生物の蝸牛(耳)を模倣。波形のゼロ交差発火とスパイク間隔(ISI)だけで音声の周波数を学習する。"
}

import json
import os
import random
from typing import List, Dict, Optional

class SNNAudioClassifierConfig:
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes

    def to_dict(self):
        return {"num_classes": self.num_classes}

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

class SpikingAudioClassifier:
    def __init__(self, config: SNNAudioClassifierConfig):
        self.config = config
        # 周波数(スパイクの間隔)ごとの反応を記憶するシナプス辞書
        self.class_synapses: List[Dict[int, float]] = [{} for _ in range(config.num_classes)]

    def forward(self, waveform: List[float], learning: bool = False, target_class: Optional[int] = None) -> int:
        # 1. Biological Cochlea Encoding (Hair Cell Zero-Crossing Spikes)
        spikes = []
        for i in range(1, len(waveform)):
            # 波形がマイナスからプラスに転じる瞬間を「スパイク発火」とする
            if waveform[i-1] <= 0 and waveform[i] > 0:
                spikes.append(i)

        # 2. Extract Inter-Spike Intervals (ISI) - Time-coding of frequencies
        isi_counts = {}
        for i in range(1, len(spikes)):
            interval = spikes[i] - spikes[i-1]
            isi_counts[interval] = isi_counts.get(interval, 0) + 1

        # 3. Integrate Potentials based on ISI resonances
        class_potentials = [0.0] * self.config.num_classes
        for interval, count in isi_counts.items():
            for c_id in range(self.config.num_classes):
                # その間隔(周波数帯)のシナプス強度 × 発生回数
                class_potentials[c_id] += self.class_synapses[c_id].get(interval, 0.0) * count

        # 4. Fire (Winner-Takes-All)
        predicted_class = 0
        if max(class_potentials) > 0:
            predicted_class = class_potentials.index(max(class_potentials))
        else:
            predicted_class = random.randint(0, self.config.num_classes - 1)

        # 5. Reward-modulated STDP Learning
        if learning and target_class is not None:
            if predicted_class != target_class:
                for interval in isi_counts.keys():
                    # LTP: 正解クラスのシナプスを強化
                    current_w = self.class_synapses[target_class].get(interval, 0.0)
                    self.class_synapses[target_class][interval] = min(10.0, current_w + 1.0)
                    
                    # LTD: 誤予測クラスのシナプスを減衰
                    wrong_w = self.class_synapses[predicted_class].get(interval, 0.0)
                    if wrong_w > 0:
                        self.class_synapses[predicted_class][interval] = max(0.0, wrong_w - 0.5)

        return predicted_class

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=4)
        with open(os.path.join(save_directory, "audio_synapses.json"), "w", encoding="utf-8") as f:
            json.dump(self.class_synapses, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: str):
        with open(os.path.join(save_directory, "config.json"), "r", encoding="utf-8") as f:
            config = SNNAudioClassifierConfig.from_dict(json.load(f))
        model = cls(config)
        synapses_path = os.path.join(save_directory, "audio_synapses.json")
        if os.path.exists(synapses_path):
            with open(synapses_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                model.class_synapses = [{int(k): float(v) for k, v in d.items()} for d in loaded]
        return model