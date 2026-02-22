_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/spiking_feature_extractor.py",
    "//": "ファイルの日本語タイトル: スパイキング・特徴抽出器",
    "//": "ファイルの目的や内容: 発火電位に閾値(Threshold)を設け、微小なノイズ電位をゼロに落とすことで、真の疎分散表現(SDR)を生成しベクトル類似度を劇的に向上させる。"
}

import json
import os
import random
from typing import List

class SNNFeatureExtractorConfig:
    def __init__(self, vocab_size: int = 256, reservoir_size: int = 4096, context_length: int = 32):
        self.vocab_size = vocab_size
        self.reservoir_size = reservoir_size
        self.context_length = context_length

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "reservoir_size": self.reservoir_size,
            "context_length": self.context_length
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class SpikingFeatureExtractor:
    def __init__(self, config: SNNFeatureExtractorConfig):
        self.config = config
        self.sdr_map = {}
        
        # 空間マッピング: 文字(バイト)とその時間的な遅延位置に対して、ニューロンを割り当てる
        random.seed(42)
        for delay in range(config.context_length):
            for tok in range(config.vocab_size):
                self.sdr_map[(delay, tok)] = random.sample(range(config.reservoir_size), 4)
        random.seed()

    def forward(self, token_ids: List[int]) -> List[float]:
        """
        行列演算を使わず、Leak-Integrate-and-Fire(LIF)の概念を用いて
        スパイク電位の蓄積と閾値処理を行い、疎な特徴ベクトル(SDR)を生成する。
        """
        potentials = [0.0] * self.config.reservoir_size
        delay_buffer: List[int] = []

        for tok in token_ids:
            delay_buffer.insert(0, tok)
            if len(delay_buffer) > self.config.context_length:
                delay_buffer.pop()

            # Leak (時間減衰 - 記憶を長く保つため0.99)
            for i in range(self.config.reservoir_size):
                potentials[i] *= 0.99

            # Integrate (入力に基づく発火電位の統合)
            for delay, d_tok in enumerate(delay_buffer):
                if (delay, d_tok) in self.sdr_map:
                    for r_idx in self.sdr_map[(delay, d_tok)]:
                        potentials[r_idx] += 1.0

        # Fire (生物学的な発火閾値の適用)
        # ノイズ(微小な電位)をゼロに落とし、真の疎分散表現(SDR)にする
        max_p = max(potentials) if max(potentials) > 0 else 1.0
        threshold = max_p * 0.25  # 最大電位の25%未満は発火させない(ノイズカット)
        
        sparse_features = []
        for p in potentials:
            if p > threshold:
                sparse_features.append(p / max_p)
            else:
                sparse_features.append(0.0)

        return sparse_features

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: str):
        config_path = os.path.join(save_directory, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = SNNFeatureExtractorConfig.from_dict(json.load(f))
            return cls(config)
        return cls(SNNFeatureExtractorConfig())