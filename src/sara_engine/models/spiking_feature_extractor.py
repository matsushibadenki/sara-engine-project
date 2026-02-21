_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/spiking_feature_extractor.py",
    "//": "ファイルの日本語タイトル: スパイキング特徴抽出モデル",
    "//": "ファイルの目的や内容: 短期シナプス抑制(STD)に加え、長期的な恒常性シナプス可塑性(Homeostatic Plasticity)を導入し、生物学的なTF-IDF機能を実現する。"
}

import json
import os
import random
import math
from typing import List, Dict

class SNNFeatureExtractorConfig:
    def __init__(self, embedding_dim: int = 1024, leak_rate: float = 0.98, std_decay: float = 0.2, std_recovery: float = 0.05):
        self.embedding_dim = embedding_dim
        self.leak_rate = leak_rate
        self.std_decay = std_decay
        self.std_recovery = std_recovery

    def to_dict(self):
        return {
            "embedding_dim": self.embedding_dim,
            "leak_rate": self.leak_rate,
            "std_decay": self.std_decay,
            "std_recovery": self.std_recovery
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class SpikingFeatureExtractor:
    """
    Bio-inspired Feature Extractor based on Liquid State Machine.
    Uses Homeostatic Plasticity (long-term downscaling) and STD (short-term fatigue)
    to naturally filter out high-frequency noise and extract semantic embeddings.
    """
    def __init__(self, config: SNNFeatureExtractorConfig):
        self.config = config
        self.sdr_map = {}
        # 恒常性可塑性による各トークンのベース発火エネルギー（初期値は1.0）
        self.token_excitability: Dict[int, float] = {}

    def _get_sdr(self, tok: int) -> List[int]:
        if tok not in self.sdr_map:
            random.seed(tok * 1337)
            sparsity = max(1, self.config.embedding_dim // 30)  # よりスパースに(約3.3%)
            self.sdr_map[tok] = random.sample(range(self.config.embedding_dim), sparsity)
            random.seed()
        return self.sdr_map[tok]

    def habituate(self, token_lists: List[List[int]]):
        """
        Homeostatic Plasticity (恒常性シナプス可塑性) のシミュレーション。
        コーパス全体を経験させ、頻出する入力（助詞・句読点など）の受容体をダウン・スケーリングする。
        """
        token_counts = {}
        for seq in token_lists:
            for tok in seq:
                token_counts[tok] = token_counts.get(tok, 0) + 1
                
        for tok, count in token_counts.items():
            # 頻出するほどベースのエネルギーが対数的に低下する（生物学的TF-IDF）
            self.token_excitability[tok] = 1.0 / math.log(2.0 + count)

    def forward(self, token_ids: List[int]) -> List[float]:
        potentials = [0.0] * self.config.embedding_dim
        token_resources = {}

        for tok in token_ids:
            for i in range(self.config.embedding_dim):
                potentials[i] *= self.config.leak_rate

            # 恒常性可塑性によるベース・エネルギーを取得
            base_exc = self.token_excitability.get(tok, 1.0)
            resource = token_resources.get(tok, base_exc)

            active_neurons = self._get_sdr(tok)
            for idx in active_neurons:
                potentials[idx] += resource

            # 発火による短期的なシナプス疲労 (Short-Term Depression)
            token_resources[tok] = resource * self.config.std_decay

            # 全シナプスの時間経過による回復 (ベース・エネルギーまでしか回復しない)
            for k in list(token_resources.keys()):
                target_exc = self.token_excitability.get(k, 1.0)
                token_resources[k] += (target_exc - token_resources[k]) * self.config.std_recovery

        norm = math.sqrt(sum(p * p for p in potentials))
        if norm > 0:
            potentials = [p / norm for p in potentials]

        return potentials

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=4)
            
        exc_path = os.path.join(save_directory, "excitability.json")
        with open(exc_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.token_excitability.items()}, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: str):
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = SNNFeatureExtractorConfig.from_dict(json.load(f))
            
        model = cls(config)
        exc_path = os.path.join(save_directory, "excitability.json")
        if os.path.exists(exc_path):
            with open(exc_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                model.token_excitability = {int(k): float(v) for k, v in data.items()}
        return model