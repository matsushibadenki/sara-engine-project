_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/data_loader.py",
    "//": "ファイルの日本語タイトル: 連合スパイク・エンコーダー",
    "//": "ファイルの目的や内容: 複数のトークンを時間的に重ね合わせ、概念間の連合学習（Associative Learning）を可能にする。"
}

import random
from typing import List, Iterator

class SpikeStreamDataLoader:
    def __init__(self, data: List[List[float]], time_steps: int = 10, max_rate: float = 0.5):
        self.data = data
        self.time_steps = time_steps
        self.max_rate = max_rate

    def __iter__(self) -> Iterator[List[List[int]]]:
        for vector in self.data:
            stream = []
            for _ in range(self.time_steps):
                spikes = [idx for idx, val in enumerate(vector) if random.random() < (val * self.max_rate)]
                stream.append(spikes)
            yield stream

class TextToSpikeEncoder:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def encode_sequence(self, token_ids: List[int], time_per_token: int = 5) -> List[List[int]]:
        full_stream = []
        for token_id in token_ids:
            for _ in range(time_per_token):
                spikes = [token_id % self.vocab_size]
                if random.random() > 0.8:
                    spikes.append(random.randint(0, self.vocab_size - 1))
                full_stream.append(spikes)
        return full_stream

class SemanticSpikeEncoder:
    def __init__(self, vocab_size: int, embed_dim: int, ensemble_size: int = 8):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.ensemble_size = ensemble_size
        state = random.getstate()
        random.seed(42)
        self.token_maps = {
            i: random.sample(range(embed_dim), ensemble_size) 
            for i in range(vocab_size)
        }
        random.setstate(state)

    def encode_token_stream(self, token_id: int, duration: int = 5) -> List[List[int]]:
        ensemble = self.token_maps.get(token_id % self.vocab_size, [])
        stream = []
        for _ in range(duration):
            spikes = [n for n in ensemble if random.random() > 0.2]
            if random.random() > 0.9:
                spikes.append(random.randint(0, self.embed_dim - 1))
            stream.append(list(set(spikes)))
        return stream

    def encode_associative_stream(self, token_ids: List[int], duration: int = 10) -> List[List[int]]:
        """複数のトークン（概念）を同時に提示するストリームを生成"""
        combined_ensembles = []
        for tid in token_ids:
            combined_ensembles.extend(self.token_maps.get(tid % self.vocab_size, []))
        
        stream = []
        for _ in range(duration):
            # 提示される概念の集合から確率的に発火
            spikes = [n for n in combined_ensembles if random.random() > 0.4]
            stream.append(list(set(spikes)))
        return stream