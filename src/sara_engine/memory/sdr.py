import numpy as np
import hashlib
from collections import OrderedDict
from typing import List, Dict
import sys
import os

# Utilsからのインポート
try:
    from ..utils.tokenizer import SaraTokenizer
except ImportError:
    # フォールバック（単体テスト用など）
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from utils.tokenizer import SaraTokenizer

class SDREncoder:
    def __init__(self, input_size: int, density: float = 0.02, use_tokenizer: bool = True, cache_size: int = 10000):
        self.input_size = input_size
        self.density = density
        self.cache: 'OrderedDict[str, List[int]]' = OrderedDict()
        self.cache_size = cache_size
        self.use_tokenizer = use_tokenizer
        
        if self.use_tokenizer:
            # プロジェクトルート付近の語彙ファイルを探す
            vocab_path = "sara_vocab.json"
            self.tokenizer = SaraTokenizer(vocab_size=2000, model_path=vocab_path)
            self.token_sdr_map: Dict[int, List[int]] = {}

    def _get_token_sdr(self, token_id: int) -> List[int]:
        if token_id in self.token_sdr_map:
            return self.token_sdr_map[token_id]
        
        rng = np.random.RandomState(token_id)
        target_n = int(self.input_size * self.density)
        indices = rng.choice(self.input_size, target_n, replace=False)
        indices.sort()
        result = indices.tolist()
        
        self.token_sdr_map[token_id] = result
        return result

    def encode(self, text: str) -> List[int]:
        if text in self.cache:
            self.cache.move_to_end(text)
            return self.cache[text]
        
        result = []
        if not self.use_tokenizer:
            hash_obj = hashlib.sha256(text.encode('utf-8'))
            seed = int(hash_obj.hexdigest(), 16) % (2**32)
            rng = np.random.RandomState(seed)
            target_n = int(self.input_size * self.density)
            indices = rng.choice(self.input_size, target_n, replace=False)
            indices.sort()
            result = indices.tolist()
        else:
            token_ids = self.tokenizer.encode(text)
            if not token_ids:
                return []
            
            combined_indices = set()
            for tid in token_ids:
                combined_indices.update(self._get_token_sdr(tid))
            
            result = sorted(list(combined_indices))
            target_size = int(self.input_size * self.density * max(1, len(token_ids) * 0.5))
            if len(result) > target_size:
                result = result[-target_size:]

        self.cache[text] = result
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
            
        return result

    def decode(self, sdr: List[int], candidates: List[str]) -> str:
        if not sdr: return ""
        best_word = "<unk>"
        best_overlap = -1
        sdr_set = set(sdr)
        
        for word in candidates:
            target_sdr = self.encode(word)
            overlap = len(sdr_set.intersection(target_sdr))
            if overlap > best_overlap:
                best_overlap = overlap
                best_word = word
        return best_word