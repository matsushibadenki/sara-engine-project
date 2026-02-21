_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/encoders/spike_tokenizer.py",
    "//": "ファイルの日本語タイトル: スパイクトークナイザー",
    "//": "ファイルの目的や内容: 句読点の分離機能を強化し、未知語（UNK）の発生を抑えるよう改良したトークナイザー。"
}

import json
import os
import re
from typing import List, Dict

class SpikeTokenizer:
    def __init__(self):
        self.vocab_to_id: Dict[str, int] = {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3}
        self.id_to_vocab: Dict[int, str] = {0: "[UNK]", 1: "[PAD]", 2: "[BOS]", 3: "[EOS]"}
        self.vocab_size: int = 4

    def _tokenize(self, text: str) -> List[str]:
        """内部トークン化処理。空白区切りの言語は句読点を分離し、そうでない言語は文字単位。"""
        if " " in text:
            # 句読点の前にスペースを入れて独立したトークンにする
            text = re.sub(r'([.,!?])', r' \1 ', text)
            return text.split()
        else:
            return list(text)

    def train(self, texts: List[str]):
        for text in texts:
            tokens = self._tokenize(text)
            for token in tokens:
                if token not in self.vocab_to_id:
                    self.vocab_to_id[token] = self.vocab_size
                    self.id_to_vocab[self.vocab_size] = token
                    self.vocab_size += 1

    def encode(self, text: str) -> List[int]:
        tokens = self._tokenize(text)
        return [self.vocab_to_id.get(token, 0) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        words = [self.id_to_vocab.get(tid, "") for tid in token_ids if tid > 3]
        
        result = ""
        for word in words:
            if not word:
                continue
            if result and re.match(r'[A-Za-z0-9]', result[-1]) and re.match(r'[A-Za-z0-9]', word[0]):
                result += " " + word
            else:
                result += word
                
        return result.replace(" 。", "。").replace(" 、", "、")

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({"vocab_to_id": self.vocab_to_id}, f, ensure_ascii=False, indent=4)

    def load(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab_to_id = data.get("vocab_to_id", {})
        self.id_to_vocab = {int(v): k for k, v in self.vocab_to_id.items()}
        self.vocab_size = len(self.vocab_to_id)