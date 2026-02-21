_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/utils/tokenizer.py",
    "//": "タイトル: SARA トークナイザー",
    "//": "目的: Janomeによる自動形態素解析（分かち書き）を導入し、自然な日本語入力に対応。空白の有無に依存せず常に一貫したトークン化を行うよう修正。"
}

import json
import os
from typing import List, Dict, Optional, Any

try:
    from janome.tokenizer import Tokenizer as JanomeTokenizer  # type: ignore
    _HAS_JANOME = True
except ImportError:
    _HAS_JANOME = False

class SaraTokenizer:
    def __init__(self, vocab_size: int = 2000, model_path: str = "workspace/sara_vocab.json"):
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.next_id = 0
        self._janome_tokenizer: Any = None
        
        self.special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
        for token in self.special_tokens:
            self._add_token(token)
            
        if os.path.exists(self.model_path):
            self.load()

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        
        if _HAS_JANOME:
            if self._janome_tokenizer is None:
                self._janome_tokenizer = JanomeTokenizer()
            # 形態素解析を行い、不要なスペースのみのトークンは除外する
            return [token.surface for token in self._janome_tokenizer.tokenize(text) if token.surface.strip()]
        else:
            # Janomeがない場合のフォールバック
            return [w for w in text.replace("　", " ").split(" ") if w]

    def _add_token(self, token: str) -> int:
        if token not in self.vocab:
            if self.next_id < self.vocab_size:
                tid = self.next_id
                self.vocab[token] = tid
                self.id_to_token[tid] = token
                self.next_id += 1
                return tid
            else:
                return self.vocab["<unk>"]
        return self.vocab[token]

    def train(self, corpus: List[str]):
        freq: Dict[str, int] = {}
        for text in corpus:
            words = self.split_text(text)
            for w in words:
                freq[w] = freq.get(w, 0) + 1
        
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        for w, _ in sorted_words:
            if self.next_id >= self.vocab_size:
                break
            self._add_token(w)
        
        self.save()

    def encode(self, text: str) -> List[int]:
        ids = []
        words = self.split_text(text)
        for w in words:
            ids.append(self.vocab.get(w, self.vocab["<unk>"]))
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for i in ids:
            tokens.append(self.id_to_token.get(i, "<unk>"))
        return " ".join(tokens)

    def save(self):
        data = {
            "vocab": self.vocab,
            "next_id": self.next_id
        }
        with open(self.model_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        try:
            with open(self.model_path, 'r') as f:
                data = json.load(f)
            self.vocab = data["vocab"]
            self.next_id = data["next_id"]
            self.id_to_token = {v: k for k, v in self.vocab.items()}
        except Exception as e:
            print(f"Warning: Failed to load tokenizer model: {e}")