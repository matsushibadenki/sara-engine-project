import json
import os
from typing import List, Dict, Optional

class SaraTokenizer:
    def __init__(self, vocab_size: int = 2000, model_path: str = "sara_vocab.json"):
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.next_id = 0
        
        # 特殊トークン
        self.special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
        for token in self.special_tokens:
            self._add_token(token)
            
        # 既存モデルがあればロード
        if os.path.exists(self.model_path):
            self.load()

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
        """簡易的な頻度ベースのトークナイザー学習"""
        freq: Dict[str, int] = {}
        for text in corpus:
            words = text.split()
            for w in words:
                freq[w] = freq.get(w, 0) + 1
        
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        # 頻度順に登録
        for w, _ in sorted_words:
            if self.next_id >= self.vocab_size:
                break
            self._add_token(w)
        
        self.save()

    def encode(self, text: str) -> List[int]:
        ids = []
        words = text.split()
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