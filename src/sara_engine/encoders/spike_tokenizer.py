_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/encoders/spike_tokenizer.py",
    "//": "タイトル: スパイク トークナイザー",
    "//": "目的: 自然言語テキストをSNNに入力するためのトークンID列に変換(エンコード)し、出力をテキストに戻す(デコード)軽量モジュール。"
}

import json
import os
from typing import List, Dict

class SpikeTokenizer:
    def __init__(self):
        # 特殊トークンを初期登録
        self.vocab_to_id: Dict[str, int] = {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3}
        self.id_to_vocab: Dict[int, str] = {0: "[UNK]", 1: "[PAD]", 2: "[BOS]", 3: "[EOS]"}
        self.vocab_size: int = 4

    def train(self, texts: List[str]):
        """
        学習データから語彙(Vocabulary)を自動構築します。
        簡易的な空白区切り(単語レベル)と文字レベルのハイブリッド。
        """
        for text in texts:
            # 日本語なども扱えるように、基本は空白区切りにしつつ、
            # 未知の文字列が来ても対応できるように柔軟に処理します。
            tokens = text.split() if " " in text else list(text)
            
            for token in tokens:
                if token not in self.vocab_to_id:
                    self.vocab_to_id[token] = self.vocab_size
                    self.id_to_vocab[self.vocab_size] = token
                    self.vocab_size += 1

    def encode(self, text: str) -> List[int]:
        """テキストをトークンIDのリストに変換します。"""
        tokens = text.split() if " " in text else list(text)
        return [self.vocab_to_id.get(token, 0) for token in tokens] # 0 is [UNK]

    def decode(self, token_ids: List[int]) -> str:
        """トークンIDのリストをテキストに復元します。"""
        # 特殊トークンを除外して結合
        words = [self.id_to_vocab.get(tid, "[UNK]") for tid in token_ids if tid > 3]
        
        # 簡易的な結合（英語なら空白、日本語ならそのまま等、高度化も可能ですがまずは空白で結合）
        return " ".join(words).replace(" 。", "。").replace(" 、", "、")

    def save(self, filepath: str):
        """構築した語彙辞書をJSON形式で保存します。"""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({"vocab_to_id": self.vocab_to_id}, f, ensure_ascii=False, indent=4)

    def load(self, filepath: str):
        """保存された語彙辞書を読み込みます。"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab_to_id = data.get("vocab_to_id", {})
        self.id_to_vocab = {int(v): k for k, v in self.vocab_to_id.items()}
        self.vocab_size = len(self.vocab_to_id)