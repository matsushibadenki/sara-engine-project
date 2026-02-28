_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/encoders/spike_tokenizer.py",
    "//": "ファイルの日本語タイトル: スパイクトークナイザー（BPE版）",
    "//": "ファイルの目的や内容: 未知語や多言語（日本語・英語）に頑健に対応するため、本格的なBPE（Byte-Pair Encoding）サブワードトークナイザーを導入。tokenizersライブラリがない環境向けにフォールバック機能も実装。"
}

import os
import json
import re
from typing import List, Dict

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False

class SpikeTokenizer:
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.has_tokenizers = HAS_TOKENIZERS
        
        if self.has_tokenizers:
            # 本格的なBPEトークナイザーの初期化
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = Whitespace()
            self.trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], vocab_size=vocab_size)
        else:
            # 従来の簡易トークナイザー (Fallback)
            self.vocab_to_id: Dict[str, int] = {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3}
            self.id_to_vocab: Dict[int, str] = {0: "[UNK]", 1: "[PAD]", 2: "[BOS]", 3: "[EOS]"}
            self.current_vocab_size: int = 4

    def train(self, texts: List[str]):
        """コーパスから語彙を学習する"""
        if self.has_tokenizers:
            # BPEの学習
            self.tokenizer.train_from_iterator(texts, self.trainer)
            self.vocab_size = self.tokenizer.get_vocab_size()
        else:
            # 簡易学習
            for text in texts:
                if " " in text:
                    tokens = re.sub(r'([.,!?])', r' \1 ', text).split()
                else:
                    tokens = list(text)
                for token in tokens:
                    if token not in self.vocab_to_id:
                        self.vocab_to_id[token] = self.current_vocab_size
                        self.id_to_vocab[self.current_vocab_size] = token
                        self.current_vocab_size += 1
            self.vocab_size = self.current_vocab_size

    def encode(self, text: str) -> List[int]:
        """テキストをトークンIDのリストに変換"""
        if self.has_tokenizers:
            return self.tokenizer.encode(text).ids
        else:
            if " " in text:
                tokens = re.sub(r'([.,!?])', r' \1 ', text).split()
            else:
                tokens = list(text)
            return [self.vocab_to_id.get(token, 0) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """トークンIDのリストをテキストに復元"""
        if self.has_tokenizers:
            return self.tokenizer.decode(token_ids)
        else:
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
        """トークナイザーの語彙データを保存"""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        if self.has_tokenizers:
            # 拡張子を統一するためにJSON形式で保存
            if not filepath.endswith('.json'):
                filepath += '.json'
            self.tokenizer.save(filepath)
        else:
            if not filepath.endswith('.json'):
                filepath += '.json'
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({"vocab_to_id": self.vocab_to_id}, f, ensure_ascii=False, indent=4)

    def load(self, filepath: str):
        """トークナイザーの語彙データを読み込み"""
        if not filepath.endswith('.json'):
            filepath += '.json'
            
        if self.has_tokenizers:
            self.tokenizer = Tokenizer.from_file(filepath)
            self.vocab_size = self.tokenizer.get_vocab_size()
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.vocab_to_id = data.get("vocab_to_id", {})
            self.id_to_vocab = {int(v): k for k, v in self.vocab_to_id.items()}
            self.vocab_size = len(self.vocab_to_id)
            
    def get_vocab(self) -> Dict[str, int]:
        """現在の語彙（単語からIDへのマッピング）を返す"""
        if self.has_tokenizers:
            return self.tokenizer.get_vocab()
        else:
            return self.vocab_to_id