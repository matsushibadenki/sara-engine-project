# {
#     "//": "ディレクトリパス: src/sara_engine/utils/tokenizer.py",
#     "//": "ファイルの日本語タイトル: SARA BPE トークナイザー",
#     "//": "ファイルの目的や内容: SNNの推論速度と精度を劇的に向上させるため、BPE (Byte-Pair Encoding) サブワードアルゴリズムをネイティブ実装。頻出する文字列を自動結合し、行列演算なしで高速なトークン化を行う。正規表現のエスケープエラーを修正。"
# }

import json
import os
import re
from typing import List, Dict, Tuple, Any
from .project_paths import ensure_parent_directory, workspace_path

try:
    from janome.tokenizer import Tokenizer as JanomeTokenizer  # type: ignore
    _HAS_JANOME = True
except ImportError:
    _HAS_JANOME = False

class SaraTokenizer:
    def __init__(self, vocab_size: int = 4096, model_path: str = workspace_path("tokenizers", "sara_vocab.json")):
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        self.next_id = 0
        self._janome_tokenizer: Any = None
        
        self.special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>", "\n", " ", "　"]
        for token in self.special_tokens:
            self._add_token(token)
            
        if os.path.exists(self.model_path):
            self.load()

    def pre_tokenize(self, text: str) -> List[str]:
        """単語境界を跨いだ不要な結合を防ぐための事前分割"""
        if not text:
            return []
        
        if _HAS_JANOME:
            if self._janome_tokenizer is None:
                self._janome_tokenizer = JanomeTokenizer()
            return [token.surface for token in self._janome_tokenizer.tokenize(text)]
        else:
            # Janomeがない場合の簡易的な単語・句読点分割
            return [w for w in re.split(r'(\s+|[。、！？.!,?])', text) if w]

    def split_text(self, text: str) -> List[str]:
        """互換API: 学習済みBPE規則を考慮しつつ、空白や句読点を保持して分割する。"""
        pieces: List[str] = []
        for token in self.pre_tokenize(text):
            if token == "":
                continue
            if token.isspace() or token in {"。", "、", "！", "？", ".", ",", "!", "?"}:
                pieces.append(token)
                continue
            pieces.extend(self._tokenize_word(token))
        return pieces

    def _add_token(self, token: str) -> int:
        if token not in self.vocab:
            if self.next_id >= self.vocab_size:
                self.vocab_size = self.next_id + 1
            tid = self.next_id
            self.vocab[token] = tid
            self.id_to_token[tid] = token
            self.next_id += 1
            return tid
        return self.vocab[token]

    def _get_stats(self, splits: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        counts: Dict[Tuple[str, str], int] = {}
        for word, freq in splits.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                counts[pair] = counts.get(pair, 0) + freq
        return counts

    def _merge_vocab(self, pair: Tuple[str, str], v_in: Dict[str, int]) -> Dict[str, int]:
        v_out: Dict[str, int] = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        replacement = ''.join(pair)
        for word in v_in:
            # 修正箇所: 文字列を直接渡さず lambda を使うことで、エスケープ文字のエラーを回避
            w_out = p.sub(lambda _: replacement, word)
            v_out[w_out] = v_in[word]
        return v_out

    def train(self, corpus: List[str]):
        """BPEアルゴリズムを用いた語彙の学習"""
        word_freqs: Dict[str, int] = {}
        for text in corpus:
            words = self.pre_tokenize(text)
            for w in words:
                word_freqs[w] = word_freqs.get(w, 0) + 1
        
        # 初期状態: 各単語を文字単位に分割してスペースで結合
        splits = {" ".join(list(w)): freq for w, freq in word_freqs.items()}
        
        # 基本となる文字をVocabに登録
        for word in word_freqs.keys():
            for char in word:
                self._add_token(char)
                
        # BPEマージループ (頻出する文字ペアを結合していく)
        num_merges = self.vocab_size - self.next_id
        for i in range(num_merges):
            stats = self._get_stats(splits)
            if not stats:
                break
            # 最も頻出するペアを取得
            best_pair = max(stats, key=stats.get)
            splits = self._merge_vocab(best_pair, splits)
            
            # 学習したマージ規則を順位付きで保存
            self.merge_ranks[best_pair] = len(self.merge_ranks)
            self._add_token("".join(best_pair))
            
            if self.next_id >= self.vocab_size:
                break
                
        self.save()

    def _tokenize_word(self, word: str) -> List[str]:
        """学習されたBPEマージルールに従って1単語をサブワードに分割"""
        if len(word) <= 1:
            return [word]
            
        symbols = list(word)
        while True:
            pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
            if not pairs:
                break
                
            # 学習時と全く同じ順序（優先度）でマージを適用する
            best_pair = None
            lowest_rank = float('inf')
            for pair in pairs:
                rank = self.merge_ranks.get(pair, float('inf'))
                if rank < lowest_rank:
                    lowest_rank = rank
                    best_pair = pair
                    
            if best_pair is None:
                break
                
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols)-1 and (symbols[i], symbols[i+1]) == best_pair:
                    new_symbols.append("".join(best_pair))
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text: str) -> List[int]:
        """テキストをBPEトークンIDのリストに変換"""
        ids = []
        words = self.pre_tokenize(text)
        for w in words:
            subwords = self._tokenize_word(w)
            for sw in subwords:
                ids.append(self.vocab.get(sw, self.vocab.get("<unk>", 1)))
        return ids

    def decode(self, ids: List[int]) -> str:
        """トークンIDのリストを自然なテキストに復元"""
        tokens = [self.id_to_token.get(i, "<unk>") for i in ids]
        return "".join(tokens)

    def save(self):
        output_path = ensure_parent_directory(self.model_path)
        data = {
            "vocab": self.vocab,
            "next_id": self.next_id,
            "merges": [[p[0], p[1]] for p in sorted(self.merge_ranks.keys(), key=lambda k: self.merge_ranks[k])]
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        try:
            with open(self.model_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.vocab = data["vocab"]
            self.next_id = data["next_id"]
            self.id_to_token = {int(v): k for k, v in self.vocab.items()}
            if "merges" in data:
                self.merge_ranks = {tuple(p): i for i, p in enumerate(data["merges"])}
        except Exception as e:
            print(f"Warning: Failed to load tokenizer model: {e}")
