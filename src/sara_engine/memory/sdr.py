# Directory Path: src/sara_engine/memory/sdr.py
# English Title: PPMI-Enhanced Sparse Distributed Representation Encoder
# Purpose/Content: Numpyへの依存を完全に排除し純粋なPython実装化。単なる共起回数ではなくPPMI(正の自己相互情報量)を用いて意味ネットワークを構築することで、ストップワードのノイズを排除し、極めて精度の高い意味融合(SDR)を実現する。多言語対応。

import hashlib
import random
import math
from typing import List, Dict, Set
from collections import defaultdict

try:
    from sara_engine.utils.tokenizer import SaraTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False


class SemanticNetwork:
    """
    PPMI (Positive Pointwise Mutual Information) を用いた高精度意味ネットワーク。
    ノイズとなる高頻度語を抑圧し、本当に意味の近いSDR同士を融合させる。
    """

    def __init__(self, sdr_size: int, density: float):
        self.sdr_size = sdr_size
        self.density = density
        self.active_bits = int(sdr_size * density)
        
        self.co_occurrence: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.word_counts: Dict[int, int] = defaultdict(int)
        self.total_words = 0
        self.total_pairs = 0
        self.semantic_sdrs: Dict[int, List[int]] = {}

    def get_status(self, lang: str = "en") -> str:
        messages = {
            "en": "SemanticNetwork: Pure Python mode. PPMI fusion active.",
            "ja": "SemanticNetwork: 純粋Pythonモード。PPMI意味融合が有効です。",
            "fr": "SemanticNetwork: Mode Python pur. Fusion PPMI active."
        }
        return messages.get(lang, messages["en"])

    def build_from_corpus(self, tokenized_corpus: List[List[int]], window_size: int = 4):
        for sentence in tokenized_corpus:
            length = len(sentence)
            for i in range(length):
                target = sentence[i]
                self.word_counts[target] += 1
                self.total_words += 1
                
                start = max(0, i - window_size)
                end = min(length, i + window_size + 1)
                for j in range(start, end):
                    if i != j:
                        context_word = sentence[j]
                        self.co_occurrence[target][context_word] += 1
                        self.total_pairs += 1

    def _calculate_ppmi(self, target: int, context: int) -> float:
        """PPMIを計算し、偶然以上の強い結びつきのみを評価する"""
        if self.total_pairs == 0 or self.total_words == 0:
            return 0.0
            
        co_count = self.co_occurrence[target].get(context, 0)
        if co_count == 0:
            return 0.0
            
        p_target_context = co_count / self.total_pairs
        p_target = self.word_counts[target] / self.total_words
        p_context = self.word_counts[context] / self.total_words
        
        pmi = math.log2(p_target_context / (p_target * p_context))
        return max(0.0, pmi)

    def fuse_semantics_into_sdrs(self, base_sdrs: Dict[int, List[int]], epochs: int = 2) -> Dict[int, List[int]]:
        current_sdrs = {k: set(v) for k, v in base_sdrs.items()}

        for _ in range(epochs):
            next_sdrs = {}
            for target, contexts in self.co_occurrence.items():
                if target not in current_sdrs:
                    continue

                new_sdr_set = set(current_sdrs[target])
                
                # 単純な頻度ではなくPPMIスコアで上位のコンテキストを抽出
                context_scores = [(c, self._calculate_ppmi(target, c)) for c in contexts.keys()]
                sorted_contexts = sorted(context_scores, key=lambda x: x[1], reverse=True)[:5]

                for context_id, score in sorted_contexts:
                    if score > 0.5 and context_id in current_sdrs:
                        context_sdr = list(current_sdrs[context_id])
                        sample_size = min(len(context_sdr), int(self.active_bits * 0.15))
                        sampled_bits = random.sample(context_sdr, sample_size)
                        new_sdr_set.update(sampled_bits)

                if len(new_sdr_set) > self.active_bits:
                    next_sdrs[target] = set(random.sample(list(new_sdr_set), self.active_bits))
                else:
                    next_sdrs[target] = new_sdr_set

            for k, v in next_sdrs.items():
                current_sdrs[k] = v

        self.semantic_sdrs = {k: sorted(list(v)) for k, v in current_sdrs.items()}
        return self.semantic_sdrs


class SDREncoder:
    """
    Numpy非依存の純粋Python実装 SDRエンコーダ。
    """

    def __init__(self, input_size: int, density: float = 0.02, use_tokenizer: bool = True, cache_size: int = 10000, apply_vsa: bool = True):
        self.input_size = input_size
        self.density = density
        self.use_tokenizer = use_tokenizer and HAS_TOKENIZER
        self.cache_size = cache_size
        self.apply_vsa = apply_vsa
        self.token_sdr_map: Dict[int, List[int]] = {}
        self.semantic_net = SemanticNetwork(input_size, density)

        if self.use_tokenizer:
            self.tokenizer = SaraTokenizer()

        self.role_offsets = {
            "SUBJECT": 0,
            "OBJECT": int(input_size * 0.25),
            "VERB": int(input_size * 0.50),
            "MODIFIER": int(input_size * 0.75),
            "DEFAULT": 0
        }

    def train_semantic_network(self, corpus: List[str], window_size: int = 4, epochs: int = 2):
        if not self.use_tokenizer:
            return

        tokenized_corpus = [self.tokenizer.encode(text) for text in corpus]

        base_sdrs = {}
        for sentence in tokenized_corpus:
            for tid in sentence:
                if tid not in base_sdrs:
                    base_sdrs[tid] = self._get_base_token_sdr(tid)

        self.semantic_net.build_from_corpus(tokenized_corpus, window_size=window_size)
        fused_sdrs = self.semantic_net.fuse_semantics_into_sdrs(base_sdrs, epochs=epochs)

        for tid, sdr in fused_sdrs.items():
            self.token_sdr_map[tid] = sdr

    def _get_base_token_sdr(self, token_id: int) -> List[int]:
        # Numpyを排除し、組み込みのrandomモジュールでシード固定による一意なビット生成
        rng = random.Random(token_id)
        target_n = int(self.input_size * self.density)
        # 効率的に一意なインデックスをサンプリング
        indices = rng.sample(range(self.input_size), target_n)
        indices.sort()
        return indices

    def _get_token_sdr(self, token_id: int) -> List[int]:
        if token_id in self.token_sdr_map:
            return self.token_sdr_map[token_id]

        sdr = self._get_base_token_sdr(token_id)
        if len(self.token_sdr_map) < self.cache_size:
            self.token_sdr_map[token_id] = sdr
        return sdr

    def _determine_roles_by_ids(self, token_ids: List[int]) -> List[str]:
        roles = ["DEFAULT"] * len(token_ids)
        if not self.use_tokenizer:
            return roles

        wa_id = self.tokenizer.vocab.get("は", -1)
        ga_id = self.tokenizer.vocab.get("が", -1)
        wo_id = self.tokenizer.vocab.get("を", -1)
        ni_id = self.tokenizer.vocab.get("に", -1)
        no_id = self.tokenizer.vocab.get("の", -1)

        verb_ids = {self.tokenizer.vocab.get(w, -1) for w in ["持って", "住んで", "渡し", "好き", "です", "います", "ました", "食べ"]}

        for i, tid in enumerate(token_ids):
            if i + 1 < len(token_ids):
                next_tid = token_ids[i+1]
                if next_tid in [wa_id, ga_id]:
                    roles[i] = "SUBJECT"
                elif next_tid in [wo_id, ni_id]:
                    roles[i] = "OBJECT"
                elif next_tid in [no_id]:
                    roles[i] = "MODIFIER"

            if tid in verb_ids:
                roles[i] = "VERB"

        return roles

    def encode(self, text: str) -> List[int]:
        if self.use_tokenizer:
            token_ids = self.tokenizer.encode(text)
            roles = self._determine_roles_by_ids(token_ids) if self.apply_vsa else ["DEFAULT"] * len(token_ids)

            union_set: Set[int] = set()
            for tid, role in zip(token_ids, roles):
                base_sdr = self._get_token_sdr(tid)

                offset = self.role_offsets.get(role, 0)
                if offset > 0:
                    shifted_sdr = [(idx + offset) % self.input_size for idx in base_sdr]
                    union_set.update(shifted_sdr)
                else:
                    union_set.update(base_sdr)

            result = list(union_set)
            result.sort()
            return result
        else:
            hash_val = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
            seed = hash_val % (2**32 - 1)
            rng = random.Random(seed)
            target_n = int(self.input_size * self.density)
            indices = rng.sample(range(self.input_size), target_n)
            indices.sort()
            return indices