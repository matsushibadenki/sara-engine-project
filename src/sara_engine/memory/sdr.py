_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/memory/sdr.py",
    "//": "タイトル: 疎分散表現 (SDR) エンコーダ (意味ネットワーク & VSA統合版)",
    "//": "目的: VSAの巡回シフト(Circular Shift)を用いた役割バインディング(Role Binding)を実装し、文法構造をSDRにエンコードする。"
}

import numpy as np
from collections import defaultdict
from typing import List, Dict, Set
import sys
import os
import random
import hashlib

try:
    from ..utils.tokenizer import SaraTokenizer
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from utils.tokenizer import SaraTokenizer


class SemanticNetwork:
    """
    Random Indexingとヘッブ則の知見に基づく意味ネットワーク構築クラス。
    共起関係からSDRのビットを連合(Binding)させ、意味的オーバーラップを形成する。
    """
    def __init__(self, sdr_size: int, density: float):
        self.sdr_size = sdr_size
        self.density = density
        self.active_bits = int(sdr_size * density)
        self.co_occurrence: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.semantic_sdrs: Dict[int, List[int]] = {}

    def build_from_corpus(self, tokenized_corpus: List[List[int]], window_size: int = 4):
        for sentence in tokenized_corpus:
            length = len(sentence)
            for i in range(length):
                target = sentence[i]
                start = max(0, i - window_size)
                end = min(length, i + window_size + 1)
                for j in range(start, end):
                    if i != j:
                        context_word = sentence[j]
                        self.co_occurrence[target][context_word] += 1

    def fuse_semantics_into_sdrs(self, base_sdrs: Dict[int, List[int]], epochs: int = 2) -> Dict[int, List[int]]:
        current_sdrs = {k: set(v) for k, v in base_sdrs.items()}
        
        for _ in range(epochs):
            next_sdrs = {}
            for target, contexts in self.co_occurrence.items():
                if target not in current_sdrs:
                    continue
                
                new_sdr_set = set(current_sdrs[target])
                sorted_contexts = sorted(contexts.items(), key=lambda x: x[1], reverse=True)[:5]
                
                for context_id, freq in sorted_contexts:
                    if context_id in current_sdrs:
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
    テキストからSDRへのエンコードを行うクラス。
    意味ネットワークによる学習と、VSAの役割バインディング機能を統合。
    """
    def __init__(self, input_size: int, density: float = 0.02, use_tokenizer: bool = True, cache_size: int = 10000, apply_vsa: bool = True):
        self.input_size = input_size
        self.density = density
        self.use_tokenizer = use_tokenizer
        self.cache_size = cache_size
        self.apply_vsa = apply_vsa
        self.token_sdr_map: Dict[int, List[int]] = {}
        self.semantic_net = SemanticNetwork(input_size, density)
        
        if self.use_tokenizer:
            self.tokenizer = SaraTokenizer()

        # VSAにおける役割(Role)ごとの巡回シフト量
        # SDRサイズの適当な割合でシフトさせ、それぞれの役割を直交空間に配置する
        self.role_offsets = {
            "SUBJECT": 0,                      # 主語はシフトなし
            "OBJECT": int(input_size * 0.25),  # 目的語は1/4シフト
            "VERB": int(input_size * 0.50),    # 述語は1/2シフト
            "MODIFIER": int(input_size * 0.75),# 修飾語は3/4シフト
            "DEFAULT": 0
        }

    def train_semantic_network(self, corpus: List[str], window_size: int = 4, epochs: int = 2):
        if not self.use_tokenizer:
            print("Warning: Semantic network training requires use_tokenizer=True.")
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
        rng = np.random.RandomState(token_id)
        target_n = int(self.input_size * self.density)
        indices = rng.choice(self.input_size, target_n, replace=False)
        indices.sort()
        return indices.tolist()
        
    def _get_token_sdr(self, token_id: int) -> List[int]:
        if token_id in self.token_sdr_map:
            return self.token_sdr_map[token_id]
        
        sdr = self._get_base_token_sdr(token_id)
        if len(self.token_sdr_map) < self.cache_size:
            self.token_sdr_map[token_id] = sdr
        return sdr

    def _determine_roles_by_ids(self, token_ids: List[int]) -> List[str]:
        """トークンの並びと助詞のIDから簡易的な文法的役割(Role)を推定する"""
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
            # 次の助詞を見て現在の単語の役割を決定
            if i + 1 < len(token_ids):
                next_tid = token_ids[i+1]
                if next_tid in [wa_id, ga_id]:
                    roles[i] = "SUBJECT"
                elif next_tid in [wo_id, ni_id]:
                    roles[i] = "OBJECT"
                elif next_tid in [no_id]:
                    roles[i] = "MODIFIER"
                    
            # 自身が動詞リストに含まれていればVERB
            if tid in verb_ids:
                roles[i] = "VERB"
                
        return roles

    def encode(self, text: str) -> List[int]:
        """文字列をSDR(オンビットのインデックスリスト)に変換"""
        if self.use_tokenizer:
            token_ids = self.tokenizer.encode(text)
            roles = self._determine_roles_by_ids(token_ids) if self.apply_vsa else ["DEFAULT"] * len(token_ids)
            
            union_set: Set[int] = set()
            for tid, role in zip(token_ids, roles):
                base_sdr = self._get_token_sdr(tid)
                
                # VSAのRole Binding (役割に応じた巡回シフト)
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
            rng = np.random.RandomState(seed)
            target_n = int(self.input_size * self.density)
            indices = rng.choice(self.input_size, target_n, replace=False)
            indices.sort()
            return indices.tolist()