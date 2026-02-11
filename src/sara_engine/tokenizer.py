# src/sara_engine/tokenizer.py
# SARA Engine Native Tokenizer (BPE Implementation)
# 外部ライブラリ非依存・軽量・自己完結型

import os
import json
import collections

class SaraTokenizer:
    def __init__(self, vocab_size=1000, model_path="sara_vocab.json"):
        self.vocab_size = vocab_size
        self.model_path = model_path
        
        # 特殊トークン
        self.vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2, # Begin of sentence
            "<eos>": 3, # End of sentence
        }
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        self.merges = {}      # pair -> new_token
        self.merge_order = [] # [(pair, token), ...] 学習順序を保存
        
        # 既存モデルがあればロード
        if os.path.exists(model_path):
            self.load()

    def train(self, corpus_list):
        """
        BPE (Byte Pair Encoding) アルゴリズムによる学習
        """
        print(f"Training tokenizer on {len(corpus_list)} sentences...")
        
        # 1. 前処理: 単語ごとの頻度をカウント
        word_counts = collections.Counter()
        for text in corpus_list:
            text = text.lower().strip()
            if not text: continue
            words = text.split()
            for w in words:
                # 文字区切りにし、末尾に識別子をつける
                tokenized_word = " ".join(list(w)) + " </w>"
                word_counts[tokenized_word] += 1

        # 2. 指定サイズになるまでマージを繰り返す
        current_vocab_size = len(self.vocab) + len(set("".join(corpus_list))) 
        num_merges = self.vocab_size - current_vocab_size
        
        if num_merges < 0:
            num_merges = 100 

        print(f"Start merging... (Target: {num_merges} merges)")

        for i in range(num_merges):
            pairs = self._get_stats(word_counts)
            if not pairs:
                break
            
            # 最も頻出するペアを見つける
            best_pair = max(pairs, key=pairs.get)
            new_token = "".join(best_pair)
            
            # 順序を記録して保存
            self.merges[best_pair] = new_token
            self.merge_order.append((best_pair, new_token))
            
            # 単語リスト内のペアを結合
            word_counts = self._merge_vocab(best_pair, word_counts)
            
            if (i+1) % 100 == 0:
                print(f"  Merge {i+1}/{num_merges}: {best_pair} -> {new_token}")

        # 3. 最終的な語彙リストを構築
        next_id = 4
        all_tokens = set()
        for word in word_counts:
            tokens = word.split()
            for t in tokens:
                all_tokens.add(t)
                
        for token in sorted(list(all_tokens)):
            if token not in self.vocab:
                self.vocab[token] = next_id
                self.inverse_vocab[next_id] = token
                next_id += 1
                
        print(f"Training complete. Final vocab size: {len(self.vocab)}")
        self.save()

    def encode(self, text):
        """
        テキスト -> トークンIDのリスト (学習順序を厳守)
        """
        if not text: return []
        text = text.lower().strip()
        words = text.split()
        ids = []
        
        for word in words:
            # 1. 文字単位に分割
            word_tokens = list(word) + ["</w>"]
            
            # 2. 学習したマージ順序に従って適用
            for pair, new_token in self.merge_order:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i+1]) == tuple(pair):
                        word_tokens[i] = new_token
                        del word_tokens[i+1]
                        # マージしたらインデックスを進めず再チェック（連鎖マージ対応）
                    else:
                        i += 1
            
            # 3. ID変換
            for t in word_tokens:
                if t in self.vocab:
                    ids.append(self.vocab[t])
                else:
                    ids.append(self.vocab["<unk>"])
                    
        return ids

    def decode(self, ids):
        tokens = []
        for i in ids:
            token = self.inverse_vocab.get(i, "<unk>")
            tokens.append(token)
        text = "".join(tokens).replace("</w>", " ").strip()
        return text

    def _get_stats(self, vocab):
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    def _merge_vocab(self, pair, v_in):
        """
        指定されたペアを結合して語彙データを更新（トークン境界保護版）
        """
        v_out = {}
        replacement = "".join(pair)
        
        for word, freq in v_in.items():
            w_tokens = word.split()
            new_tokens = []
            i = 0
            
            while i < len(w_tokens):
                # ペアが一致するか確認（境界チェック）
                if (i < len(w_tokens) - 1 and 
                    w_tokens[i] == pair[0] and 
                    w_tokens[i+1] == pair[1]):
                    new_tokens.append(replacement)
                    i += 2
                else:
                    new_tokens.append(w_tokens[i])
                    i += 1
            
            v_out[" ".join(new_tokens)] = freq
            
        return v_out

    def save(self):
        # タプルはJSONキーにできないので文字列化
        merges_str = {f"{k[0]} {k[1]}": v for k, v in self.merges.items()}
        # merge_orderも保存
        merge_order_list = [[list(p), t] for p, t in self.merge_order]
        
        data = {
            "vocab": self.vocab,
            "merges": merges_str,
            "merge_order": merge_order_list
        }
        with open(self.model_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {self.model_path}")

    def load(self):
        with open(self.model_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.vocab = data["vocab"]
            self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
            
            self.merges = {}
            if "merges" in data:
                for k, v in data["merges"].items():
                    pair = tuple(k.split(" "))
                    self.merges[pair] = v
            
            self.merge_order = []
            if "merge_order" in data:
                for p_list, t in data["merge_order"]:
                    self.merge_order.append((tuple(p_list), t))
                    
        print(f"Tokenizer loaded. Vocab size: {len(self.vocab)}")