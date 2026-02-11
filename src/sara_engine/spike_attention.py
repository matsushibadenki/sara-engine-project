# src/sara_engine/spike_attention.py
# title: Spike-based Attention Mechanism
# description: 行列演算を使わず、スパイクの一致度（Overlap）を用いて過去の記憶を参照する注意機構。

import numpy as np
from typing import List, Tuple, Dict

class SpikeAttention:
    """
    スパイクベースの注意機構 (Spike Attention)
    
    Query: 現在のニューロン発火パターン
    Key:   過去のニューロン発火パターン（短期記憶バッファ）
    Value: 過去の活動に基づく想起信号
    
    行列演算（Dot Product）の代わりに、スパイク位置の「集合積（Intersection）」を用いて
    類似度（Attention Score）を計算します。
    """
    
    def __init__(self, input_size: int, hidden_size: int, memory_size: int = 50):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size  # 保持する過去のステップ数
        
        # 記憶バッファ (Ring Buffer)
        # keys: 過去のスパイクパターンの履歴
        # values: その時の入力やコンテキスト情報（ここではシンプルにスパイク自体をValueとする）
        self.memory_keys: List[List[int]] = []
        self.memory_values: List[List[int]] = []
        
        # 学習可能な投影重み (Projection Weights) - Sparse
        # Query/Key/Value変換用だが、軽量化のためIdentityに近いスパース接続を採用
        # 入力スパイクiが、Attention空間の次元jに投影される
        self.w_query = self._init_sparse_mapping(input_size, hidden_size, density=0.1)
        self.w_key = self._init_sparse_mapping(input_size, hidden_size, density=0.1)
        self.w_value = self._init_sparse_mapping(input_size, hidden_size, density=0.1)
        
        # Attention強度調整
        self.temperature = 5.0 # 類似度の感度
        self.decay = 0.95      # 記憶の減衰率

    def _init_sparse_mapping(self, n_in: int, n_out: int, density: float) -> List[List[int]]:
        """スパースな接続マップを生成 (行列なし)"""
        mapping = []
        for _ in range(n_in):
            n_targets = int(n_out * density)
            if n_targets > 0:
                targets = np.random.choice(n_out, n_targets, replace=False).tolist()
                mapping.append(targets)
            else:
                mapping.append([])
        return mapping

    def _project(self, spikes: List[int], mapping: List[List[int]]) -> List[int]:
        """スパイクをAttention空間に投影 (発火したインデックスの変換)"""
        projected = set()
        for idx in spikes:
            if idx < len(mapping):
                for target in mapping[idx]:
                    projected.add(target)
        return list(projected)

    def update_memory(self, spikes: List[int]):
        """現在のスパイクパターンを記憶に追加"""
        # KeyとValueを生成して保存
        key = self._project(spikes, self.w_key)
        value = self._project(spikes, self.w_value)
        
        self.memory_keys.append(key)
        self.memory_values.append(value)
        
        # バッファあふれなら古いものを削除
        if len(self.memory_keys) > self.memory_size:
            self.memory_keys.pop(0)
            self.memory_values.pop(0)

    def compute(self, current_spikes: List[int]) -> List[int]:
        """
        Attentionを計算し、想起されたスパイク列を返す
        
        Args:
            current_spikes: 現在のレイヤーの発火
            
        Returns:
            List[int]: Attentionによって想起されたスパイク（Valueの重み付け和に相当）
        """
        if not self.memory_keys:
            return []
            
        # 1. Query生成
        query = self._project(current_spikes, self.w_query)
        query_set = set(query)
        
        if not query_set:
            return []

        # 2. Attention Score計算 (QueryとKeyのOverlap)
        scores = []
        for i, key in enumerate(self.memory_keys):
            # 時間的減衰（直近の記憶ほど強い）
            time_factor = self.decay ** (len(self.memory_keys) - 1 - i)
            
            # Jaccard係数風の類似度計算
            key_set = set(key)
            if not key_set:
                scores.append(0.0)
                continue
                
            intersection = len(query_set.intersection(key_set))
            union = len(query_set.union(key_set))
            
            overlap_score = (intersection / union) * self.temperature * time_factor
            scores.append(overlap_score)
        
        # 3. Softmax的選択 (Winner-Take-All for Attention)
        # スパイクベースなので、確率的にサンプリングするか、閾値以上のものを統合する
        
        attention_spikes = set()
        max_score = max(scores) if scores else 0.0
        
        if max_score > 0.1: # 閾値
            # スコアが高い上位の記憶（Top-K）からValueを取り出す
            # ここではシンプルに閾値を超えた記憶をマージ
            threshold = max_score * 0.8
            
            for i, score in enumerate(scores):
                if score >= threshold:
                    # この記憶(Value)を想起
                    for neuron_idx in self.memory_values[i]:
                        attention_spikes.add(neuron_idx)
        
        return list(attention_spikes)

    def reset(self):
        self.memory_keys.clear()
        self.memory_values.clear()