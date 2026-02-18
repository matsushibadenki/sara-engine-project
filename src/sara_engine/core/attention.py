_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/attention.py",
    "//": "タイトル: Spike-based Multi-Head Attention",
    "//": "目的: 行列演算を使わず、SDRの共通集合（Overlap）で類似度を計算する。Rust実装があればそちらを優先する。"
}

import random
from typing import List, Dict, Set, Optional

try:
    from .. import sara_rust_core  # type: ignore
    RUST_AVAILABLE = True
except ImportError:
    try:
        import sara_rust_core  # type: ignore
        RUST_AVAILABLE = True
    except ImportError:
        RUST_AVAILABLE = False

class SpikeAttention:
    def __init__(self, input_size: int, hidden_size: int, memory_size: int = 50, num_heads: int = 4, use_rust: Optional[bool] = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_heads = num_heads
        
        if use_rust is None:
            self.use_rust = RUST_AVAILABLE
        else:
            self.use_rust = use_rust and RUST_AVAILABLE
        
        if self.use_rust:
            self.core = sara_rust_core.RustSpikeAttention(input_size, hidden_size, num_heads, memory_size)
            print("SpikeAttention: Rust core initialized.")
        else:
            print("SpikeAttention: Python fallback mode.")
            # メモリ: [TimeStep][Head] -> Set[int] (スパイクインデックスの集合)
            self.memory_keys: List[List[Set[int]]] = [] 
            self.memory_values: List[List[List[int]]] = [] 
            
            # 射影重み (スパース結合): [input_idx] -> [mapped_hidden_indices...]
            self.w_query = self._init_projection(input_size, hidden_size)
            self.w_key = self._init_projection(input_size, hidden_size)
            self.w_value = self._init_projection(input_size, hidden_size)

    def _init_projection(self, size_in: int, size_out: int) -> List[List[int]]:
        """ランダムなスパース結合を作成"""
        weights = []
        for _ in range(size_in):
            # 1つの入力から少数の出力ニューロンへ接続
            n_connections = max(1, int(size_out * 0.05))
            targets = random.sample(range(size_out), n_connections)
            weights.append(targets)
        return weights

    def _project(self, input_spikes: List[int], mapping: List[List[int]]) -> List[int]:
        """スパイクを射影してWinner-Take-Allで活性化させる"""
        potentials = {}
        for idx in input_spikes:
            if idx < len(mapping):
                for target in mapping[idx]:
                    potentials[target] = potentials.get(target, 0) + 1
        
        if not potentials:
            return []
            
        # 上位K個を発火させる (Sparsity維持)
        k = max(1, int(self.hidden_size * 0.1))
        # 電位順にソート
        sorted_neurons = sorted(potentials.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, val in sorted_neurons[:k]]

    def compute(self, input_spikes: List[int]) -> List[int]:
        if self.use_rust:
            return self.core.compute(input_spikes)
            
        # 1. 現在のスパイクをQ, K, Vに変換
        q_full = self._project(input_spikes, self.w_query)
        k_full = self._project(input_spikes, self.w_key)
        v_full = self._project(input_spikes, self.w_value)
        
        # ヘッドごとに分割 (単純な剰余分割)
        q_heads: List[Set[int]] = [set() for _ in range(self.num_heads)]
        k_heads: List[Set[int]] = [set() for _ in range(self.num_heads)]
        v_heads: List[List[int]] = [[] for _ in range(self.num_heads)]
        
        for idx in q_full: q_heads[idx % self.num_heads].add(idx)
        for idx in k_full: k_heads[idx % self.num_heads].add(idx)
        for idx in v_full: v_heads[idx % self.num_heads].append(idx)
        
        # 2. メモリに現在のK, Vを保存
        if len(self.memory_keys) >= self.memory_size:
            self.memory_keys.pop(0)
            self.memory_values.pop(0)
            
        self.memory_keys.append(k_heads)
        self.memory_values.append(v_heads)
        
        if len(self.memory_keys) < 2:
            return []
            
        # 3. Attention計算 (Set Intersection)
        context_potentials: Dict[int, float] = {}
        
        # 最新（自分自身）を除く過去のメモリと比較
        current_time = len(self.memory_keys) - 1
        
        for h in range(self.num_heads):
            q_set = q_heads[h]
            if not q_set: continue
            
            for t, past_keys_heads in enumerate(self.memory_keys[:-1]): # 自分自身は含めない
                k_set = past_keys_heads[h]
                if not k_set: continue
                
                # 共通集合のサイズ = 類似度
                overlap = len(q_set.intersection(k_set))
                
                if overlap > 0:
                    # 時間的減衰
                    age = current_time - t
                    decay = 0.9 ** age # 古い記憶ほど薄れる
                    score = overlap * decay
                    
                    # 関連するValueのニューロンに電位を加算
                    for v_idx in self.memory_values[t][h]:
                        context_potentials[v_idx] = context_potentials.get(v_idx, 0.0) + score
                        
        # 4. コンテキスト出力の生成 (WTA)
        if not context_potentials:
            return []
            
        # 電位上位を取得
        result_k = max(1, int(self.hidden_size * 0.1))
        sorted_context = sorted(context_potentials.items(), key=lambda x: x[1], reverse=True)
        return sorted([idx for idx, _ in sorted_context[:result_k]])

    def reset(self):
        if self.use_rust:
            self.core.reset()
        else:
            self.memory_keys = []
            self.memory_values = []