_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/transformer.py",
    "//": "タイトル: Spike Transformer モデル",
    "//": "目的: Attention層(時間)とSTDP層(空間)を統合し、完全なSNN Transformerを構築する。"
}

import json
import random
from typing import List, Dict, Optional

from sara_engine.core.spike_attention import SpikeAttention
from sara_engine.learning.stdp import STDPLayer

class SpikePositionalEncoding:
    """
    randomを用いて疎な位置エンコーディングを生成。
    """
    def __init__(self, d_model: int, max_len: int = 1000, density: float = 0.1):
        self.d_model = d_model
        self.max_len = max_len
        self.pos_spikes: Dict[int, List[int]] = {}
        self.density = density
        
        # 再現性のためのシード固定
        self.rng = random.Random(42)
        
    def get_pe(self, pos: int) -> List[int]:
        if pos in self.pos_spikes:
            return self.pos_spikes[pos]
        
        # その位置専用のランダムスパイクパターンを生成
        num_active = max(1, int(self.d_model * self.density))
        # 位置ごとに異なるシードを使用
        local_rng = random.Random(42 + pos)
        spikes = sorted(local_rng.sample(range(self.d_model), num_active))
        
        self.pos_spikes[pos] = spikes
        return spikes

class SpikeTransformerBlock:
    def __init__(self, d_model: int, num_heads: int, ffn_hidden: int):
        self.attention = SpikeAttention(decay_rate=0.9, threshold=1.2)
        # STDP層: Feed Forward Networkの代わり
        # ※ ここではインターフェースのみ利用。
        self.ffn = STDPLayer(d_model, d_model) 
        self.norm1_params = [0.0] * d_model # 簡易Normalization用
        
    def forward(self, x_spikes: List[int], learning: bool = True) -> List[int]:
        # 1. Self-Attention
        # Q, K, Vは現状すべて同じx_spikesから生成（簡易版）
        attn_out, _ = self.attention.process_step(x_spikes, x_spikes, x_spikes)
        
        # Residual Connection (OR演算)
        x_post_attn = sorted(list(set(x_spikes).union(set(attn_out))))
        
        # 2. Feed Forward (STDP)
        ffn_out = self.ffn.forward(x_post_attn, learning=learning)
        
        # Residual Connection
        output = sorted(list(set(x_post_attn).union(set(ffn_out))))
        
        return output

class SpikeTransformer:
    def __init__(self, vocab_size: int, d_model: int = 256, num_layers: int = 2, num_heads: int = 4):
        self.d_model = d_model
        self.pe = SpikePositionalEncoding(d_model)
        self.blocks = [SpikeTransformerBlock(d_model, num_heads, d_model*2) for _ in range(num_layers)]
        
        # 入力埋め込み（簡易版：語彙IDをランダムSDRに変換）
        self.embedding_table: Dict[int, List[int]] = {}
        self.rng = random.Random(100)
        
    def _get_embedding(self, token_id: int) -> List[int]:
        if token_id not in self.embedding_table:
            num_active = int(self.d_model * 0.05)
            # トークンIDごとの決定論的乱数
            local_rng = random.Random(token_id * 777)
            self.embedding_table[token_id] = sorted(local_rng.sample(range(self.d_model), num_active))
        return self.embedding_table[token_id]

    def reset(self):
        for block in self.blocks:
            block.attention.reset()
            # FFN(STDP)の状態はリセットしない（学習結果なので）

    def compute(self, token_ids: List[int], learning: bool = True) -> List[List[int]]:
        """
        トークンIDのシーケンスを入力し、各ステップの出力SDRリストを返す
        """
        self.reset()
        outputs = []
        
        for pos, token in enumerate(token_ids):
            # 1. Embedding + Positional Encoding
            emb = self._get_embedding(token)
            pe = self.pe.get_pe(pos)
            # 加算（和集合）
            x = sorted(list(set(emb).union(set(pe))))
            
            # 2. Transformer Blocks
            for block in self.blocks:
                x = block.forward(x, learning=learning)
            
            outputs.append(x)
            
        return outputs