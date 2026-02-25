_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/spiking_causal_lm.py",
    "//": "ファイルの日本語タイトル: スパイキング因果言語モデル (予測符号化 + Attention)",
    "//": "ファイルの目的や内容: nn.SNNModuleを使用し、Predictive Coding層とSpike Self-Attentionを融合したTransformer代替の次世代ジェネレーティブモデル。トークンIDベースの処理に変更し、デコード時のループ防止機能を実装。"
}

import os
import json
import pickle
import random
from typing import List, Dict, Optional

from sara_engine import nn

class SpikingCausalLMConfig:
    def __init__(self, vocab_size: int = 256, embed_dim: int = 128, context_size: int = 64):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_size = context_size

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "context_size": self.context_size
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class PredictiveTransformerBlock(nn.SNNModule):
    def __init__(self, embed_dim: int, context_size: int):
        super().__init__()
        # 予測符号化を用いて入力を高次元表現に変換(省エネ化)
        self.predictive_encoder = nn.PredictiveSpikeLayer(in_features=embed_dim, out_features=embed_dim, density=0.3)
        # 文脈の依存関係を捉える
        self.attention = nn.SpikeSelfAttention(embed_dim=embed_dim, density=0.2, context_size=context_size)

    def forward(self, spikes: List[int], learning: bool = False) -> List[int]:
        # 予測誤差のみがAttentionへ送られる
        pred_spikes = self.predictive_encoder(spikes, learning=learning)
        attn_spikes = self.attention(pred_spikes, learning=learning)
        
        # Residual connection (入力スパイクとAttention出力を統合)
        out_spikes = list(set(spikes + attn_spikes))
        return out_spikes


class SpikingCausalLM(nn.SNNModule):
    def __init__(self, config: SpikingCausalLMConfig):
        super().__init__()
        self.config = config
        
        # 固定の入力エンコーディング (SDRマップ)
        self.sdr_map = {}
        random.seed(42)
        for tok in range(config.vocab_size):
            self.sdr_map[tok] = random.sample(range(config.embed_dim), max(1, config.embed_dim // 10))
        random.seed()

        # Transformer代替ブロック
        self.block = PredictiveTransformerBlock(config.embed_dim, config.context_size)
        
        # 出力層(Readout): Embed次元からVocab次元への変換
        self.readout = nn.LinearSpike(in_features=config.embed_dim, out_features=config.vocab_size, density=0.5)

    def _encode_token(self, token_id: int) -> List[int]:
        return self.sdr_map.get(token_id, [])

    def forward_step(self, token_id: int, learning: bool = False, target_id: Optional[int] = None) -> List[int]:
        in_spikes = self._encode_token(token_id)
        
        # ブロックを通過
        block_out = self.block(in_spikes, learning=learning)
        
        # Readout層の出力スパイクは語彙(Vocab)のIDに直接対応する
        out_spikes = self.readout(block_out, learning=learning)
        
        # LinearSpikeは内部でポテンシャルが高い順にソートして出力するため、最初の要素が最も確信度の高いトークン
        best_token = out_spikes[0] if len(out_spikes) > 0 else 0

        # 学習時のReadout層の直接STDP補完
        if learning and target_id is not None:
             for s in block_out:
                 if s < self.readout.in_features:
                     # ターゲットIDへの結合を直接強化 (LTP)
                     current_w = self.readout.weights[s].get(target_id, 0.0)
                     self.readout.weights[s][target_id] = min(3.0, current_w + 0.3)
                     
                     # 間違った予測への結合を抑制 (LTD)
                     if best_token != target_id and best_token != 0:
                         wrong_w = self.readout.weights[s].get(best_token, 0.0)
                         self.readout.weights[s][best_token] = max(0.0, wrong_w - 0.1)

        return out_spikes

    def learn_sequence(self, input_ids: List[int]):
        """トークンIDのリストを受け取り、系列としてSTDP学習を行う"""
        self.reset_state()
        for i in range(len(input_ids) - 1):
            self.forward_step(input_ids[i], learning=True, target_id=input_ids[i+1])

    def generate(self, input_ids: List[int], max_length: int = 50) -> List[int]:
        """トークンIDのリストから次のトークンを予測し、生成されたIDリストを返す"""
        self.reset_state()
        if not input_ids:
            return []
            
        # コンテキストの構築
        last_out_spikes = []
        for token_id in input_ids:
            last_out_spikes = self.forward_step(token_id, learning=False)
            
        generated_ids = []
        
        for _ in range(max_length):
            next_token = 0
            # ループ防止機構: 直近5トークン以内に同じトークンがある場合は避けて次に確信度の高いトークンを選ぶ
            for candidate in last_out_spikes:
                if candidate not in generated_ids[-5:] and candidate != 0:
                    next_token = candidate
                    break
            
            # 全ての候補が制限に引っかかった場合や候補がない場合は、一番上のトークンを強制採用
            if next_token == 0 and last_out_spikes:
                next_token = last_out_spikes[0]
                
            if next_token == 0: # 予測不能な場合は終了
                break
                
            generated_ids.append(next_token)
            last_out_spikes = self.forward_step(next_token, learning=False)
            
        return generated_ids

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=4)
        with open(os.path.join(save_directory, "model_state.pkl"), "wb") as f:
            pickle.dump(self.state_dict(), f)

    @classmethod
    def from_pretrained(cls, save_directory: str):
        with open(os.path.join(save_directory, "config.json"), "r", encoding="utf-8") as f:
            config = SpikingCausalLMConfig.from_dict(json.load(f))
        model = cls(config)
        state_path = os.path.join(save_directory, "model_state.pkl")
        if os.path.exists(state_path):
            with open(state_path, "rb") as f:
                model.load_state_dict(pickle.load(f))
        return model