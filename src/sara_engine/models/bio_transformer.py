# // src/sara_engine/models/bio_transformer.py
# // 生物学的スパイキングTransformerモデル
# // 目的や内容: AttentionとFFNを組み合わせ、逆伝播・行列演算不使用のTransformerアーキテクチャ全体を提供します。多言語トークンをスパイクとして処理します。

from src.sara_engine.core.bio_attention import BioSpikingSelfAttention
from src.sara_engine.core.bio_layers import BioHomeostasis, BioSpikingFFN

class BioSpikingTransformerBlock:
    def __init__(self, seq_len: int, d_model: int, d_ff: int):
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.attention = BioSpikingSelfAttention(seq_len, d_model)
        self.norm1 = BioHomeostasis(seq_len, d_model)
        self.ffn = BioSpikingFFN(seq_len, d_model, d_ff)
        self.norm2 = BioHomeostasis(seq_len, d_model)

    def forward(self, x_spikes: list[list[int]], timestep: int) -> list[list[int]]:
        attn_out = self.attention.forward(x_spikes, timestep)
        
        res_spikes = [[0 for _ in range(self.d_model)] for _ in range(self.seq_len)]
        for i in range(self.seq_len):
            for d in range(self.d_model):
                res_spikes[i][d] = 1 if (x_spikes[i][d] > 0 or attn_out[i][d] > 0) else 0
                
        norm1_out = self.norm1.forward(res_spikes)
        
        ffn_out = self.ffn.forward(norm1_out, timestep)
        
        out_spikes = [[0 for _ in range(self.d_model)] for _ in range(self.seq_len)]
        for i in range(self.seq_len):
            for d in range(self.d_model):
                out_spikes[i][d] = 1 if (norm1_out[i][d] > 0 or ffn_out[i][d] > 0) else 0
                
        return self.norm2.forward(out_spikes)

class BioSpikingTransformer:
    def __init__(self, num_layers: int, seq_len: int, d_model: int, d_ff: int):
        self.layers = [BioSpikingTransformerBlock(seq_len, d_model, d_ff) for _ in range(num_layers)]

    def forward(self, x_spikes: list[list[int]], timestep: int) -> list[list[int]]:
        current_spikes = x_spikes
        for layer in self.layers:
            current_spikes = layer.forward(current_spikes, timestep)
        return current_spikes