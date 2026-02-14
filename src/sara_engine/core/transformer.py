_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/transformer.py",
    "//": "タイトル: 恒常性統合・遷移特化型 Spike Transformer",
    "//": "目的: STDPの暴走を防ぐシナプス恒常性と厳格な不応期を導入し、S->A->R->A の連鎖を正確に学習・想起する。"
}

import numpy as np
from typing import List, Dict
from .attention import SpikeAttention

class SpikePositionalEncoding:
    def __init__(self, d_model: int, max_len: int = 1000, density: float = 0.1):
        self.d_model = d_model
        self.max_len = max_len
        self.pos_spikes = []
        rng = np.random.RandomState(42)
        num_active = max(1, int(d_model * density))
        for _ in range(max_len):
            spikes = rng.choice(d_model, num_active, replace=False).tolist()
            self.pos_spikes.append(spikes)

    def get_spikes(self, pos: int) -> List[int]:
        if pos < self.max_len:
            return self.pos_spikes[pos]
        return self.pos_spikes[-1]

class PlasticSpikeFFN:
    """
    STDP + シナプス恒常性 (Synaptic Scaling) + 厳格な不応期を備えたFFN。
    """
    def __init__(self, d_model: int, d_ff: int, learning_rate: float = 0.1):
        self.d_model = d_model
        self.d_ff = d_ff
        self.lr = learning_rate
        
        # 接続辞書 [pre] -> {post: weight}
        self.w_up: List[Dict[int, float]] = [{} for _ in range(d_model)]
        self.w_down: List[Dict[int, float]] = [{} for _ in range(d_ff)]
        self._init_sparse_weights()
        
        # トレース（1ステップ前の状態を保持）
        self.trace_pre = np.zeros(d_model)
        
        # 厳格な不応期カウンタ (発火直後のニューロンを完全に沈黙させる)
        self.refractory_o = np.zeros(d_model)

    def _init_sparse_weights(self):
        rng = np.random.RandomState()
        for i in range(self.d_model):
            targets = rng.choice(self.d_ff, max(1, int(self.d_ff * 0.1)), replace=False)
            for t in targets:
                self.w_up[i][t] = rng.uniform(0.1, 0.5)
        for i in range(self.d_ff):
            targets = rng.choice(self.d_model, max(1, int(self.d_model * 0.1)), replace=False)
            for t in targets:
                self.w_down[i][t] = rng.uniform(0.1, 0.5)

    def compute(self, input_spikes: List[int], learning: bool = False) -> List[int]:
        valid_in = [i for i in input_spikes if i < self.d_model]
        
        # 不応期のカウントダウン
        self.refractory_o = np.maximum(0, self.refractory_o - 1)
        
        if learning:
            # --- 学習フェーズ (Teacher Forcing) ---
            # 隠れ層の発火（入力と過去のトレースから）
            v_h = np.zeros(self.d_ff)
            for i in valid_in:
                for tgt, w in self.w_up[i].items():
                    v_h[tgt] += w * 2.0
            for i in range(self.d_model):
                if self.trace_pre[i] > 0:
                    for tgt, w in self.w_up[i].items():
                        v_h[tgt] += w

            fired_h = np.where(v_h >= 1.0)[0].tolist()
            k_h = max(1, int(self.d_ff * 0.05))
            if len(fired_h) > k_h:
                fired_h = np.argsort(v_h)[-k_h:].tolist()
                
            # 出力層は教師データ(valid_in)で発火
            fired_o = valid_in.copy()

            # STDP と シナプス恒常性 (Synaptic Scaling)
            # 重みの合計が一定になるように正規化し、特定シナプスの暴走を防ぐ
            for h in fired_h:
                sum_w_up = sum(self.w_up[pre].get(h, 0) for pre in range(self.d_model))
                for pre in range(self.d_model):
                    if self.trace_pre[pre] > 0 and h in self.w_up[pre]:
                        # LTP (強化)
                        self.w_up[pre][h] += self.lr
                    elif h in self.w_up[pre]:
                        # LTD (減衰) + 恒常性維持
                        self.w_up[pre][h] -= self.lr * 0.1 * (sum_w_up / 5.0)
                    if h in self.w_up[pre]:
                        self.w_up[pre][h] = np.clip(self.w_up[pre][h], 0.0, 3.0)
                        
            for o in fired_o:
                sum_w_down = sum(self.w_down[h].get(o, 0) for h in fired_h)
                for h in fired_h:
                    if o in self.w_down[h]:
                        self.w_down[h][o] += self.lr
                    else:
                        if o in self.w_down[h]:
                            self.w_down[h][o] -= self.lr * 0.1 * (sum_w_down / 5.0)
                    if o in self.w_down[h]:
                        self.w_down[h][o] = np.clip(self.w_down[h][o], 0.0, 3.0)

            # トレース更新
            self.trace_pre.fill(0)
            for i in valid_in:
                self.trace_pre[i] = 1.0
                
            return fired_o

        else: 
            # --- 推論・生成フェーズ ---
            v_h = np.zeros(self.d_ff)
            for i in valid_in:
                for tgt, w in self.w_up[i].items():
                    v_h[tgt] += w
                    
            fired_h = np.where(v_h >= 0.8)[0].tolist() 
            k_h = max(1, int(self.d_ff * 0.05))
            if len(fired_h) > k_h:
                fired_h = np.argsort(v_h)[-k_h:].tolist()
                
            v_o = np.zeros(self.d_model)
            for h in fired_h:
                for tgt, w in self.w_down[h].items():
                    # 厳格な不応期: 直前に発火したニューロンや現在入力されている文字は発火させない
                    if self.refractory_o[tgt] <= 0 and tgt not in valid_in:
                        v_o[tgt] += w
                    
            fired_o = np.where(v_o >= 0.8)[0].tolist()
            k_o = max(1, int(self.d_model * 0.05))
            if len(fired_o) > k_o:
                fired_o = np.argsort(v_o)[-k_o:].tolist()
            
            # 発火したニューロンを不応期に入れる (2ステップ休む)
            if fired_o:
                self.refractory_o[fired_o] = 2
                
            return fired_o

    def reset(self):
        self.trace_pre.fill(0)
        self.refractory_o.fill(0)

class PlasticTransformerBlock:
    def __init__(self, d_model: int, num_heads: int, memory_size: int = 50):
        self.d_model = d_model
        self.attention = SpikeAttention(d_model, d_model, memory_size=memory_size, num_heads=num_heads)
        self.ffn = PlasticSpikeFFN(d_model, d_model * 4)
        self.pos_encoder = SpikePositionalEncoding(d_model)

    def compute(self, input_spikes: List[int], pos: int, learning: bool = False) -> List[int]:
        pos_spikes = self.pos_encoder.get_spikes(pos)
        x = list(set(input_spikes) | set(pos_spikes))
        attn_out = self.attention.compute(x)
        x_ctx = list(set(x) | set(attn_out))
        return self.ffn.compute(x_ctx, learning=learning)

    def generate_next(self, current_spikes: List[int], pos: int) -> List[int]:
        eval_pos = pos - 1 if pos > 0 else 0
        pos_spikes = self.pos_encoder.get_spikes(eval_pos)
        valid_in = [s for s in current_spikes if s < self.d_model]
        x = list(set(valid_in) | set(pos_spikes))
        attn_out = self.attention.compute(x)
        x_ctx = list(set(x) | set(attn_out))
        return self.ffn.compute(x_ctx, learning=False)

    def reset(self):
        self.attention.reset()
        self.ffn.reset()