# filepath: src/sara_engine/models/spiking_llm.py
# title: スパイキング・大規模言語モデルブロック
# description: 発振（0発火と過剰発火の往復）を防ぐため、kWTA（K-Winner-Take-All）的な即時側抑制と、緩やかなホメオスタシス閾値調整を導入したSNN Transformerモデル。

import random
from src.sara_engine.core.spike_attention import SpikingSelfAttention

class SpikingLayerNorm:
    def __init__(self, sdr_size, base_threshold=1.0, target_active_ratio=0.05):
        self.sdr_size = sdr_size
        self.base_threshold = base_threshold
        self.thresholds = [base_threshold] * sdr_size
        self.target_spikes = max(1, int(sdr_size * target_active_ratio))

    def forward(self, input_potentials):
        # 1. ネットワーク全体の興奮度に基づくソフトなベースライン抑制
        active_potentials = [(i, p) for i, p in enumerate(input_potentials) if p > 0]
        
        if not active_potentials:
            # 入力がない場合は緩やかに閾値を回復
            for i in range(self.sdr_size):
                self.thresholds[i] = max(0.8, self.thresholds[i] - 0.005)
            return []

        # 平均電位の半分をベースライン抑制とする（強すぎる抑制で0発火になるのを避ける）
        avg_potential = sum(p for _, p in active_potentials) / len(active_potentials)
        global_inhibition = avg_potential * 0.5
        
        # 2. 実効電位の計算と発火判定
        spikes = []
        for i, p in enumerate(input_potentials):
            effective_p = p - global_inhibition
            if effective_p >= self.thresholds[i]:
                spikes.append(i)

        # 3. K-Winner-Take-All (kWTA) 的な強い即時側抑制
        # 目標数の2倍を超えて発火しそうな場合は、電位の高い順に足切りを行う（暴走の強制ストップ）
        max_allowed = self.target_spikes * 2
        if len(spikes) > max_allowed:
            spikes.sort(key=lambda x: input_potentials[x], reverse=True)
            spikes = spikes[:max_allowed]

        # 4. ホメオスタシス（恒常性）による緩やかな長期閾値調整
        # 発振を防ぐため、調整幅(adjustment_rate)を極めて小さくする
        adjustment_rate = 0.01
        
        for i in range(self.sdr_size):
            if i in spikes:
                # 発火したニューロン：疲労（閾値上昇）
                self.thresholds[i] += adjustment_rate
            else:
                # 発火しなかったニューロン：回復（閾値低下）
                self.thresholds[i] -= adjustment_rate * 0.2
                
            # 閾値が極端な値にならないようクリッピング
            self.thresholds[i] = max(0.8, min(self.thresholds[i], self.base_threshold * 3.0))

        return spikes

class SpikingTransformerBlock:
    def __init__(self, sdr_size):
        self.sdr_size = sdr_size
        self.attention = SpikingSelfAttention(sdr_size)
        
        # 層ごとにベース閾値を微調整
        self.layer_norm1 = SpikingLayerNorm(sdr_size, base_threshold=1.2, target_active_ratio=0.05)
        self.layer_norm2 = SpikingLayerNorm(sdr_size, base_threshold=1.5, target_active_ratio=0.05)
        
        self.ffn_w = [{} for _ in range(sdr_size)]
        self._init_sparse_weights(self.ffn_w, density=0.1)

    def _init_sparse_weights(self, weights, density):
        for i in range(self.sdr_size):
            num_connections = int(self.sdr_size * density)
            targets = random.sample(range(self.sdr_size), num_connections)
            for t in targets:
                weights[i][t] = random.uniform(0.1, 0.5)

    def forward(self, input_spikes):
        # 1. Spiking Attention
        att_spikes = self.attention.forward(input_spikes)
        
        # 2. Spiking Residual Connection 1 & Norm
        res_potentials_1 = [0.0] * self.sdr_size
        for s in input_spikes + att_spikes:
            res_potentials_1[s] += 1.0
        norm1_spikes = self.layer_norm1.forward(res_potentials_1)
        
        # 3. FFN (Feed Forward Network)
        ffn_potentials = [0.0] * self.sdr_size
        for pre_id in norm1_spikes:
            for post_id, w in self.ffn_w[pre_id].items():
                ffn_potentials[post_id] += w
                
        # 4. Spiking Residual Connection 2 & Norm
        res_potentials_2 = list(ffn_potentials)
        for s in norm1_spikes:
            res_potentials_2[s] += 1.0
        output_spikes = self.layer_norm2.forward(res_potentials_2)
        
        return output_spikes