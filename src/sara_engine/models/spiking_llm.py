# filepath: src/sara_engine/models/spiking_llm.py
"""
{
    "title": "スパイキング・大規模言語モデルブロック（多層・STDP対応）",
    "description": "誤差逆伝播法と行列演算を排除し、純粋なPython実装による多層SNN Transformerブロック。各層の結合重みは、発火タイミングに依存する局所的なシナプス可塑性（STDP）により教師なしで学習・更新されます。"
}
"""
import math
import random
from src.sara_engine.core.spike_attention import SpikingSelfAttention

class SpikingLayerNorm:
    def __init__(self, sdr_size, base_threshold=1.0, target_active_ratio=0.05):
        self.sdr_size = sdr_size
        self.base_threshold = base_threshold
        self.thresholds = [base_threshold] * sdr_size
        self.target_spikes = max(1, int(sdr_size * target_active_ratio))

    def forward(self, input_potentials):
        active_potentials = [(i, p) for i, p in enumerate(input_potentials) if p > 0]
        
        if not active_potentials:
            for i in range(self.sdr_size):
                self.thresholds[i] = max(0.1, self.thresholds[i] - 0.005)
            return []

        active_ratio = len(active_potentials) / self.sdr_size
        avg_potential = sum(p for _, p in active_potentials) / len(active_potentials)
        global_inhibition = avg_potential * active_ratio * 0.5
        
        spikes = []
        for i, p in enumerate(input_potentials):
            effective_p = p - global_inhibition
            if effective_p >= self.thresholds[i]:
                spikes.append(i)

        max_allowed = self.target_spikes * 2
        min_required = max(1, int(self.target_spikes * 0.5))

        if len(spikes) > max_allowed:
            spikes.sort(key=lambda x: input_potentials[x], reverse=True)
            spikes = spikes[:max_allowed]
        elif len(spikes) < min_required and active_potentials:
            sorted_active = sorted(active_potentials, key=lambda x: x[1], reverse=True)
            for idx, p in sorted_active:
                if len(spikes) >= min_required:
                    break
                if idx not in spikes:
                    spikes.append(idx)

        adjustment_rate = 0.01
        for i in range(self.sdr_size):
            if i in spikes:
                self.thresholds[i] += adjustment_rate
            else:
                self.thresholds[i] -= adjustment_rate * 0.2
            self.thresholds[i] = max(0.1, min(self.thresholds[i], self.base_threshold * 3.0))

        return spikes


class STDP:
    def __init__(self, sdr_size, a_plus=0.01, a_minus=0.012, tau_plus=5.0, tau_minus=5.0, w_max=1.0, w_min=0.0):
        self.sdr_size = sdr_size
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_max = w_max
        self.w_min = w_min
        
        self.last_pre_times = [-1.0] * sdr_size
        self.last_post_times = [-1.0] * sdr_size

    def update_weights(self, t_step, pre_spikes, post_spikes, weights):
        # タイムスタンプの更新
        for pre_id in pre_spikes:
            self.last_pre_times[pre_id] = float(t_step)
        for post_id in post_spikes:
            self.last_post_times[post_id] = float(t_step)
            
        # 発火したポストニューロンをトリガーとして、プレニューロンとのシナプスを更新
        for post_id in post_spikes:
            t_post = self.last_post_times[post_id]
            for pre_id in range(self.sdr_size):
                if post_id in weights[pre_id]:
                    t_pre = self.last_pre_times[pre_id]
                    if t_pre >= 0:
                        delta_t = t_post - t_pre
                        
                        # LTP (Long-Term Potentiation): プレが先、または同時発火
                        if delta_t >= 0:
                            dw = self.a_plus * math.exp(-delta_t / self.tau_plus)
                        # LTD (Long-Term Depression): ポストが先
                        else:
                            dw = -self.a_minus * math.exp(delta_t / self.tau_minus)
                            
                        # 重みの更新とクリッピング
                        new_w = weights[pre_id][post_id] + dw
                        weights[pre_id][post_id] = max(self.w_min, min(self.w_max, new_w))


class SpikingTransformerBlock:
    def __init__(self, sdr_size, enable_learning=True):
        self.sdr_size = sdr_size
        self.enable_learning = enable_learning
        self.attention = SpikingSelfAttention(sdr_size)
        
        self.layer_norm1 = SpikingLayerNorm(sdr_size, base_threshold=1.2, target_active_ratio=0.05)
        self.layer_norm2 = SpikingLayerNorm(sdr_size, base_threshold=1.5, target_active_ratio=0.05)
        
        self.ffn_w = [{} for _ in range(sdr_size)]
        self._init_sparse_weights(self.ffn_w, density=0.1)
        
        if self.enable_learning:
            self.stdp = STDP(sdr_size)

    def _init_sparse_weights(self, weights, density):
        for i in range(self.sdr_size):
            num_connections = int(self.sdr_size * density)
            targets = random.sample(range(self.sdr_size), num_connections)
            for t in targets:
                weights[i][t] = random.uniform(0.1, 0.5)

    def forward(self, input_spikes, t_step=0):
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
        
        # 5. 学習（STDP）の適用
        if self.enable_learning:
            # FFNの入力（norm1_spikes）と出力（output_spikes）間で因果学習を行う
            self.stdp.update_weights(t_step, norm1_spikes, output_spikes, self.ffn_w)
            
        return output_spikes


class MultiLayerSpikingTransformer:
    def __init__(self, num_layers, sdr_size, enable_learning=True):
        self.num_layers = num_layers
        self.sdr_size = sdr_size
        self.layers = [SpikingTransformerBlock(sdr_size, enable_learning) for _ in range(num_layers)]

    def forward(self, input_spikes, t_step=0):
        current_spikes = input_spikes
        for layer_idx, layer in enumerate(self.layers):
            current_spikes = layer.forward(current_spikes, t_step=t_step)
        return current_spikes