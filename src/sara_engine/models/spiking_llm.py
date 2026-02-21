# filepath: src/sara_engine/models/spiking_llm.py
"""
{
    "title": "スパイキング・大規模言語モデルブロック（多層・STDP対応）",
    "description": "誤差逆伝播法と行列演算を排除し、純粋なPython実装による多層SNN Transformerおよび言語モデル。generateメソッドの引数（prompt_tokens, max_new_tokens）をデモスクリプトに適合するよう柔軟に修正し、系列からの生成に対応しました。"
}
"""
import math
import random
from sara_engine.core.spike_attention import SpikingSelfAttention

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


class SpikingLLM:
    def __init__(self, num_layers=2, sdr_size=128, vocab_size=10000, enable_learning=True, **kwargs):
        """
        行列演算を排除した、SDRベースの純粋なPython実装による大規模言語モデルのラッパー。
        """
        self.sdr_size = kwargs.get('d_model', sdr_size)
        self.vocab_size = vocab_size
        self.enable_learning = enable_learning
        
        # トランスフォーマーコア
        self.transformer = MultiLayerSpikingTransformer(num_layers, self.sdr_size, enable_learning)
        
        # Readout層（LM Head）: 隠れ層のSDR（スパイク）から語彙次元へのスパース結合
        self.lm_head_w = [{} for _ in range(self.sdr_size)]
        self._init_lm_head_weights(density=0.1)

    def _init_lm_head_weights(self, density):
        for i in range(self.sdr_size):
            num_connections = max(1, int(self.vocab_size * density))
            sample_size = min(num_connections, self.vocab_size)
            targets = random.sample(range(self.vocab_size), sample_size)
            for t in targets:
                self.lm_head_w[i][t] = random.uniform(0.1, 0.8)

    def forward(self, input_spikes, t_step=0):
        # 1. ネットワーク本体（Transformer層）の順伝播処理
        hidden_spikes = self.transformer.forward(input_spikes, t_step=t_step)
        
        # 2. LM Head (語彙サイズへのマッピング)
        vocab_potentials = [0.0] * self.vocab_size
        for pre_id in hidden_spikes:
            if pre_id < len(self.lm_head_w):
                for post_id, w in self.lm_head_w[pre_id].items():
                    vocab_potentials[post_id] += w
                    
        return vocab_potentials

    def learn_sequence(self, token_ids):
        """
        デモスクリプトからの呼び出しに対応。
        時系列のトークンIDリストを受け取り、局所的なヘッブ則を用いて次トークン予測の学習を行います。
        """
        if not self.enable_learning or len(token_ids) < 2:
            return

        for t in range(len(token_ids) - 1):
            current_token = token_ids[t]
            next_token = token_ids[t + 1]

            # 入力スパイク化 (generate時と同じ簡易エンコーディング)
            input_spikes = [current_token % self.sdr_size]

            # Transformerの順伝播と内部層（STDP）の駆動
            hidden_spikes = self.transformer.forward(input_spikes, t_step=t)

            # LM Headの局所的ヘッブ則（Hebbian Learning）による重み更新
            for pre_id in hidden_spikes:
                if pre_id < len(self.lm_head_w):
                    # 結合が存在しなければ初期化
                    if next_token not in self.lm_head_w[pre_id]:
                        self.lm_head_w[pre_id][next_token] = 0.1
                    
                    # LTP (Long-Term Potentiation): 正解への結合を強化
                    self.lm_head_w[pre_id][next_token] += 0.05
                    self.lm_head_w[pre_id][next_token] = min(1.0, self.lm_head_w[pre_id][next_token])
                    
                    # LTD (Long-Term Depression): 同じプレニューロンからの他の結合を僅かに減衰
                    for post_id in list(self.lm_head_w[pre_id].keys()):
                        if post_id != next_token:
                            self.lm_head_w[pre_id][post_id] -= 0.005
                            self.lm_head_w[pre_id][post_id] = max(0.0, self.lm_head_w[pre_id][post_id])

    def generate(self, prompt_tokens=None, max_new_tokens=5, threshold=0.1, **kwargs):
        """
        デモなどの利用に向けた推論・生成インターフェース。
        勝者総取り（Winner-takes-all）方式で最大ポテンシャルをトークンとして扱います。
        """
        # 後方互換性と引数名の揺れ（input_spikes, max_length等）を吸収
        if prompt_tokens is None:
            prompt_tokens = kwargs.get('input_spikes', [])
        
        # max_lengthが指定された場合はmax_new_tokensを上書き
        max_new_tokens = kwargs.get('max_length', max_new_tokens)

        generated_sequence = []
        
        if not prompt_tokens:
            return generated_sequence

        # プロンプトの最後のトークンを初期入力とする
        current_token = prompt_tokens[-1]
        current_spikes = [current_token % self.sdr_size]
        
        for t in range(max_new_tokens):
            vocab_potentials = self.forward(current_spikes, t_step=t)
            
            if not vocab_potentials:
                break
                
            max_potential = max(vocab_potentials)
            if max_potential < threshold:
                break
                
            best_vocab_id = vocab_potentials.index(max_potential)
            generated_sequence.append(best_vocab_id)
            
            # 次のステップのための擬似的なエンコーディング
            current_spikes = [best_vocab_id % self.sdr_size]
            
        return generated_sequence