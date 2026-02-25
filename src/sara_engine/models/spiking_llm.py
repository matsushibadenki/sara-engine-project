_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/spiking_llm.py",
    "//": "ファイルの日本語タイトル: スパイキング・大規模言語モデルブロック（多層・STDP・恒常性可塑性対応）",
    "//": "ファイルの目的や内容: Winner-Takes-Allによるモード崩壊を防ぐため、Homeostatic Plasticity（恒常性可塑性）と厳密な不応期を導入する。"
}

import math
import random
from sara_engine.core.spike_attention import SpikeSelfAttention

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
                self.thresholds[i] = max(0.01, self.thresholds[i] - 0.02)
            return []

        active_ratio = len(active_potentials) / self.sdr_size
        avg_potential = sum(p for _, p in active_potentials) / len(active_potentials)
        global_inhibition = avg_potential * active_ratio * 0.1
        
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
                if len(spikes) >= min_required: break
                if idx not in spikes: spikes.append(idx)

        adjustment_rate = 0.01
        for i in range(self.sdr_size):
            if i in spikes:
                self.thresholds[i] += adjustment_rate
            else:
                self.thresholds[i] -= adjustment_rate * 0.8
            self.thresholds[i] = max(0.01, min(self.thresholds[i], self.base_threshold * 3.0))

        return sorted(spikes)


class STDP:
    def __init__(self, sdr_size, a_plus=0.01, a_minus=0.005, tau_plus=5.0, tau_minus=5.0, w_max=1.0, w_min=0.0):
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
        for pre_id in pre_spikes:
            self.last_pre_times[pre_id] = float(t_step)
        for post_id in post_spikes:
            self.last_post_times[post_id] = float(t_step)
            
        for post_id in post_spikes:
            t_post = self.last_post_times[post_id]
            for pre_id in range(self.sdr_size):
                if post_id in weights[pre_id]:
                    t_pre = self.last_pre_times[pre_id]
                    if t_pre >= 0:
                        delta_t = t_post - t_pre
                        if delta_t >= 0:
                            dw = self.a_plus * math.exp(-delta_t / self.tau_plus)
                        else:
                            dw = -self.a_minus * math.exp(delta_t / self.tau_minus)
                        new_w = weights[pre_id][post_id] + dw
                        weights[pre_id][post_id] = max(self.w_min, min(self.w_max, new_w))


class SpikingTransformerBlock:
    def __init__(self, sdr_size, enable_learning=True):
        self.sdr_size = sdr_size
        self.enable_learning = enable_learning
        self.attention = SpikeSelfAttention(embed_dim=sdr_size, density=0.05)
        
        self.layer_norm1 = SpikingLayerNorm(sdr_size, base_threshold=1.0, target_active_ratio=0.1)
        self.layer_norm2 = SpikingLayerNorm(sdr_size, base_threshold=1.2, target_active_ratio=0.1)
        
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
        att_spikes = self.attention.forward(input_spikes, learning=self.enable_learning)
        
        res_potentials_1 = [0.0] * self.sdr_size
        for s in set(input_spikes).union(set(att_spikes)):
            res_potentials_1[s] += 1.0
        norm1_spikes = self.layer_norm1.forward(res_potentials_1)
        
        ffn_potentials = [0.0] * self.sdr_size
        for pre_id in norm1_spikes:
            for post_id, w in self.ffn_w[pre_id].items():
                ffn_potentials[post_id] += w
                
        res_potentials_2 = list(ffn_potentials)
        for s in norm1_spikes:
            res_potentials_2[s] += 1.0
        output_spikes = self.layer_norm2.forward(res_potentials_2)
        
        if self.enable_learning:
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
        self.sdr_size = kwargs.get('d_model', sdr_size)
        self.vocab_size = vocab_size
        self.enable_learning = enable_learning
        self.transformer = MultiLayerSpikingTransformer(num_layers, self.sdr_size, enable_learning)
        self.lm_head_w = [{} for _ in range(self.sdr_size)]
        self._init_lm_head_weights()

    def _init_lm_head_weights(self):
        pass

    def forward(self, input_spikes, t_step=0):
        hidden_spikes = self.transformer.forward(input_spikes, t_step=t_step)
        combined_spikes = list(set(input_spikes + hidden_spikes))
        
        vocab_potentials = [0.0] * self.vocab_size
        for pre_id in combined_spikes:
            if pre_id < len(self.lm_head_w):
                for post_id, w in self.lm_head_w[pre_id].items():
                    if post_id < self.vocab_size:
                        vocab_potentials[post_id] += w
                        
        return vocab_potentials, combined_spikes

    def learn_sequence(self, token_ids):
        if not self.enable_learning or len(token_ids) < 2: return
        
        context_tokens = []
        for t in range(len(token_ids) - 1):
            current_token = token_ids[t]
            next_token = token_ids[t + 1]
            
            context_tokens.append(current_token)
            if len(context_tokens) > 4:
                context_tokens.pop(0)
                
            input_spikes = []
            for i, tok in enumerate(context_tokens):
                spike_id = (tok * 37 + (len(context_tokens) - i) * 19) % self.sdr_size
                input_spikes.append(spike_id)
            
            input_spikes = list(set(input_spikes))
            
            vocab_potentials, combined_spikes = self.forward(input_spikes, t_step=t)
            
            # 競合学習と恒常性可塑性（Homeostatic Plasticity）
            for pre_id in combined_spikes:
                if pre_id < len(self.lm_head_w):
                    if next_token not in self.lm_head_w[pre_id]:
                        self.lm_head_w[pre_id][next_token] = 0.0
                    
                    if vocab_potentials[next_token] < 2.0:
                        self.lm_head_w[pre_id][next_token] += 0.2
                    
                    target_score = vocab_potentials[next_token]
                    for post_id in list(self.lm_head_w[pre_id].keys()):
                        if post_id != next_token and vocab_potentials[post_id] >= target_score - 0.5:
                            self.lm_head_w[pre_id][post_id] -= 0.1
                            if self.lm_head_w[pre_id][post_id] <= 0.0:
                                del self.lm_head_w[pre_id][post_id]
                    
                    # 重みの総和を1.0に厳密に正規化し、少数のトークンによるWinner-Takes-Allの暴走を防止
                    total_weight = sum(self.lm_head_w[pre_id].values())
                    if total_weight > 1.0:
                        for post_id in self.lm_head_w[pre_id]:
                            self.lm_head_w[pre_id][post_id] /= total_weight

    def generate(self, prompt_tokens=None, max_new_tokens=5, top_k=3, temperature=0.8, **kwargs):
        if prompt_tokens is None:
            prompt_tokens = kwargs.get('input_spikes', [])
        max_new_tokens = kwargs.get('max_length', max_new_tokens)

        generated_sequence = []
        if not prompt_tokens:
            return generated_sequence

        context_tokens = list(prompt_tokens[-4:])
        
        # ニューロンの不応期（Refractory Period）を管理
        refractory_counters = {}
        for rt in prompt_tokens:
            refractory_counters[rt] = 2
        
        for t in range(max_new_tokens):
            input_spikes = []
            for i, tok in enumerate(context_tokens):
                spike_id = (tok * 37 + (len(context_tokens) - i) * 19) % self.sdr_size
                input_spikes.append(spike_id)
            current_spikes = list(set(input_spikes))

            vocab_potentials, _ = self.forward(current_spikes, t_step=t)
            
            # 不応期中のニューロンの発火ポテンシャルを完全にゼロにして物理的に抑制
            for vocab_id in range(self.vocab_size):
                if refractory_counters.get(vocab_id, 0) > 0:
                    vocab_potentials[vocab_id] = 0.0
                
            valid_indices = [i for i, p in enumerate(vocab_potentials) if p > 0.0]
            
            if not valid_indices:
                break
                
            valid_indices.sort(key=lambda i: vocab_potentials[i], reverse=True)
            top_k_indices = valid_indices[:top_k]
            top_potentials = [vocab_potentials[i] for i in top_k_indices]
            
            if temperature != 1.0:
                top_potentials = [p ** (1.0 / temperature) for p in top_potentials]
            
            sum_p = sum(top_potentials)
            if sum_p <= 0.0:
                break
                
            probs = [p / sum_p for p in top_potentials]
            r = random.random()
            cumulative = 0.0
            best_vocab_id = top_k_indices[0]
            
            for idx, prob in zip(top_k_indices, probs):
                cumulative += prob
                if r <= cumulative:
                    best_vocab_id = idx
                    break
                    
            generated_sequence.append(best_vocab_id)
            
            # 全ての不応期カウンターを1進める
            for k in list(refractory_counters.keys()):
                refractory_counters[k] -= 1
                if refractory_counters[k] <= 0:
                    del refractory_counters[k]
            
            # 新たに発火したトークンニューロンに強い不応期を設定
            refractory_counters[best_vocab_id] = 3
            
            context_tokens.append(best_vocab_id)
            if len(context_tokens) > 4:
                context_tokens.pop(0)
            
        return generated_sequence