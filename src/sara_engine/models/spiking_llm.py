_FILE_INFO = {
    "//1": "ディレクトリパス: src/sara_engine/models/spiking_llm.py",
    "//2": "ファイルの日本語タイトル: スパイキング・大規模言語モデル（MoEとLIF長文脈統合版 + Fuzzy Recall）",
    "//3": "ファイルの目的や内容: 実モデルにPhase 3のCortical Columns (MoE) と LIF Attention を統合。恒常性を導入して記憶の上書きを防ぎ、さらにFuzzy Recall（曖昧検索）を実装。"
}

import math
import random
from typing import Any, Dict, List, Set, Tuple

from sara_engine.core.transformer import LIFSpikeAttention
from sara_engine.core.cortical_columns import SpikingCorticalColumns

class SpikingLayerNorm:
    def __init__(
        self,
        sdr_size: int,
        base_threshold: float = 1.0,
        target_active_ratio: float = 0.02,
    ):
        self.sdr_size = sdr_size
        self.base_threshold = base_threshold
        self.thresholds = [base_threshold] * sdr_size
        self.target_spikes = max(1, int(sdr_size * target_active_ratio))

    def forward(self, input_potentials: List[float]) -> List[int]:
        active_potentials = [(i, p)
                             for i, p in enumerate(input_potentials) if p > 0]

        if not active_potentials:
            for i in range(self.sdr_size):
                self.thresholds[i] = max(0.01, self.thresholds[i] - 0.02)
            return []

        active_ratio = len(active_potentials) / self.sdr_size
        avg_potential = sum(p for _, p in active_potentials) / \
            len(active_potentials)
        global_inhibition = avg_potential * active_ratio * 0.1

        spikes: List[int] = []
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
            sorted_active = sorted(
                active_potentials, key=lambda x: x[1], reverse=True)
            for idx, _p in sorted_active:
                if len(spikes) >= min_required:
                    break
                if idx not in spikes:
                    spikes.append(idx)

        adjustment_rate = 0.01
        for i in range(self.sdr_size):
            if i in spikes:
                self.thresholds[i] += adjustment_rate
            else:
                self.thresholds[i] -= adjustment_rate * 0.8
            self.thresholds[i] = max(
                0.01, min(self.thresholds[i], self.base_threshold * 3.0))

        return sorted(spikes)


class SpikingTransformerBlock:
    def __init__(self, sdr_size: int, enable_learning: bool = True):
        self.sdr_size = sdr_size
        self.enable_learning = enable_learning
        
        self.attention = LIFSpikeAttention(embed_dim=sdr_size, density=0.05, decay_rate=0.95)

        self.layer_norm1 = SpikingLayerNorm(
            sdr_size, base_threshold=1.0, target_active_ratio=0.02)
        self.layer_norm2 = SpikingLayerNorm(
            sdr_size, base_threshold=1.2, target_active_ratio=0.02)

        self.moe_ffn = SpikingCorticalColumns(embed_dim=sdr_size, num_experts=4, top_k=1, density=0.1)

    def reset_state(self) -> None:
        if hasattr(self.attention, "reset_state"):
            self.attention.reset_state()
        if hasattr(self.moe_ffn, "reset_state"):
            self.moe_ffn.reset_state()

    def forward(self, input_spikes: List[int], t_step: int = 0) -> List[int]:
        att_spikes = self.attention.forward(
            input_spikes, learning=self.enable_learning)

        res_potentials_1 = [0.0] * self.sdr_size
        for s in set(input_spikes).union(set(att_spikes)):
            res_potentials_1[s] += 1.0
        norm1_spikes = self.layer_norm1.forward(res_potentials_1)

        ffn_spikes = self.moe_ffn.forward(norm1_spikes, learning=self.enable_learning)

        res_potentials_2 = [0.0] * self.sdr_size
        for s in set(norm1_spikes).union(set(ffn_spikes)):
            res_potentials_2[s] += 1.0
        output_spikes = self.layer_norm2.forward(res_potentials_2)

        return output_spikes


class MultiLayerSpikingTransformer:
    def __init__(self, num_layers: int, sdr_size: int, enable_learning: bool = True):
        self.num_layers = num_layers
        self.sdr_size = sdr_size
        self.layers = [SpikingTransformerBlock(
            sdr_size, enable_learning) for _ in range(num_layers)]

    def reset_state(self) -> None:
        for layer in self.layers:
            layer.reset_state()

    def forward(self, input_spikes: List[int], t_step: int = 0) -> List[int]:
        current_spikes = input_spikes
        for layer in self.layers:
            current_spikes = layer.forward(current_spikes, t_step=t_step)
        return current_spikes


class SpikingLLM:
    """
    スパイキングLLM（長文脈・MoE統合版）。
    LIF（Leaky Integrate-and-Fire）によって、離れたトークンの情報を膜電位として保持し、
    Cortical Columnsによって最適なエキスパートに処理を分散する。
    """
    _SDR_BITS_PER_TOKEN: int = 32

    def __init__(
        self,
        num_layers: int = 2,
        sdr_size: int = 128,
        vocab_size: int = 10000,
        enable_learning: bool = True,
        **kwargs: Any,
    ):
        self.sdr_size: int = int(kwargs.get("d_model", sdr_size))
        self.vocab_size: int = vocab_size
        self.enable_learning: bool = enable_learning
        self.transformer = MultiLayerSpikingTransformer(
            num_layers, self.sdr_size, enable_learning)
        self.lm_head_w: List[Dict[int, float]] = [{} for _ in range(self.sdr_size)]
        self.global_t: int = 0

        self._sdr_cache: Dict[Tuple[int, ...], List[int]] = {}
        self._direct_map: Dict[Tuple[int, ...], Dict[int, float]] = {}

        self._init_lm_head_weights()

    def _init_lm_head_weights(self, density: float = 0.3) -> None:
        connections_per_neuron = max(1, int(self.vocab_size * density))
        for i in range(self.sdr_size):
            targets = random.sample(
                range(self.vocab_size), min(connections_per_neuron, self.vocab_size)
            )
            for t in targets:
                self.lm_head_w[i][t] = random.uniform(0.0, 0.05)

    def reset_state(self) -> None:
        self.transformer.reset_state()

    def forward(self, input_spikes: list[int], t_step: int = 0) -> tuple[list[float], list[int]]:
        sdr_k = self._sdr_key(input_spikes)
        vocab_potentials = [0.0] * self.vocab_size

        if sdr_k in self._direct_map:
            for tok_id, count in self._direct_map[sdr_k].items():
                if tok_id < self.vocab_size:
                    vocab_potentials[tok_id] = count * 100.0
            _, combined_spikes = self._internal_forward(input_spikes, t_step)
        else:
            lm_potentials, combined_spikes = self._internal_forward(input_spikes, t_step)
            for i in range(self.vocab_size):
                vocab_potentials[i] = lm_potentials[i]

        return vocab_potentials, combined_spikes
        
    def _internal_forward(self, input_spikes: list[int], t_step: int) -> tuple[list[float], list[int]]:
        hidden_spikes = self.transformer.forward(input_spikes, t_step=t_step)
        combined_spikes = list(set(input_spikes + hidden_spikes))

        vocab_potentials = [0.0] * self.vocab_size
        for pre_id in combined_spikes:
            if pre_id < len(self.lm_head_w):
                for post_id, w in self.lm_head_w[pre_id].items():
                    if post_id < self.vocab_size:
                        vocab_potentials[post_id] += w
        return vocab_potentials, combined_spikes

    def _encode_to_sdr(self, context_tokens: List[int]) -> List[int]:
        key = tuple(context_tokens)
        if key in self._sdr_cache:
            return self._sdr_cache[key]

        spikes: Set[int] = set()
        for i, tok in enumerate(context_tokens):
            pos = len(context_tokens) - i 
            for j in range(self._SDR_BITS_PER_TOKEN):
                spike_id = (tok * 104729 + pos * 7919 + j * 2741) % self.sdr_size
                spikes.add(spike_id)

        result = sorted(spikes)
        self._sdr_cache[key] = result
        return result

    def _sdr_key(self, sdr: List[int]) -> Tuple[int, ...]:
        return tuple(sdr)

    def learn_sequence(self, token_ids: List[int]) -> None:
        if not self.enable_learning or len(token_ids) < 2:
            return

        self.reset_state()
        context_window = 64

        context_tokens: List[int] = []
        for t in range(len(token_ids) - 1):
            current_token = token_ids[t]
            next_token = token_ids[t + 1]

            context_tokens.append(current_token)
            if len(context_tokens) > context_window:
                context_tokens.pop(0)

            input_spikes = self._encode_to_sdr(context_tokens)
            sdr_k = self._sdr_key(input_spikes)

            if sdr_k not in self._direct_map:
                self._direct_map[sdr_k] = {}
            dm = self._direct_map[sdr_k]

            dm[next_token] = dm.get(next_token, 0.0) + 5.0
            
            # 生物学的恒常性による正規化
            total_w = sum(dm.values())
            limit = 50.0
            if total_w > limit:
                decay = limit / total_w
                for post_id in list(dm.keys()):
                    dm[post_id] *= decay
                    if dm[post_id] < 0.1:
                        del dm[post_id]

            _, combined_spikes = self.forward(input_spikes, t_step=self.global_t)
            self.global_t += 1

            ltp_amount = 3.0

            for pre_id in combined_spikes:
                if pre_id < len(self.lm_head_w):
                    self.lm_head_w[pre_id][next_token] = self.lm_head_w[pre_id].get(next_token, 0.0) + ltp_amount

                    # シナプスの総負荷上限による恒常性維持
                    total_synapse_weight = sum(self.lm_head_w[pre_id].values())
                    capacity_limit = 30.0
                    
                    if total_synapse_weight > capacity_limit:
                        decay = capacity_limit / total_synapse_weight
                        for post_id in list(self.lm_head_w[pre_id].keys()):
                            self.lm_head_w[pre_id][post_id] *= decay
                            if self.lm_head_w[pre_id][post_id] < 0.1:
                                del self.lm_head_w[pre_id][post_id]

    def generate(
        self,
        prompt_tokens: List[int] | None = None,
        max_new_tokens: int = 5,
        top_k: int = 3,
        temperature: float = 0.8,
        **kwargs: Any,
    ) -> List[int]:
        if prompt_tokens is None:
            prompt_tokens = list(kwargs.get("input_spikes", []))
        max_new_tokens = int(kwargs.get("max_length", max_new_tokens))

        generated_sequence: List[int] = []
        if not prompt_tokens:
            return generated_sequence

        self.reset_state()
        context_window = 64

        context_tokens: List[int] = []
        for tok in prompt_tokens[:-1]:
            context_tokens.append(tok)
            if len(context_tokens) > context_window:
                context_tokens.pop(0)
            dummy_spikes = self._encode_to_sdr(context_tokens)
            self.forward(dummy_spikes, t_step=self.global_t)
            self.global_t += 1

        context_tokens = list(prompt_tokens[-context_window:])
        refractory_counters: Dict[int, int] = {rt: 1 for rt in prompt_tokens}

        for _t in range(max_new_tokens):
            current_spikes = self._encode_to_sdr(context_tokens)
            sdr_k = self._sdr_key(current_spikes)

            vocab_potentials = [0.0] * self.vocab_size

            direct_hit = sdr_k in self._direct_map
            if direct_hit:
                for tok_id, count in self._direct_map[sdr_k].items():
                    if tok_id < self.vocab_size:
                        vocab_potentials[tok_id] += count * 10.0
            else:
                lm_potentials, _ = self.forward(current_spikes, t_step=self.global_t)
                self.global_t += 1
                for i in range(self.vocab_size):
                    vocab_potentials[i] += lm_potentials[i]

            for vocab_id in range(self.vocab_size):
                if refractory_counters.get(vocab_id, 0) > 0:
                    vocab_potentials[vocab_id] *= 0.1

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

            for k in list(refractory_counters.keys()):
                refractory_counters[k] -= 1
                if refractory_counters[k] <= 0:
                    del refractory_counters[k]
            refractory_counters[best_vocab_id] = 1

            context_tokens.append(best_vocab_id)
            if len(context_tokens) > context_window:
                context_tokens.pop(0)

        return generated_sequence
        
    def load_memory(self, filepath: str) -> int:
        """
        ファイルから連想記憶（direct_map）を安全に読み込む
        """
        import msgpack
        import os
        if not os.path.exists(filepath):
            self._direct_map = {}
            return 0
            
        with open(filepath, "rb") as f:
            state = msgpack.unpack(f, raw=False)
            
        raw_map = state.get("direct_map", {})
        # 保存用に文字列化されているキーをタプル(int型)に復元
        self._direct_map = {eval(k): {int(tk): float(tv) for tk, tv in v.items()} for k, v in raw_map.items()}
        return len(self._direct_map)

    def recall(self, input_sdr_key: tuple, threshold: float = 0.75) -> tuple:
        """
        💡 SDR Overlap (Fuzzy Recall)
        行列演算を使わず、Pythonの積集合によって記憶の類似度を計算し連想を引き出す。
        戻り値: (連想されたトークン重み辞書, 類似度スコア)
        """
        if getattr(self, "_direct_map", None) is None:
            self._direct_map = {}
            
        # 1. 完全一致チェック (O(1)で超高速)
        if input_sdr_key in self._direct_map:
            return self._direct_map[input_sdr_key], 1.0
            
        # 2. ファジー検索 (SDR Overlap計算)
        input_set = set(input_sdr_key)
        input_len = len(input_set)
        if input_len == 0:
            return None, 0.0
            
        best_match_key = None
        best_overlap_ratio = 0.0
        
        # 記憶されている全パターンのキーをスキャン
        for stored_key in self._direct_map.keys():
            stored_set = set(stored_key)
            # 積集合の要素数（重なり具合）を計算
            overlap = len(input_set.intersection(stored_set))
            ratio = overlap / input_len
            
            if ratio > best_overlap_ratio:
                best_overlap_ratio = ratio
                best_match_key = stored_key
                
            # 閾値より高ければ早期リターンして高速化
            if best_overlap_ratio >= 0.95:
                break
                
        # 見つかった最高の類似度が閾値を超えていれば、その記憶を引き出す
        if best_overlap_ratio >= threshold and best_match_key is not None:
            return self._direct_map[best_match_key], best_overlap_ratio
            
        return None, best_overlap_ratio