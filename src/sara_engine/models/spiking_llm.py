# src/sara_engine/models/spiking_llm.py
# 日本語タイトル: スパイキング・大規模言語モデル（MoE, LIF, Direct Wiring統合版）
# 目的: 実モデルにPhase 3のCortical Columns (MoE) と LIF Attention を統合。さらに「Direct Synaptic Wiring」を統合し、超高速な事前コーパス学習と、Fuzzy Recallによるリアルタイム適応（STDP）を同時に実現する。
# {
#     "//": "行列演算、誤差逆伝播法(BP)、GPU依存を完全に排除した純粋なSNN言語モデルです。",
#     "//": "AHP（後過分極）と強力なALIF、Softmax発火を導入し、ループを完全に破壊して文脈を前へ進めます。"
# }

import os
import json
import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Optional

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
    スパイキングLLM（Direct Wiring + MoE/LIF 統合版）。
    Rustコアによる超高速事前学習（Direct Wiring）の静的知識と、
    SDRやSTDPによる動的な会話記憶（Fuzzy Recall）を融合して推論を行う。
    """
    _SDR_BITS_PER_TOKEN: int = 32

    def __init__(
        self,
        num_layers: int = 2,
        sdr_size: int = 128,
        vocab_size: int = 65536,
        enable_learning: bool = True,
        context_window: int = 15,
        **kwargs: Any,
    ):
        self.sdr_size: int = int(kwargs.get("d_model", sdr_size))
        self.vocab_size: int = vocab_size
        self.enable_learning: bool = enable_learning
        self.context_window = context_window
        
        self.transformer = MultiLayerSpikingTransformer(
            num_layers, self.sdr_size, enable_learning)
        self.lm_head_w: List[Dict[int, float]] = []
        self.global_t: int = 0
        self._sdr_cache: Dict[Tuple[int, ...], List[int]] = {}
        self._direct_map: Dict[Tuple[int, ...], Dict[int, float]] = {}

        self.pretrained_synapses: Dict[int, Dict[int, Dict[int, float]]] = {}
        
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.next_id = 0

        self._init_lm_head_weights()

    def _init_lm_head_weights(self, density: float = 0.3) -> None:
        self.lm_head_w = [{} for _ in range(self.sdr_size)]

    def _get_or_add_id(self, char: str) -> int:
        if char not in self.char_to_id:
            self.char_to_id[char] = self.next_id
            self.id_to_char[self.next_id] = char
            self.next_id += 1
        return self.char_to_id[char]

    def encode_text(self, text: str) -> List[int]:
        return [self._get_or_add_id(c) for c in text]

    def decode_text(self, token_ids: List[int]) -> str:
        return "".join([self.id_to_char.get(tid, "") for tid in token_ids])

    def reset_state(self) -> None:
        self.transformer.reset_state()

    def fit(self, text_data: str) -> 'SpikingLLM':
        tokens = self.encode_text(text_data)
        total_tokens = len(tokens)
        print(f"[SpikingLLM] Processing {total_tokens} characters for delay-line direct wiring...")
        
        try:
            from sara_engine import sara_rust_core
            print("[SpikingLLM] Utilizing Rust core for ultra-fast synaptic wiring...")
            rust_synapses = sara_rust_core.build_direct_synapses(tokens, self.context_window)
            
            self.pretrained_synapses.clear()
            for delay_key, pre_dict in rust_synapses.items():
                delay = int(delay_key)
                self.pretrained_synapses[delay] = {}
                for pre_key, post_dict in pre_dict.items():
                    pre = int(pre_key)
                    self.pretrained_synapses[delay][pre] = {}
                    for post_key, weight in post_dict.items():
                        self.pretrained_synapses[delay][pre][int(post_key)] = float(weight)
                        
            total_connections = sum(len(post_dict) for delay_dict in self.pretrained_synapses.values() for post_dict in delay_dict.values())
            print(f"[SpikingLLM] Training completed via Rust. Established {total_connections} delay-line connections.")
            
        except ImportError:
            print("[WARNING] sara_rust_core not found. Falling back to standard Python implementation...")
            co_occurrence = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            unigram_counts = defaultdict(int)

            for i in range(total_tokens):
                current_token = tokens[i]
                unigram_counts[current_token] += 1
                end_idx = min(i + self.context_window + 1, total_tokens)
                for j in range(i + 1, end_idx):
                    delay = j - i
                    next_token = tokens[j]
                    co_occurrence[delay][current_token][next_token] += 1.0

            self.pretrained_synapses.clear()
            for delay, pre_dict in co_occurrence.items():
                self.pretrained_synapses[delay] = {}
                for pre_token, posts in pre_dict.items():
                    self.pretrained_synapses[delay][pre_token] = {}
                    pre_count = unigram_counts[pre_token]
                    for post_token, count in posts.items():
                        post_count = unigram_counts[post_token]
                        weight = count / math.sqrt(pre_count * post_count)
                        self.pretrained_synapses[delay][pre_token][post_token] = weight
            print("[SpikingLLM] Training completed via Python fallback.")
            
        return self

    def forward(self, input_spikes: list[int], t_step: int = 0) -> tuple[list[float], list[int]]:
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

            dm[next_token] = dm.get(next_token, 0.0) + 1.0
            
            total_w = sum(dm.values())
            limit = 10.0
            if total_w > limit:
                decay = limit / total_w
                for post_id in list(dm.keys()):
                    dm[post_id] *= decay
                    if dm[post_id] < 0.1:
                        del dm[post_id]

            _, combined_spikes = self.forward(input_spikes, t_step=self.global_t)
            self.global_t += 1

            ltp_amount = 1.0
            for pre_id in combined_spikes:
                if pre_id < len(self.lm_head_w):
                    self.lm_head_w[pre_id][next_token] = self.lm_head_w[pre_id].get(next_token, 0.0) + ltp_amount
                    total_synapse_weight = sum(self.lm_head_w[pre_id].values())
                    capacity_limit = 10.0
                    if total_synapse_weight > capacity_limit:
                        decay = capacity_limit / total_synapse_weight
                        for post_id in list(self.lm_head_w[pre_id].keys()):
                            self.lm_head_w[pre_id][post_id] *= decay
                            if self.lm_head_w[pre_id][post_id] < 0.1:
                                del self.lm_head_w[pre_id][post_id]

    def generate(
        self,
        prompt: Optional[str] = None,
        prompt_tokens: Optional[List[int]] = None,
        max_new_tokens: int = 50,
        top_k: int = 3,
        temperature: float = 0.8,
        **kwargs: Any,
    ) -> str | List[int]:
        return_string = False
        if prompt is not None:
            prompt_tokens = [self.char_to_id[c] for c in prompt if c in self.char_to_id]
            return_string = True
        elif prompt_tokens is None:
            prompt_tokens = list(kwargs.get("input_spikes", []))
            
        max_new_tokens = int(kwargs.get("max_length", max_new_tokens))
        generated_sequence: List[int] = []
        if not prompt_tokens:
            return "" if return_string else generated_sequence

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
        
        # --- 真の生物学的な出力層 (LIF + AHP + ALIF) ---
        potentials = defaultdict(float)
        thresholds = defaultdict(lambda: 1.0)
        last_spike_times = defaultdict(lambda: -100)

        # プロンプトのトークンに対してAHPと強力な疲労（ALIF）を適用し、直後のオウム返しを防ぐ
        for idx, tok in enumerate(prompt_tokens):
            last_spike_times[tok] = -len(prompt_tokens) + idx
            potentials[tok] = -15.0  # AHP: 強力なマイナス電位
            thresholds[tok] += 30.0 * (0.95 ** (len(prompt_tokens) - idx))

        for _t in range(max_new_tokens):
            current_spikes = self._encode_to_sdr(context_tokens)
            sdr_k = self._sdr_key(current_spikes)
            
            currents = defaultdict(float)

            # 1. Direct Wiring (事前学習) からの電流
            recent_spikes = context_tokens[-self.context_window:]
            for reversed_idx, pre_token in enumerate(reversed(recent_spikes)):
                delay = reversed_idx + 1
                if delay in self.pretrained_synapses and pre_token in self.pretrained_synapses[delay]:
                    for post_token, weight in self.pretrained_synapses[delay][pre_token].items():
                        if post_token < self.vocab_size:
                            temporal_decay = 0.85 ** (delay - 1)
                            # 電流スケールを適正値に抑える
                            currents[post_token] += weight * temporal_decay * 3.0

            # 2. SDR Fuzzy Recall (短期記憶)
            direct_hit = sdr_k in self._direct_map
            if direct_hit:
                for tok_id, count in self._direct_map[sdr_k].items():
                    if tok_id < self.vocab_size:
                        currents[tok_id] += count * 2.0

            # 3. Transformer / LIFMoE
            if not direct_hit:
                lm_potentials, _ = self.forward(current_spikes, t_step=self.global_t)
                self.global_t += 1
                for i in range(self.vocab_size):
                    if lm_potentials[i] > 0:
                        currents[i] += lm_potentials[i] * 0.5

            # --- 生物学的な積分発火 (Integrate and Fire) ---
            # 1. 膜電位の自然減衰 (Leak)
            for k in list(potentials.keys()):
                potentials[k] *= 0.8  # 記憶を少し長持ちさせるための緩やかな減衰
                if abs(potentials[k]) < 0.01:
                    del potentials[k]
                    
            # 2. 閾値のホメオスタシス回復 (ALIF)
            for k in list(thresholds.keys()):
                thresholds[k] = 1.0 + (thresholds[k] - 1.0) * 0.95  # ゆっくり回復
                if thresholds[k] <= 1.01:
                    del thresholds[k]

            # 3. 電流の注入
            for k, current_val in currents.items():
                if _t - last_spike_times[k] <= 2:  # 絶対不応期 (2ターン完全ブロック)
                    continue
                potentials[k] += current_val

            # 4. 発火判定 (Softmax-WTA)
            valid_candidates = []
            for k, pot in potentials.items():
                if _t - last_spike_times[k] <= 2:
                    continue
                # 膜電位から疲労（閾値）を引いた値を実質的なスコアとする
                score = pot - thresholds[k]
                valid_candidates.append((k, score))

            if not valid_candidates:
                # 完全に沈黙した場合は、自然発火（ランダムノイズ）により探索を強制する
                fallback_pool = [i for i in range(self.vocab_size) if _t - last_spike_times.get(i, -100) > 3]
                if fallback_pool:
                    best_vocab_id = random.choice(fallback_pool)
                    valid_candidates = [(best_vocab_id, 0.0)]
                else:
                    break

            valid_candidates.sort(key=lambda x: x[1], reverse=True)
            top_k_candidates = valid_candidates[:top_k]
            
            # Softmaxによる確率的選択（スコアがマイナスでも相対的な確率で選べる）
            max_score = top_k_candidates[0][1]
            if temperature > 0.0:
                top_scores = [math.exp((x[1] - max_score) / max(0.1, temperature)) for x in top_k_candidates]
                sum_s = sum(top_scores)
                probs = [s / sum_s for s in top_scores]
                
                r = random.random()
                cumulative = 0.0
                best_vocab_id = top_k_candidates[0][0]
                for idx, prob in zip([x[0] for x in top_k_candidates], probs):
                    cumulative += prob
                    if r <= cumulative:
                        best_vocab_id = idx
                        break
            else:
                best_vocab_id = top_k_candidates[0][0]

            generated_sequence.append(best_vocab_id)

            # 5. 勝者の状態リセット (AHP: 後過分極) と疲労（ALIF）の蓄積
            # 実質的に -45.0 (-15.0 - 30.0) のペナルティを課し、同じ文字の反復を完全に破壊する
            potentials[best_vocab_id] = -15.0  
            last_spike_times[best_vocab_id] = _t
            thresholds[best_vocab_id] += 30.0  

            context_tokens.append(best_vocab_id)
            if len(context_tokens) > context_window:
                context_tokens.pop(0)

        if return_string:
            return self.decode_text(generated_sequence)
        return generated_sequence

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, "spiking_llm_weights.json")
        
        serializable_synapses = {}
        for delay, pre_dict in self.pretrained_synapses.items():
            serializable_synapses[str(delay)] = {}
            for pre, post_dict in pre_dict.items():
                serializable_synapses[str(delay)][str(pre)] = {str(post): w for post, w in post_dict.items()}
                
        raw_direct_map = {str(k): {str(tk): float(tv) for tk, tv in v.items()} for k, v in self._direct_map.items()}

        model_data = {
            "pretrained_synapses": serializable_synapses,
            "direct_map": raw_direct_map,
            "char_to_id": self.char_to_id,
            "id_to_char": {str(k): v for k, v in self.id_to_char.items()},
            "next_id": self.next_id,
            "context_window": self.context_window,
            "vocab_size": self.vocab_size,
            "sdr_size": self.sdr_size,
        }
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        print(f"[SpikingLLM] モデルを保存しました: {model_path}")

    @classmethod
    def from_pretrained(cls, load_directory: str) -> 'SpikingLLM':
        model_path = os.path.join(load_directory, "spiking_llm_weights.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
            
        with open(model_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        instance = cls(
            sdr_size=data.get("sdr_size", 128),
            vocab_size=data.get("vocab_size", 65536),
            context_window=data.get("context_window", 15)
        )
        
        instance.char_to_id = data.get("char_to_id", {})
        instance.id_to_char = {int(k): v for k, v in data.get("id_to_char", {}).items()}
        instance.next_id = data.get("next_id", 0)
        
        synapses = data.get("pretrained_synapses", {})
        for delay_str, pre_dict in synapses.items():
            delay = int(delay_str)
            instance.pretrained_synapses[delay] = {}
            for pre_str, post_dict in pre_dict.items():
                pre = int(pre_str)
                instance.pretrained_synapses[delay][pre] = {}
                for post_str, w in post_dict.items():
                    instance.pretrained_synapses[delay][pre][int(post_str)] = float(w)
                    
        def parse_tuple(s: str) -> Tuple[int, ...]:
            s = s.strip("()")
            if not s: return ()
            return tuple(int(x.strip()) for x in s.split(",") if x.strip())
            
        raw_direct_map = data.get("direct_map", {})
        instance._direct_map = {parse_tuple(k): {int(tk): float(tv) for tk, tv in v.items()} for k, v in raw_direct_map.items()}
        
        print(f"[SpikingLLM] モデルをロードしました: {model_path}")
        return instance

    def recall(self, input_sdr_key: tuple, threshold: float = 0.75) -> tuple:
        if getattr(self, "_direct_map", None) is None:
            self._direct_map = {}
        if input_sdr_key in self._direct_map:
            return self._direct_map[input_sdr_key], 1.0
        input_set = set(input_sdr_key)
        input_len = len(input_set)
        if input_len == 0:
            return None, 0.0
        best_match_key = None
        best_overlap_ratio = 0.0
        for stored_key in self._direct_map.keys():
            stored_set = set(stored_key)
            overlap = len(input_set.intersection(stored_set))
            ratio = overlap / input_len
            if ratio > best_overlap_ratio:
                best_overlap_ratio = ratio
                best_match_key = stored_key
            if best_overlap_ratio >= 0.95:
                break
        if best_overlap_ratio >= threshold and best_match_key is not None:
            return self._direct_map[best_match_key], best_overlap_ratio
        return None, best_overlap_ratio