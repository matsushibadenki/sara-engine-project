# {
#     "//": "ディレクトリパス: src/sara_engine/models/spiking_llm.py",
#     "//": "ファイルの日本語タイトル: スパイキング・大規模言語モデル（MoE, LIF, Direct Wiring統合版）",
#     "//": "ファイルの目的や内容: 実モデルにPhase 3のCortical Columns (MoE) と LIF Attention を統合。さらに「Direct Synaptic Wiring」を統合し、超高速な事前コーパス学習と、Fuzzy Recallによるリアルタイム適応（STDP）を同時に実現する。SaraTokenizerを導入し、単語・形態素レベルの推論に対応。"
# }

import os
import json
import math
import random
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Set, Tuple, Optional

from ..core.transformer import LIFSpikeAttention
from ..core.cortical_columns import SpikingCorticalColumns
from ..learning.homeostasis import AdaptiveThresholdHomeostasis
from ..utils.tokenizer import SaraTokenizer


class SpikingLayerNorm:
    def __init__(
        self,
        sdr_size: int,
        base_threshold: float = 1.0,
        target_active_ratio: float = 0.02,
    ):
        self.sdr_size = sdr_size
        self.base_threshold = base_threshold
        self.target_spikes = max(1, int(sdr_size * target_active_ratio))
        self.homeostasis = AdaptiveThresholdHomeostasis(
            target_rate=target_active_ratio,
            adaptation_rate=0.08,
            decay=0.96,
            min_threshold=0.01,
            max_threshold=base_threshold * 3.0,
            global_weight=0.35,
        )
        self.homeostasis.thresholds = {
            i: base_threshold for i in range(sdr_size)
        }

    def forward(self, input_potentials: List[float]) -> List[int]:
        active_potentials = [(i, p)
                             for i, p in enumerate(input_potentials) if p > 0]

        if not active_potentials:
            self.homeostasis.update([], population_size=self.sdr_size)
            return []

        active_ratio = len(active_potentials) / self.sdr_size
        avg_potential = sum(p for _, p in active_potentials) / \
            len(active_potentials)
        global_inhibition = avg_potential * active_ratio * 0.1

        spikes: List[int] = []
        for i, p in enumerate(input_potentials):
            effective_p = p - global_inhibition
            if effective_p >= self.homeostasis.get_threshold(i, self.base_threshold):
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

        self.homeostasis.update(spikes, population_size=self.sdr_size)

        return sorted(spikes)


class SpikingTransformerBlock:
    def __init__(self, sdr_size: int, enable_learning: bool = True):
        self.sdr_size = sdr_size
        self.enable_learning = enable_learning

        self.attention = LIFSpikeAttention(
            embed_dim=sdr_size, density=0.05, decay_rate=0.95)

        self.layer_norm1 = SpikingLayerNorm(
            sdr_size, base_threshold=1.0, target_active_ratio=0.02)
        self.layer_norm2 = SpikingLayerNorm(
            sdr_size, base_threshold=1.2, target_active_ratio=0.02)

        self.moe_ffn = SpikingCorticalColumns(
            embed_dim=sdr_size, num_experts=4, top_k=1, density=0.1)

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

        ffn_spikes = self.moe_ffn.forward(
            norm1_spikes, learning=self.enable_learning)

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
    _SDR_BITS_PER_TOKEN: int = 5

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

        self.tokenizer = SaraTokenizer(vocab_size=self.vocab_size)

        self._init_lm_head_weights()

    def _init_lm_head_weights(self, density: float = 0.3) -> None:
        self.lm_head_w = [{} for _ in range(self.sdr_size)]

    def encode_text(self, text: str) -> List[int]:
        words = self.tokenizer.split_text(text)
        return [self.tokenizer._add_token(w) for w in words]

    def decode_text(self, token_ids: List[int]) -> str:
        tokens = []
        for tid in token_ids:
            word = self.tokenizer.id_to_token.get(int(tid), "")
            if word not in self.tokenizer.special_tokens:
                tokens.append(word)
        return "".join(tokens)

    def reset_state(self) -> None:
        self.transformer.reset_state()

    def fit(self, text_data: str) -> 'SpikingLLM':
        tokens = self.encode_text(text_data)
        total_tokens = len(tokens)
        print(
            f"[SpikingLLM] Processing {total_tokens} tokens for delay-line direct wiring...")

        try:
            from .. import sara_rust_core
            print("[SpikingLLM] Utilizing Rust core for ultra-fast synaptic wiring...")
            rust_synapses = sara_rust_core.build_direct_synapses(
                tokens, self.context_window)

            self.pretrained_synapses.clear()
            for delay_key, pre_dict in rust_synapses.items():
                delay = int(delay_key)
                self.pretrained_synapses[delay] = {}
                for pre_key, post_dict in pre_dict.items():
                    pre = int(pre_key)
                    self.pretrained_synapses[delay][pre] = {}
                    for post_key, weight in post_dict.items():
                        self.pretrained_synapses[delay][pre][int(
                            post_key)] = float(weight)

            total_connections = sum(len(post_dict) for delay_dict in self.pretrained_synapses.values(
            ) for post_dict in delay_dict.values())
            print(
                f"[SpikingLLM] Training completed via Rust. Established {total_connections} delay-line connections.")

        except ImportError:
            print(
                "[WARNING] sara_rust_core not found. Falling back to standard Python implementation...")
            co_occurrence = defaultdict(
                lambda: defaultdict(lambda: defaultdict(float)))
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
                spike_id = (tok * 104729 + pos * 7919 +
                            j * 2741) % self.sdr_size
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

            _, combined_spikes = self.forward(
                input_spikes, t_step=self.global_t)
            self.global_t += 1

            ltp_amount = 1.0
            for pre_id in combined_spikes:
                if pre_id < len(self.lm_head_w):
                    self.lm_head_w[pre_id][next_token] = self.lm_head_w[pre_id].get(
                        next_token, 0.0) + ltp_amount
                    total_synapse_weight = sum(self.lm_head_w[pre_id].values())
                    capacity_limit = 10.0
                    if total_synapse_weight > capacity_limit:
                        decay = capacity_limit / total_synapse_weight
                        for post_id in list(self.lm_head_w[pre_id].keys()):
                            self.lm_head_w[pre_id][post_id] *= decay
                            if self.lm_head_w[pre_id][post_id] < 0.1:
                                del self.lm_head_w[pre_id][post_id]

    _PENALTY_EXEMPT_CHARS: Set[str] = {
        '。', '、', '\n', '　', ' ', '（', '）', '「', '」', '・',
        ')', '(', ',', '.', ':', ';',
    }

    def _is_ascii_char(self, tok_id: int) -> bool:
        char = self.tokenizer.id_to_token.get(int(tok_id), "")
        if not char:
            return False
        return all(ord(c) < 128 for c in char) and char not in ('\n', '。', '、')

    def _sync_vocab_size_with_tokenizer(self) -> None:
        tokenizer_size = max(
            int(getattr(self.tokenizer, "next_id", 0)),
            max(self.tokenizer.id_to_token.keys(), default=-1) + 1,
        )
        if tokenizer_size > self.vocab_size:
            self.vocab_size = tokenizer_size

    def _prepare_prompt_tokens(
        self,
        prompt: Optional[str] = None,
        prompt_tokens: Optional[List[int]] = None,
        input_spikes: Optional[List[int]] = None,
    ) -> Tuple[List[int], bool, Optional[str]]:
        if prompt is not None:
            self._sync_vocab_size_with_tokenizer()
            prompt_words = self.tokenizer.split_text(prompt)
            prepared_tokens: List[int] = []

            for word in prompt_words:
                if not word:
                    continue
                token_id = self.tokenizer.vocab.get(word)
                if token_id is not None:
                    prepared_tokens.append(token_id)
                    continue

                subwords = self.tokenizer._tokenize_word(word)
                known_subwords = [
                    self.tokenizer.vocab[subword]
                    for subword in subwords
                    if subword in self.tokenizer.vocab
                ]
                if len(known_subwords) == len(subwords) and known_subwords:
                    prepared_tokens.extend(known_subwords)
                    continue

                if word.strip():
                    dynamic_id = self.tokenizer._add_token(word)
                    if dynamic_id >= self.vocab_size:
                        self.vocab_size = dynamic_id + 1
                    prepared_tokens.append(dynamic_id)

            if not prepared_tokens and prompt.strip():
                return [], True, "（未知の入力スパイクです。記憶にありません）"

            if prepared_tokens and max(prepared_tokens) >= self.vocab_size:
                self.vocab_size = max(prepared_tokens) + 1

            return prepared_tokens, True, None

        if prompt_tokens is not None:
            return list(prompt_tokens), False, None

        if input_spikes is not None:
            return list(input_spikes), False, None

        return [], False, None

    def _candidate_metadata(
        self,
        tok_id: int,
        score: float,
        max_score: float,
        votes: Dict[int, List[int]],
    ) -> Dict[str, Any]:
        return {
            "token_id": int(tok_id),
            "token": self.tokenizer.id_to_token.get(int(tok_id), ""),
            "score": float(score),
            "confidence": float(score / max_score) if max_score > 0 else 0.0,
            "supporting_delays": sorted(votes.get(tok_id, [])),
            "support_count": len(votes.get(tok_id, [])),
        }

    def _score_next_tokens(
        self,
        context_tokens: List[int],
        repetition_penalty: float = 1.2,
        fatigue: Optional[AdaptiveThresholdHomeostasis] = None,
        suppress_initial_symbols: bool = True,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        if not context_tokens:
            return [], False
        self._sync_vocab_size_with_tokenizer()

        exempt_ids: Set[int] = {
            self.tokenizer.vocab[c] for c in self._PENALTY_EXEMPT_CHARS
            if c in self.tokenizer.vocab
        }
        end_of_sentence_ids: Set[int] = {
            self.tokenizer.vocab[c] for c in ('。', '！', '？', '\n')
            if c in self.tokenizer.vocab
        }
        recent_window = max(self.context_window, 20)
        if fatigue is None:
            fatigue = AdaptiveThresholdHomeostasis(
                target_rate=1.0 / max(4.0, float(recent_window)),
                adaptation_rate=0.35,
                decay=0.8,
                min_threshold=0.0,
                max_threshold=4.0,
                global_weight=0.0,
            )
        fatigue.update([], population_size=self.vocab_size)

        scores: Dict[int, float] = defaultdict(float)
        votes: Dict[int, List[int]] = defaultdict(list)
        active_recent = context_tokens[-self.context_window:]

        for reversed_idx in range(len(active_recent)):
            pre_token = active_recent[-(reversed_idx + 1)]
            delay = reversed_idx + 1

            if delay in self.pretrained_synapses and pre_token in self.pretrained_synapses[delay]:
                context_factor = 0.75 ** (delay - 1)
                for post_token, weight in self.pretrained_synapses[delay][pre_token].items():
                    if post_token < self.vocab_size:
                        scores[post_token] += weight * context_factor
                        votes[post_token].append(delay)

        for tok_id in list(scores.keys()):
            hits = len(votes[tok_id])
            if hits > 1:
                scores[tok_id] *= (3.0 ** (hits - 1))

        should_block = False
        if len(context_tokens) >= 2:
            valid_hits = [
                len(votes[tok_id]) for tok_id in scores.keys()
                if tok_id not in exempt_ids and tok_id not in end_of_sentence_ids
            ]
            should_block = (max(valid_hits) if valid_hits else 0) < 2

        sdr_context = context_tokens[-min(5, len(context_tokens)):]
        current_spikes = self._encode_to_sdr(sdr_context)
        recalled, confidence = self.recall(self._sdr_key(current_spikes), threshold=0.75)
        if recalled is not None:
            for tok_id, count in recalled.items():
                if tok_id < self.vocab_size:
                    scores[tok_id] += count * confidence * 2.0

        if suppress_initial_symbols:
            for pid in exempt_ids:
                if pid in scores:
                    scores[pid] = 0.0

        for tok_id in list(scores.keys()):
            strength = 0.1 if tok_id in exempt_ids else 0.8
            scores[tok_id] = fatigue.modulate(tok_id, scores[tok_id], strength=strength)

        for ngram_len in (2, 3, 4, 5, 6):
            if len(context_tokens) >= ngram_len:
                tail = tuple(context_tokens[-ngram_len:])
                for idx in range(len(context_tokens) - ngram_len):
                    if tuple(context_tokens[idx:idx + ngram_len]) == tail and idx + ngram_len < len(context_tokens):
                        repeat_tok = context_tokens[idx + ngram_len]
                        if repeat_tok in scores:
                            scores[repeat_tok] = 0.0

        recent_generated = context_tokens[-recent_window:]
        recent_counts: Dict[int, int] = defaultdict(int)
        for tok in recent_generated:
            recent_counts[tok] += 1

        for tok_id in list(scores.keys()):
            if self._is_ascii_char(tok_id):
                scores[tok_id] *= 0.1
            if tok_id in recent_counts and tok_id not in exempt_ids:
                scores[tok_id] /= repetition_penalty ** recent_counts[tok_id]
                scores[tok_id] /= (1.0 + max(0.0, presence_penalty))
                scores[tok_id] /= (1.0 + max(0.0, frequency_penalty) * recent_counts[tok_id])

        max_score = max(scores.values()) if scores else 0.0
        if max_score < 0.1:
            return [], should_block

        threshold = max(max_score * 0.2, 0.2)
        filtered = [
            self._candidate_metadata(tok_id, score, max_score, votes)
            for tok_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
            if score >= threshold
        ]
        return filtered, should_block

    def predict_next_tokens(
        self,
        prompt: Optional[str] = None,
        prompt_tokens: Optional[List[int]] = None,
        top_k: int = 5,
        repetition_penalty: float = 1.2,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        prepared_tokens, _return_string, error_message = self._prepare_prompt_tokens(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            input_spikes=kwargs.get("input_spikes"),
        )
        if error_message or not prepared_tokens:
            return []

        if len(prepared_tokens) >= 2:
            known_transitions = 0
            for idx in range(len(prepared_tokens) - 1):
                pre = prepared_tokens[idx]
                post = prepared_tokens[idx + 1]
                if 1 in self.pretrained_synapses and pre in self.pretrained_synapses[1] and post in self.pretrained_synapses[1][pre]:
                    known_transitions += 1
            if known_transitions == 0:
                return []

        candidates, should_block = self._score_next_tokens(
            prepared_tokens,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        return candidates[:max(1, int(top_k))]

    def generate_stream(
        self,
        prompt: Optional[str] = None,
        prompt_tokens: Optional[List[int]] = None,
        max_new_tokens: int = 50,
        top_k: int = 5,
        top_p: float = 1.0,
        temperature: float = 0.3,
        repetition_penalty: float = 1.2,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        stop_conditions: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        prepared_tokens, _return_string, error_message = self._prepare_prompt_tokens(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            input_spikes=kwargs.get("input_spikes"),
        )
        if error_message or not prepared_tokens:
            return

        context_tokens = list(prepared_tokens)
        generated_sequence: List[int] = []
        generated_text = ""
        end_of_sentence_ids: Set[int] = {
            self.tokenizer.vocab[c] for c in ('。', '！', '？', '\n')
            if c in self.tokenizer.vocab
        }
        if stop_conditions is None:
            stop_conditions = []
        recent_window = max(self.context_window, 20)
        fatigue = AdaptiveThresholdHomeostasis(
            target_rate=1.0 / max(4.0, float(recent_window)),
            adaptation_rate=0.35,
            decay=0.8,
            min_threshold=0.0,
            max_threshold=4.0,
            global_weight=0.0,
        )

        for step in range(max_new_tokens):
            candidates, should_block = self._score_next_tokens(
                context_tokens,
                repetition_penalty=repetition_penalty,
                fatigue=fatigue,
                suppress_initial_symbols=(step == 0),
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            if not candidates:
                break

            top_k_candidates = candidates[:max(1, int(top_k))]
            if temperature > 0.0 and len(top_k_candidates) > 1:
                max_score = top_k_candidates[0]["score"]
                exp_scores = []
                for candidate in top_k_candidates:
                    exp_val = math.exp(
                        (candidate["score"] - max_score) / max(0.05, temperature)
                    )
                    exp_scores.append((candidate, exp_val))

                ranked = sorted(exp_scores, key=lambda item: item[1], reverse=True)
                if 0.0 < top_p < 1.0:
                    total_mass = sum(val for _, val in ranked)
                    cutoff: List[Tuple[Dict[str, Any], float]] = []
                    cumulative = 0.0
                    for candidate, exp_val in ranked:
                        cutoff.append((candidate, exp_val))
                        cumulative += exp_val / total_mass if total_mass > 0 else 0.0
                        if cumulative >= top_p:
                            break
                    ranked = cutoff

                total = sum(val for _, val in ranked)
                chosen = ranked[0][0]
                if total > 0:
                    threshold = random.random()
                    cumulative = 0.0
                    for candidate, exp_val in ranked:
                        cumulative += exp_val / total
                        if threshold <= cumulative:
                            chosen = candidate
                            break
            else:
                chosen = top_k_candidates[0]

            token_id = int(chosen["token_id"])
            fatigue.update([token_id], population_size=self.vocab_size)
            generated_sequence.append(token_id)
            context_tokens.append(token_id)
            token_text = self.decode_text([token_id])
            generated_text += token_text

            yield {
                "token_id": token_id,
                "token": chosen["token"],
                "text": token_text,
                "score": chosen["score"],
                "confidence": chosen["confidence"],
                "step": step,
                "candidates": top_k_candidates,
                "generated_tokens": list(generated_sequence),
            }

            if any(generated_text.endswith(stop_text) for stop_text in stop_conditions):
                break
            if token_id in end_of_sentence_ids:
                break

    def generate(
        self,
        prompt: Optional[str] = None,
        prompt_tokens: Optional[List[int]] = None,
        max_new_tokens: int = 50,
        top_k: int = 5,
        top_p: float = 1.0,
        temperature: float = 0.3,
        repetition_penalty: float = 1.2,
        **kwargs: Any,
    ) -> str | List[int]:
        prompt_tokens, return_string, error_message = self._prepare_prompt_tokens(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            input_spikes=kwargs.get("input_spikes"),
        )
        if error_message:
            return error_message if return_string else []

        max_new_tokens = int(kwargs.get("max_length", max_new_tokens))
        if not prompt_tokens:
            return "" if return_string else []

        if len(prompt_tokens) >= 2:
            known_transitions = 0
            for i in range(len(prompt_tokens) - 1):
                pre = prompt_tokens[i]
                post = prompt_tokens[i+1]
                if 1 in self.pretrained_synapses and pre in self.pretrained_synapses[1] and post in self.pretrained_synapses[1][pre]:
                    known_transitions += 1

            if known_transitions == 0:
                return "（知識ネットワークにこの文脈に続く概念が見つかりません。別の言葉で試してください）" if return_string else []

        stream = list(self.generate_stream(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=float(kwargs.get("top_p", top_p)),
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            presence_penalty=float(kwargs.get("presence_penalty", 0.0)),
            frequency_penalty=float(kwargs.get("frequency_penalty", 0.0)),
            stop_conditions=kwargs.get("stop_conditions"),
            **kwargs,
        ))
        generated_sequence = [int(item["token_id"]) for item in stream]

        if kwargs.get("return_dict_in_generate"):
            result: Dict[str, Any] = {
                "sequences": self.decode_text(generated_sequence) if return_string else generated_sequence,
            }
            if kwargs.get("output_scores"):
                result["scores"] = [item["candidates"] for item in stream]
            if kwargs.get("output_tokens"):
                result["tokens"] = [item["token"] for item in stream]
            return result

        if return_string:
            return self.decode_text(generated_sequence)
        return generated_sequence

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, "spiking_llm_weights.json")

        self.tokenizer.model_path = os.path.join(
            save_directory, "sara_vocab.json")
        self.tokenizer.save()

        serializable_synapses = {}
        for delay, pre_dict in self.pretrained_synapses.items():
            serializable_synapses[str(delay)] = {}
            for pre, post_dict in pre_dict.items():
                serializable_synapses[str(delay)][str(pre)] = {
                    str(post): w for post, w in post_dict.items()}

        raw_direct_map = {str(k): {str(tk): float(tv) for tk, tv in v.items()}
                          for k, v in self._direct_map.items()}

        model_data = {
            "pretrained_synapses": serializable_synapses,
            "direct_map": raw_direct_map,
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

        vocab_path = os.path.join(load_directory, "sara_vocab.json")
        if os.path.exists(vocab_path):
            instance.tokenizer.model_path = vocab_path
            instance.tokenizer.load()
        else:
            instance.tokenizer.vocab = data.get("char_to_id", {})
            instance.tokenizer.id_to_token = {
                int(k): v for k, v in data.get("id_to_char", {}).items()}
            instance.tokenizer.next_id = data.get("next_id", 0)

        synapses = data.get("pretrained_synapses", {})
        for delay_str, pre_dict in synapses.items():
            delay = int(delay_str)
            instance.pretrained_synapses[delay] = {}
            for pre_str, post_dict in pre_dict.items():
                pre = int(pre_str)
                instance.pretrained_synapses[delay][pre] = {}
                for post_str, w in post_dict.items():
                    instance.pretrained_synapses[delay][pre][int(
                        post_str)] = float(w)

        def parse_tuple(s: str) -> Tuple[int, ...]:
            s = s.strip("()")
            if not s:
                return ()
            return tuple(int(x.strip()) for x in s.split(",") if x.strip())

        raw_direct_map = data.get("direct_map", {})
        instance._direct_map = {parse_tuple(k): {int(tk): float(
            tv) for tk, tv in v.items()} for k, v in raw_direct_map.items()}

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
