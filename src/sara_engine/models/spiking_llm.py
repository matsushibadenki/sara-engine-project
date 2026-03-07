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
from typing import Any, Dict, List, Set, Tuple, Optional

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

    def generate(
        self,
        prompt: Optional[str] = None,
        prompt_tokens: Optional[List[int]] = None,
        max_new_tokens: int = 50,
        top_k: int = 5,
        temperature: float = 0.3,
        repetition_penalty: float = 1.2,
        **kwargs: Any,
    ) -> str | List[int]:

        return_string = False
        if prompt is not None:
            prompt_words = self.tokenizer.split_text(prompt)
            prompt_tokens = []
            unk_word = None
            unk_id = self.tokenizer.vocab.get("<unk>", 1)

            # 💡 修正点: 入力に未知語が含まれている場合、無視せずに検知して即時停止する
            for w in prompt_words:
                if w not in self.tokenizer.vocab or self.tokenizer.vocab[w] == unk_id:
                    if w.strip():  # 空白や改行以外の実質的な未知語
                        unk_word = w
                        break
                prompt_tokens.append(self.tokenizer.vocab.get(w, unk_id))

            return_string = True

            if unk_word:
                return f"（未知の単語「{unk_word}」が含まれているため、文脈を認識できません）" if return_string else []

            if not prompt_tokens and prompt.strip():
                return "（未知の入力スパイクです。記憶にありません）" if return_string else []

        elif prompt_tokens is None:
            prompt_tokens = list(kwargs.get("input_spikes", []))

        max_new_tokens = int(kwargs.get("max_length", max_new_tokens))
        generated_sequence: List[int] = []

        if not prompt_tokens:
            return "" if return_string else generated_sequence

        exempt_ids: Set[int] = {
            self.tokenizer.vocab[c] for c in self._PENALTY_EXEMPT_CHARS
            if c in self.tokenizer.vocab
        }

        end_of_sentence_ids: Set[int] = {
            self.tokenizer.vocab[c] for c in ('。', '！', '？', '\n')
            if c in self.tokenizer.vocab
        }

        # 💡 修正点: 知っている単語同士でも、コーパス内で繋がったことがない文脈なら推論を停止する
        if len(prompt_tokens) >= 2:
            known_transitions = 0
            for i in range(len(prompt_tokens) - 1):
                pre = prompt_tokens[i]
                post = prompt_tokens[i+1]
                if 1 in self.pretrained_synapses and pre in self.pretrained_synapses[1] and post in self.pretrained_synapses[1][pre]:
                    known_transitions += 1

            if known_transitions == 0:
                return "（知識ネットワークにこの文脈に続く概念が見つかりません。別の言葉で試してください）" if return_string else []

        context_tokens: List[int] = list(prompt_tokens)
        recent_window = max(self.context_window, 20)

        fatigue = AdaptiveThresholdHomeostasis(
            target_rate=1.0 / max(4.0, float(recent_window)),
            adaptation_rate=0.35,
            decay=0.8,
            min_threshold=0.0,
            max_threshold=4.0,
            global_weight=0.0,
        )

        for _t in range(max_new_tokens):
            fatigue.update([], population_size=self.vocab_size)
            scores: Dict[int, float] = defaultdict(float)
            votes: Dict[int, List[int]] = defaultdict(list)

            active_recent = context_tokens[-self.context_window:]

            # === 1. Direct Wiring (遅延シナプス) ===
            for reversed_idx in range(len(active_recent)):
                pre_token = active_recent[-(reversed_idx + 1)]
                delay = reversed_idx + 1

                if delay in self.pretrained_synapses and pre_token in self.pretrained_synapses[delay]:
                    context_factor = 0.75 ** (delay - 1)
                    for post_token, weight in self.pretrained_synapses[delay][pre_token].items():
                        if post_token < self.vocab_size:
                            scores[post_token] += weight * context_factor
                            votes[post_token].append(delay)

            # === 2. 樹状突起での同時発火検出 (Coincidence Detection) ===
            for tok_id in list(scores.keys()):
                hits = len(votes[tok_id])
                if hits > 1:
                    scores[tok_id] *= (3.0 ** (hits - 1))

            # 💡 修正点: 最初の生成ステップで、複数文脈からの支持(hits>=2)が全く得られない場合は沈黙する
            if not generated_sequence and len(prompt_tokens) >= 2:
                valid_hits = [len(votes[t]) for t in scores.keys(
                ) if t not in exempt_ids and t not in end_of_sentence_ids]
                max_hits = max(valid_hits) if valid_hits else 0
                if max_hits < 2:
                    return "（この文脈に続く明確な知識ネットワークが形成されていません）" if return_string else []

            # === 3. SDR Fuzzy Recall (エピソード記憶 / STDP) ===
            sdr_context = context_tokens[-min(5, len(context_tokens)):]
            current_spikes = self._encode_to_sdr(sdr_context)
            sdr_k = self._sdr_key(current_spikes)

            recalled, confidence = self.recall(sdr_k, threshold=0.75)
            if recalled is not None:
                for tok_id, count in recalled.items():
                    if tok_id < self.vocab_size:
                        scores[tok_id] += count * confidence * 2.0

            if not scores:
                break

            # 発話開始時の記号ニューロン抑制
            if not generated_sequence:
                for pid in exempt_ids:
                    if pid in scores:
                        scores[pid] = 0.0

            # === 4. ALIF (適応的閾値) によるニューロンの疲労ペナルティ ===
            for tok_id in list(scores.keys()):
                strength = 0.1 if tok_id in exempt_ids else 0.8
                scores[tok_id] = fatigue.modulate(tok_id, scores[tok_id], strength=strength)

            # N-gram 堂々巡りの強力なブロック
            for ngram_len in (2, 3, 4, 5, 6):
                if len(context_tokens) >= ngram_len:
                    tail = tuple(context_tokens[-ngram_len:])
                    for i in range(len(context_tokens) - ngram_len):
                        if tuple(context_tokens[i:i+ngram_len]) == tail:
                            if i + ngram_len < len(context_tokens):
                                repeat_tok = context_tokens[i+ngram_len]
                                if repeat_tok in scores:
                                    scores[repeat_tok] = 0.0

            # ASCIIペナルティ
            for tok_id in list(scores.keys()):
                if self._is_ascii_char(tok_id):
                    scores[tok_id] *= 0.1

            # Repetition Penalty
            recent_generated = context_tokens[-recent_window:]
            recent_counts: Dict[int, int] = defaultdict(int)
            for tok in recent_generated:
                recent_counts[tok] += 1

            for tok_id in list(scores.keys()):
                if tok_id in recent_counts and tok_id not in exempt_ids:
                    count = recent_counts[tok_id]
                    penalty = repetition_penalty ** count
                    scores[tok_id] /= penalty

            # === 5. 低スコアノイズの絶対的カット ===
            max_score = max(scores.values()) if scores else 0.0
            if max_score < 0.1:
                break

            threshold = max(max_score * 0.2, 0.2)
            for tok_id in list(scores.keys()):
                if scores[tok_id] < threshold:
                    del scores[tok_id]

            if not scores:
                break

            sorted_candidates = sorted(
                scores.items(), key=lambda x: x[1], reverse=True)
            top_k_candidates = sorted_candidates[:top_k]

            if not top_k_candidates:
                break

            if temperature > 0.0:
                max_score_topk = top_k_candidates[0][1]
                exp_scores = []
                for tok_id, score in top_k_candidates:
                    exp_val = math.exp(
                        (score - max_score_topk) / max(0.05, temperature))
                    exp_scores.append((tok_id, exp_val))

                sum_exp = sum(v for _, v in exp_scores)
                if sum_exp <= 0:
                    best_id = top_k_candidates[0][0]
                else:
                    r = random.random()
                    cumulative = 0.0
                    best_id = exp_scores[0][0]
                    for tok_id, exp_val in exp_scores:
                        cumulative += exp_val / sum_exp
                        if r <= cumulative:
                            best_id = tok_id
                            break
            else:
                best_id = top_k_candidates[0][0]

            fatigue.update([best_id], population_size=self.vocab_size)

            generated_sequence.append(best_id)
            context_tokens.append(best_id)

            # 文末記号が出現した時点で、文章の生成を綺麗に終了する
            if best_id in end_of_sentence_ids:
                break

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
