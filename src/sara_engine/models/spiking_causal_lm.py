# src/sara_engine/models/spiking_causal_lm.py
# Spiking Causal Language Model v5.2 (Rust Hybrid Ultra-Optimized)
# Rustコアを統合し、汎用シナプスのSTDP学習と電位計算をオフロードして超高速化。Python側では collections.deque を導入し、履歴管理（リストへの挿入）を O(N) から O(1) へ最適化して生成ループのオーバーヘッドを劇的に削減しました。

from ..core.transformer import SpikeTransformerModel
from ..sara_rust_core import CausalSynapses
from typing import List, Dict, Optional
import math
import random
import json
import heapq
from collections import deque

_FILE_INFO = {
    "path":  "src/sara_engine/models/spiking_causal_lm.py",
    "title": "Spiking Causal Language Model v5.2 (Rust Hybrid Ultra-Optimized)",
    "description": "Rustコアを統合し、汎用シナプスのSTDP学習と電位計算をオフロードして超高速化。Python側では collections.deque を導入し履歴管理をO(1)化。"
}

_SYMBOL_CHARS = frozenset(
    "。、！？.,!?…・「」『』()（） 　\n\t"
)

_EXCLUDE_WORDS = frozenset([
    "User", "Assistant", ":", "はい", "いいえ",
])


class SpikingCausalLM:
    def __init__(self, vocab_size: int, embed_dim: int = 1024, hidden_dim: int = 2048,
                 num_layers: int = 2, use_lif: bool = True):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_delay = 10

        self.transformer = SpikeTransformerModel(
            num_layers=num_layers,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            use_lif=use_lif
        )
        self.token_to_sdr: Dict[int, List[int]] = {}

        # --- Rust Core への差し替え ---
        self.rust_synapses = CausalSynapses(max_delay=self.max_delay)

        self.qa_weights: Dict[int, Dict[int, float]] = {}

        self._id_to_token: Dict[int, str] = {}
        self._vocab_registered = False
        self._hub_penalty_cache: Dict[int, float] = {}

    def _get_sdr_for_token(self, token_id: int, sparsity: float = 0.05) -> List[int]:
        if token_id not in self.token_to_sdr:
            random.seed(token_id)
            num_spikes = max(1, int(self.embed_dim * sparsity))
            self.token_to_sdr[token_id] = random.sample(
                range(self.embed_dim), num_spikes)
            random.seed()
        return self.token_to_sdr[token_id]

    def _context_key(self, prompt_ids: List[int]) -> int:
        return abs(hash(frozenset(prompt_ids)))

    def register_vocab(self, id_to_token: Dict[int, str]) -> None:
        self._id_to_token = id_to_token
        self._vocab_registered = True
        sample = list(id_to_token.items())[:5]
        print(f"  [register_vocab] registered {len(id_to_token)} entries. sample={sample}")

    def _is_excluded(self, token_id: int) -> bool:
        if not self._vocab_registered:
            return False

        raw_text = self._id_to_token.get(token_id, "")
        clean = raw_text.strip().lstrip("▁Ġ")

        if not clean:
            return True
        if len(clean) == 1 and clean in _SYMBOL_CHARS:
            return True
        if clean in _EXCLUDE_WORDS:
            return True

        if "終" in clean or "＜" in clean or "＞" in clean:
            return True

        return False

    def reset_context(self) -> None:
        self.transformer.reset_state()

    def train_step(self, sequence: List[int], learning_rate: float = 0.5) -> None:
        self.reset_context()
        # dequeを使用してO(1)の先頭挿入を実現
        spike_history: deque = deque(maxlen=self.max_delay + 1)

        for i in range(len(sequence) - 1):
            curr = sequence[i]
            nxt = sequence[i + 1]
            input_spikes = self._get_sdr_for_token(curr)
            output_spikes = self.transformer.forward(input_spikes, learning=False)
            offset = [s + self.embed_dim for s in output_spikes]
            combined = input_spikes + offset

            spike_history.appendleft(combined)

            # --- RustコアにSTDP学習を完全オフロード ---
            self.rust_synapses.train_step(list(spike_history), nxt, learning_rate)

    def supervised_qa_train(
        self,
        question_ids:  List[int],
        answer_ids:    List[int],
        learning_rate: float = 2.0,
        max_qa_tokens: int = 15,
    ) -> None:
        if not question_ids or not answer_ids:
            return

        ctx_key = self._context_key(question_ids)
        if ctx_key not in self.qa_weights:
            self.qa_weights[ctx_key] = {}

        registered = 0
        for tok_id in answer_ids:
            if registered >= max_qa_tokens:
                break
            if self._is_excluded(tok_id):
                continue

            decay = max(0.4, 1.0 - registered * 0.05)
            alpha = min(0.5, learning_rate * 0.2)

            old_w = self.qa_weights[ctx_key].get(tok_id, 0.0)
            self.qa_weights[ctx_key][tok_id] = old_w + alpha * (decay - old_w)
            registered += 1

    def _get_qa_bonus(self, question_ids: List[int]) -> Dict[int, float]:
        ctx_key = self._context_key(question_ids)
        return self.qa_weights.get(ctx_key, {})

    def _get_hub_penalty(self, t_id: int, token_fan_in: Dict[int, float]) -> float:
        if t_id not in self._hub_penalty_cache:
            hub = token_fan_in.get(t_id, 1.0)
            self._hub_penalty_cache[t_id] = math.pow(hub, 0.2) if hub > 1.0 else 1.0
        return self._hub_penalty_cache[t_id]

    def generate(
        self,
        prompt_tokens:  List[int],
        max_new_tokens: int = 25,
        temperature:    float = 0.01,
        question_ids:   Optional[List[int]] = None,
        stop_token_ids: Optional[List[int]] = None,
        repetition_penalty: float = 0.01,
        repetition_window: int = 3,
    ) -> List[int]:
        self.reset_context()
        generated:     List[int] = []
        # 生成時もdequeで履歴管理をO(1)化
        spike_history: deque = deque(maxlen=self.max_delay + 1)

        qa_ids = question_ids if question_ids is not None else prompt_tokens
        qa_bonus = self._get_qa_bonus(qa_ids)

        QA_SCALE = 500.0
        qa_scale_current = QA_SCALE

        token_fan_in = self.rust_synapses.get_token_fan_in()

        for t in prompt_tokens[:-1]:
            input_spikes = self._get_sdr_for_token(t)
            output_spikes = self.transformer.forward(input_spikes, learning=False)
            combined = input_spikes + [s + self.embed_dim for s in output_spikes]
            spike_history.appendleft(combined)

        current_token = prompt_tokens[-1]

        for _ in range(max_new_tokens):
            input_spikes = self._get_sdr_for_token(current_token)
            output_spikes = self.transformer.forward(input_spikes, learning=False)
            combined = input_spikes + [s + self.embed_dim for s in output_spikes]
            spike_history.appendleft(combined)

            token_potentials = self.rust_synapses.calculate_potentials(list(spike_history))

            if not token_potentials:
                break

            forbidden = set(generated[-repetition_window:] + [current_token]) if repetition_window > 0 else set()

            for t_id, pot in token_potentials.items():
                if qa_scale_current > 0 and t_id in qa_bonus:
                    pot += qa_bonus[t_id] * qa_scale_current

                hub_pen = self._get_hub_penalty(t_id, token_fan_in)
                if hub_pen > 1.0:
                    pot /= hub_pen

                if repetition_penalty < 1.0 and t_id in forbidden:
                    pot *= repetition_penalty

                token_potentials[t_id] = pot

            top_k_count = min(5, len(token_potentials))
            if top_k_count == 0:
                break
                
            candidates = heapq.nlargest(top_k_count, token_potentials.items(), key=lambda x: x[1])

            if temperature <= 0.05:
                next_token = candidates[0][0]
            else:
                inv_temp = 1.0 / temperature
                max_p = candidates[0][1]
                exp_pots = [(t_id, math.exp(max(-20.0, (p - max_p) * inv_temp))) for t_id, p in candidates]
                total_pot = sum(ep for _, ep in exp_pots)
                
                r = random.uniform(0, total_pot)
                next_token = candidates[0][0]
                cumulative = 0.0
                for t_id, ep in exp_pots:
                    cumulative += ep
                    if r <= cumulative:
                        next_token = t_id
                        break

            if token_potentials.get(next_token, 0) < 0.001:
                break

            generated.append(next_token)

            if stop_token_ids and next_token in stop_token_ids:
                break

            current_token = next_token

            gen_len = len(generated)
            qa_scale_current = QA_SCALE * max(0.0, 1.0 - gen_len * 0.02)

        return generated

    def save_pretrained(self, filepath: str) -> None:
        print("[WARN] Rust Hybrid v5.2 では、汎用シナプスのシリアライズは未実装です。QA専用シナプスのみ保存します。")
        serializable_qa: Dict[str, Dict[str, float]] = {
            str(ctx_k): {str(tok_id): w for tok_id, w in tok_dict.items()}
            for ctx_k, tok_dict in self.qa_weights.items()
        }

        state = {
            "vocab_size":  self.vocab_size,
            "embed_dim":   self.embed_dim,
            "max_delay":   self.max_delay,
            "transformer": self.transformer.state_dict(),
            "qa_weights":  serializable_qa,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load_pretrained(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)

        self.vocab_size = state["vocab_size"]
        self.embed_dim = state["embed_dim"]
        self.max_delay = state.get("max_delay", 10)
        self.transformer.load_state_dict(state["transformer"])

        self.qa_weights = {}
        for ctx_k_str, tok_dict in state.get("qa_weights", {}).items():
            ctx_k = int(ctx_k_str)
            self.qa_weights[ctx_k] = {
                int(tok_id): float(w) for tok_id, w in tok_dict.items()
            }