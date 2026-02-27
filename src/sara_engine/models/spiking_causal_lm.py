# src/sara_engine/models/spiking_causal_lm.py
# v4.1: _is_excluded のデバッグ強化版
#   qa_entries=0 の原因: get_vocab() のトークン文字列が
#   '▁SARAです' のようなBPE内部形式を返している可能性があり、
#   strip() 後も除外リストに一致しないはずだが何らかの理由で全除外されている。
#   → 除外判定を「1文字かつ記号」または「空文字」のみに絞り込む。
#   → register_vocab が呼ばれていない場合は除外処理をスキップする。

from sara_engine.core.transformer import SpikeTransformerModel
from typing import List, Dict, Optional
import math
import random
import json

_FILE_INFO = {
    "path":  "src/sara_engine/models/spiking_causal_lm.py",
    "title": "スパイキング因果言語モデル v4.1",
}

# 除外対象の記号文字セット（1文字トークンのみ対象）
_SYMBOL_CHARS = frozenset(
    "。、！？.,!?…・「」『』()（） 　\n\t"
)

# 除外対象の短い汎用語（完全一致）
_EXCLUDE_WORDS = frozenset([
    "User", "Assistant", ":", "はい", "いいえ",
])


class SpikingCausalLM:
    def __init__(self, vocab_size: int, embed_dim: int = 1024, hidden_dim: int = 2048,
                 num_layers: int = 2, use_lif: bool = True):
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.max_delay   = 10

        self.transformer = SpikeTransformerModel(
            num_layers=num_layers,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            use_lif=use_lif
        )
        self.token_to_sdr: Dict[int, List[int]] = {}

        # weights[delay][spike_id][token_id]  汎用シナプス
        self.weights: List[Dict[int, Dict[int, float]]] = [
            {} for _ in range(self.max_delay + 1)
        ]

        # qa_weights[context_key][token_id]  QA専用シナプス（Homeostasisなし）
        self.qa_weights: Dict[int, Dict[int, float]] = {}

        # id→テキスト の逆引きテーブル（register_vocab で登録）
        self._id_to_token: Dict[int, str] = {}
        self._vocab_registered = False

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
        """id→テキストの逆引きテーブルを登録する。supervised_qa_train で使用。"""
        self._id_to_token = id_to_token
        self._vocab_registered = True
        sample = list(id_to_token.items())[:5]
        print(f"  [register_vocab] registered {len(id_to_token)} entries. sample={sample}")

    def _is_excluded(self, token_id: int) -> bool:
        """
        除外判定:
          - register_vocab が呼ばれていない → 除外しない（False）
          - テキストが空 → 除外
          - テキストが1文字かつ記号文字 → 除外
          - テキストが除外語完全一致 → 除外
          - それ以外 → 除外しない（コンテンツトークン）
        """
        if not self._vocab_registered:
            return False

        raw_text = self._id_to_token.get(token_id, "")
        # BPE内部の先頭記号（▁ U+2581 や Ġ など）を除去
        clean = raw_text.strip().lstrip("▁Ġ")

        if clean == "":
            return True
        if len(clean) == 1 and clean in _SYMBOL_CHARS:
            return True
        if clean in _EXCLUDE_WORDS:
            return True
        return False

    def reset_context(self) -> None:
        self.transformer.reset_state()

    # ------------------------------------------------------------------
    # 汎用シーケンス学習
    # ------------------------------------------------------------------
    def train_step(self, sequence: List[int], learning_rate: float = 0.5) -> None:
        self.reset_context()
        spike_history: List[List[int]] = []

        for i in range(len(sequence) - 1):
            curr = sequence[i]
            nxt  = sequence[i + 1]
            input_spikes  = self._get_sdr_for_token(curr)
            output_spikes = self.transformer.forward(input_spikes, learning=False)
            offset        = [s + self.embed_dim for s in output_spikes]
            combined      = input_spikes + offset

            spike_history.insert(0, combined)
            if len(spike_history) > self.max_delay + 1:
                spike_history.pop()

            for delay, active_spikes in enumerate(spike_history):
                eff_lr = learning_rate * (1.0 - delay * 0.08)
                if eff_lr <= 0:
                    continue
                for s in active_spikes:
                    if s not in self.weights[delay]:
                        self.weights[delay][s] = {}
                    for existing_nxt in list(self.weights[delay][s].keys()):
                        if existing_nxt != nxt:
                            self.weights[delay][s][existing_nxt] *= (1.0 - eff_lr * 0.05)
                    old_w = self.weights[delay][s].get(nxt, 0.0)
                    self.weights[delay][s][nxt] = old_w + eff_lr * (1.0 - old_w)
                    total_w = sum(self.weights[delay][s].values())
                    if total_w > 3.0:
                        scale = 3.0 / total_w
                        for k in self.weights[delay][s]:
                            self.weights[delay][s][k] *= scale

    # ------------------------------------------------------------------
    # QA専用学習
    # ------------------------------------------------------------------
    def supervised_qa_train(
        self,
        question_ids:  List[int],
        answer_ids:    List[int],
        learning_rate: float = 2.0,
        max_qa_tokens: int   = 3,
    ) -> None:
        """
        回答の先頭 max_qa_tokens 件のコンテンツトークンを qa_weights に登録。
        _is_excluded=True のトークン（記号・汎用語）はスキップする。
        rank減衰: rank0=1.0, rank1=0.5, rank2=0.25
        """
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
            decay  = 1.0 / (2.0 ** registered)
            eff_lr = learning_rate * decay
            old_w  = self.qa_weights[ctx_key].get(tok_id, 0.0)
            self.qa_weights[ctx_key][tok_id] = old_w + eff_lr * (1.0 - old_w)
            registered += 1

    def _get_qa_bonus(self, question_ids: List[int]) -> Dict[int, float]:
        ctx_key = self._context_key(question_ids)
        return self.qa_weights.get(ctx_key, {})

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt_tokens:  List[int],
        max_new_tokens: int   = 15,
        temperature:    float = 1.0,
        question_ids:   Optional[List[int]] = None,
    ) -> List[int]:
        self.reset_context()
        generated:     List[int]       = []
        spike_history: List[List[int]] = []

        qa_ids   = question_ids if question_ids is not None else prompt_tokens
        qa_bonus = self._get_qa_bonus(qa_ids)

        # step0: 1000, step1: 300, step2以降: 0
        QA_SCALE         = 1000.0
        qa_scale_current = QA_SCALE

        token_fan_in: Dict[int, float] = {}
        for delay_dict in self.weights:
            for s_dict in delay_dict.values():
                for t, w in s_dict.items():
                    token_fan_in[t] = token_fan_in.get(t, 0.0) + w

        for t in prompt_tokens[:-1]:
            input_spikes  = self._get_sdr_for_token(t)
            output_spikes = self.transformer.forward(input_spikes, learning=False)
            combined      = input_spikes + [s + self.embed_dim for s in output_spikes]
            spike_history.insert(0, combined)
            if len(spike_history) > self.max_delay + 1:
                spike_history.pop()

        current_token = prompt_tokens[-1]

        for _ in range(max_new_tokens):
            input_spikes  = self._get_sdr_for_token(current_token)
            output_spikes = self.transformer.forward(input_spikes, learning=False)
            combined      = input_spikes + [s + self.embed_dim for s in output_spikes]
            spike_history.insert(0, combined)
            if len(spike_history) > self.max_delay + 1:
                spike_history.pop()

            token_potentials:    Dict[int, float] = {}
            token_support_count: Dict[int, int]   = {}

            for delay, active_spikes in enumerate(spike_history):
                time_decay  = max(0.1, 1.0 - delay * 0.08)
                supported: set = set()
                for s in active_spikes:
                    if s in self.weights[delay]:
                        for t_id, weight in self.weights[delay][s].items():
                            token_potentials[t_id] = (
                                token_potentials.get(t_id, 0.0) + weight * time_decay
                            )
                            supported.add(t_id)
                for t_id in supported:
                    token_support_count[t_id] = token_support_count.get(t_id, 0) + 1

            if not token_potentials:
                break

            for t_id in token_potentials:
                count = token_support_count.get(t_id, 1)
                token_potentials[t_id] *= (count ** 1.2)

            # QAボーナス（固定スケール加算）
            if qa_bonus and qa_scale_current > 0:
                for t_id, bonus_w in qa_bonus.items():
                    token_potentials[t_id] = (
                        token_potentials.get(t_id, 0.0) + bonus_w * qa_scale_current
                    )

            for t_id in list(token_potentials.keys()):
                hub = token_fan_in.get(t_id, 1.0)
                if hub > 1.0:
                    token_potentials[t_id] /= math.pow(hub, 0.2)

            forbidden = set(generated[-3:] + [current_token])
            for f_id in forbidden:
                if f_id in token_potentials:
                    token_potentials[f_id] *= 0.01

            sorted_candidates = sorted(
                token_potentials.items(), key=lambda x: x[1], reverse=True
            )

            if not sorted_candidates:
                break

            if temperature <= 0.1:
                next_token = sorted_candidates[0][0]
            else:
                top_k      = min(5, len(sorted_candidates))
                candidates = sorted_candidates[:top_k]
                total_pot  = sum(pow(p, 1.0 / temperature) for _, p in candidates)
                r          = random.uniform(0, total_pot)
                next_token = candidates[0][0]
                cumulative = 0.0
                for t_id, pot in candidates:
                    cumulative += pow(pot, 1.0 / temperature)
                    if r <= cumulative:
                        next_token = t_id
                        break

            if token_potentials.get(next_token, 0) < 0.001:
                break

            generated.append(next_token)
            current_token = next_token

            # QAボーナスを段階的に減衰
            # step0=1000 → step1=500 → step2=200 → step3以降=0
            gen_len = len(generated)
            if gen_len == 1:
                qa_scale_current = QA_SCALE * 0.5
            elif gen_len == 2:
                qa_scale_current = QA_SCALE * 0.2
            elif gen_len >= 3:
                qa_scale_current = 0.0

        return generated

    # ------------------------------------------------------------------
    # 保存 / 読み込み
    # ------------------------------------------------------------------
    def save_pretrained(self, filepath: str) -> None:
        serializable_weights = []
        for delay_dict in self.weights:
            serializable_weights.append({
                str(k): {str(tk): v for tk, v in tv.items()}
                for k, tv in delay_dict.items()
            })

        serializable_qa: Dict[str, Dict[str, float]] = {
            str(ctx_k): {str(tok_id): w for tok_id, w in tok_dict.items()}
            for ctx_k, tok_dict in self.qa_weights.items()
        }

        state = {
            "vocab_size":  self.vocab_size,
            "embed_dim":   self.embed_dim,
            "max_delay":   self.max_delay,
            "transformer": self.transformer.state_dict(),
            "weights":     serializable_weights,
            "qa_weights":  serializable_qa,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load_pretrained(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)

        self.vocab_size = state["vocab_size"]
        self.embed_dim  = state["embed_dim"]
        self.max_delay  = state.get("max_delay", 10)
        self.transformer.load_state_dict(state["transformer"])

        self.weights = []
        for delay_dict in state.get("weights", []):
            self.weights.append({
                int(k): {int(tk): float(v) for tk, v in tv.items()}
                for k, tv in delay_dict.items()
            })

        self.qa_weights = {}
        for ctx_k_str, tok_dict in state.get("qa_weights", {}).items():
            ctx_k = int(ctx_k_str)
            self.qa_weights[ctx_k] = {
                int(tok_id): float(w) for tok_id, w in tok_dict.items()
            }