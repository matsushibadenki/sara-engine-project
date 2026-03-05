# ディレクトリパス: src/sara_engine/models/snn_transformer.py
# ファイルの日本語タイトル: スパイキング・トランスフォーマーモデル v1.3.3
# ファイルの目的や内容: 破滅的忘却を防ぐため、過剰なLTD（一律減衰）を廃止し、穏やかな重み減衰（Weight Decay）と誤答へのピンポイントなペナルティに変更。文脈も正しくブロックに入力するよう修正。

from sara_engine.core.spike_attention import SpikeMultiPathwayAttention
from sara_engine.nn.attention import SpikeFuzzyAttention
from sara_engine import nn
from typing import List, Dict, Optional, Tuple
import operator
import pickle
import random
import os
import json
import math

# ---- 定数 ----------------------------------------------------------------
_MODEL_VERSION: str = "1.3.3"          
_UNICODE_MAX: int = 0x10FFFF           
_SYNAPSE_MAX_WEIGHT: float = 20.0      
_SYNAPSE_PRUNE_THRESH: float = 1.0     
_SYNAPSE_BUCKET_MAX: int = 8192        
_SYNAPSE_PRUNE_TARGET: int = 4096      

class SNNTransformerConfig:
    def __init__(
        self,
        vocab_size: int = 1114112,
        embed_dim: int = 64,
        num_layers: int = 1,
        ffn_dim: int = 256,
        num_pathways: int = 4,
        dropout_p: float = 0.1,
        target_spikes_ratio: float = 0.15,
        use_fuzzy: bool = False,
        replay_count: int = 2,
    ) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.num_pathways = num_pathways
        self.dropout_p = dropout_p
        self.target_spikes_ratio = target_spikes_ratio
        self.use_fuzzy = use_fuzzy
        self.replay_count = replay_count

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_version": _MODEL_VERSION,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "ffn_dim": self.ffn_dim,
            "num_pathways": self.num_pathways,
            "dropout_p": self.dropout_p,
            "target_spikes_ratio": self.target_spikes_ratio,
            "use_fuzzy": self.use_fuzzy,
            "replay_count": self.replay_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "SNNTransformerConfig":
        filtered: Dict[str, object] = {
            k: v for k, v in data.items() if k != "model_version"
        }
        return cls(**filtered)


class SNNTransformerBlock(nn.SNNModule):
    def __init__(self, config: SNNTransformerConfig) -> None:
        super().__init__()
        self.config = config

        target_spikes = max(1, int(config.embed_dim * config.target_spikes_ratio))

        self.norm1 = nn.SpikeLayerNorm(target_spikes=target_spikes)
        self.dropout1 = nn.SpikeDropout(p=config.dropout_p)

        if config.use_fuzzy:
            self.attention = SpikeFuzzyAttention(
                embed_dim=config.embed_dim,
                threshold=0.2,
                top_k=3,
            )
        else:
            self.attention = SpikeMultiPathwayAttention(
                embed_dim=config.embed_dim,
                num_pathways=config.num_pathways,
                context_size=128,
            )

        self.norm2 = nn.SpikeLayerNorm(target_spikes=target_spikes)
        self.dropout2 = nn.SpikeDropout(p=config.dropout_p)

        self.ffn = nn.Sequential(
            nn.LinearSpike(
                in_features=config.embed_dim,
                out_features=config.ffn_dim,
                density=0.2,
            ),
            nn.LinearSpike(
                in_features=config.ffn_dim,
                out_features=config.embed_dim,
                density=0.2,
            ),
        )
        self.max_block_spikes: int = max(1, config.embed_dim // 2)

    def forward(self, spikes: List[int], learning: bool = True) -> List[int]:
        norm_spikes1 = self.norm1(spikes, learning=learning)
        attn_out = self.attention.forward(norm_spikes1, learning=learning)
        drop_attn = self.dropout1(attn_out, learning=learning)
        res1_spikes: List[int] = list(set(spikes + drop_attn))

        norm_spikes2 = self.norm2(res1_spikes, learning=learning)
        ffn_out = self.ffn(norm_spikes2, learning=learning)
        drop_ffn = self.dropout2(ffn_out, learning=learning)
        res2_spikes: List[int] = list(set(res1_spikes + drop_ffn))

        if len(res2_spikes) > self.max_block_spikes:
            res2_spikes = sorted(res2_spikes)[:self.max_block_spikes]

        return res2_spikes


class SpikingTransformerModel(nn.SNNModule):
    def __init__(self, config: SNNTransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.context_length: int = 16
        self.reservoir_size: int = 8192

        self.total_readout_size: int = (self.reservoir_size * 3) + config.embed_dim

        layers: List[SNNTransformerBlock] = [
            SNNTransformerBlock(config) for _ in range(config.num_layers)
        ]
        self.transformer_layers = nn.Sequential(*layers)

        self.delay_buffer: List[int] = []

        self.readout_synapses: List[Dict[int, float]] = [
            {} for _ in range(self.total_readout_size)
        ]
        
        self.token_counts: Dict[int, int] = {}
        
        self.register_state("readout_synapses")
        self.register_state("token_counts")

    def reset_state(self) -> None:
        super().reset_state()
        self.delay_buffer.clear()
        for layer in getattr(self.transformer_layers, "modules", []):
            if hasattr(layer, "attention"):
                layer.attention.reset_state()

    def _get_reservoir_spikes(self, token_id: int) -> List[int]:
        self.delay_buffer.insert(0, token_id)
        if len(self.delay_buffer) > self.context_length:
            self.delay_buffer.pop()

        spikes: set = set()
        prev_tok: Optional[int] = None
        prev_prev_tok: Optional[int] = None

        for delay, tok in enumerate(self.delay_buffer):
            num_spikes = max(2, 24 - int(delay * 0.4))

            seed_u = (tok * 31337) ^ (delay * 982451653) ^ 0x5A5A5A5A
            state_u = seed_u & 0xFFFFFFFF
            if state_u == 0:
                state_u = 1
            for _ in range(num_spikes):
                state_u ^= (state_u << 13) & 0xFFFFFFFF
                state_u ^= (state_u >> 17) & 0xFFFFFFFF
                state_u ^= (state_u << 5) & 0xFFFFFFFF
                spikes.add(state_u % self.reservoir_size)

            if prev_tok is not None:
                seed_b = (tok * 31) ^ (prev_tok * 53) ^ (delay * 17) ^ 0x12345678
                state_b = seed_b & 0xFFFFFFFF
                if state_b == 0:
                    state_b = 1
                for _ in range(num_spikes // 2 + 1):
                    state_b ^= (state_b << 13) & 0xFFFFFFFF
                    state_b ^= (state_b >> 17) & 0xFFFFFFFF
                    state_b ^= (state_b << 5) & 0xFFFFFFFF
                    spikes.add((state_b % self.reservoir_size) + self.reservoir_size)

            if prev_prev_tok is not None and prev_tok is not None:
                seed_t = (
                    (tok * 13)
                    ^ (prev_tok * 37)
                    ^ (prev_prev_tok * 71)
                    ^ (delay * 23)
                    ^ 0x87654321
                )
                state_t = seed_t & 0xFFFFFFFF
                if state_t == 0:
                    state_t = 1
                for _ in range(max(1, num_spikes // 3)):
                    state_t ^= (state_t << 13) & 0xFFFFFFFF
                    state_t ^= (state_t >> 17) & 0xFFFFFFFF
                    state_t ^= (state_t << 5) & 0xFFFFFFFF
                    spikes.add(
                        (state_t % self.reservoir_size) + self.reservoir_size * 2
                    )

            prev_prev_tok = prev_tok
            prev_tok = tok

        return list(spikes)

    def _prune_synapses(
        self, synapses: Dict[int, float], protect_id: int
    ) -> None:
        if len(synapses) <= _SYNAPSE_BUCKET_MAX:
            return

        weak_keys = [
            k for k, v in synapses.items()
            if v < _SYNAPSE_PRUNE_THRESH and k != protect_id
        ]
        for k in weak_keys:
            del synapses[k]

        if len(synapses) > _SYNAPSE_BUCKET_MAX:
            sorted_keys = sorted(synapses.keys(), key=lambda k: synapses[k])
            for k in sorted_keys[:_SYNAPSE_PRUNE_TARGET]:
                if k != protect_id:
                    del synapses[k]

    @staticmethod
    def _temperature_sample(
        candidates: List[Tuple[int, float]],
        temperature: float,
    ) -> int:
        weights = [pow(max(1e-9, item[1]), 1.0 / temperature) for item in candidates]
        total_weight = sum(weights)
        r = random.uniform(0.0, total_weight)
        cumulative = 0.0
        for item, w in zip(candidates, weights):
            cumulative += w
            if r <= cumulative:
                return item[0]
        return candidates[0][0]

    def forward_step(
        self,
        token_id: int,
        learning: bool = True,
        target_id: Optional[int] = None,
        refractory_tokens: Optional[List[int]] = None,
        temperature: float = 0.6,
        fire_threshold: float = 10.0,
    ) -> int:
        def _is_valid_output_token(tid: int) -> bool:
            if tid <= 0 or tid > _UNICODE_MAX:
                return False
            if 0xD800 <= tid <= 0xDFFF:
                return False
            return True

        res_spikes: List[int] = self._get_reservoir_spikes(token_id)

        # 修正箇所: スパース性を保ちつつ文脈(res_spikes)をTransformerに入力するため、均等サンプリングを行う
        block_input_raw = sorted(list(set([s % self.config.embed_dim for s in res_spikes])))
        max_active = max(1, self.config.embed_dim // 4)
        if len(block_input_raw) > max_active:
            step = len(block_input_raw) / max_active
            block_input = [block_input_raw[int(i * step)] for i in range(max_active)]
        else:
            block_input = block_input_raw

        block_spikes: List[int] = self.transformer_layers(
            block_input, learning=learning
        )

        block_offset = self.reservoir_size * 3
        readout_spikes: List[int] = list(
            set(res_spikes + [s + block_offset for s in block_spikes])
        )

        out_potentials: Dict[int, float] = {}
        for s in readout_spikes:
            if s < self.total_readout_size:
                for v_idx, w in self.readout_synapses[s].items():
                    current = out_potentials.get(v_idx, 0.0)
                    out_potentials[v_idx] = current + w

        # 修正箇所: 頻出トークンの強すぎるペナルティを緩和 (0.4 -> 0.15)
        if out_potentials:
            for v_idx in out_potentials:
                fan_in = self.token_counts.get(v_idx, 1)
                out_potentials[v_idx] /= (math.log1p(fan_in) * 0.15 + 1.0)

        if not learning and refractory_tokens:
            decay_factor = 0.6
            for r_tok in reversed(refractory_tokens):
                if r_tok in out_potentials:
                    out_potentials[r_tok] *= decay_factor
                decay_factor += 0.1
                if decay_factor > 1.0:
                    decay_factor = 1.0

        predicted_id = 0
        margin = 0.0

        if out_potentials:
            if not learning:
                for k in out_potentials:
                    out_potentials[k] *= random.uniform(0.95, 1.05)

            sorted_items = sorted(
                out_potentials.items(),
                key=operator.itemgetter(1),
                reverse=True,
            )

            if learning:
                if sorted_items[0][1] > 0.1:
                    predicted_id = sorted_items[0][0]
                    if len(sorted_items) > 1:
                        margin = sorted_items[0][1] - sorted_items[1][1]
                    else:
                        margin = sorted_items[0][1]
            else:
                top_k = sorted_items[:5]
                if top_k[0][1] > fire_threshold:
                    valid_candidates = [
                        (tid, pot) for tid, pot in top_k
                        if _is_valid_output_token(tid)
                    ]
                    if valid_candidates:
                        predicted_id = self._temperature_sample(
                            valid_candidates, temperature
                        )

        if learning and target_id is not None:
            is_correct = (predicted_id == target_id)
            
            count = self.token_counts.get(target_id, 0) + 1
            self.token_counts[target_id] = count
            lr_scale = 1.0 / math.sqrt(count)

            if is_correct:
                reward_factor = max(0.5, 4.0 - margin)
                punish_factor = 0.2
            else:
                surprise = 1.0 + margin
                punish_factor = min(2.5, surprise * 1.5)
                reward_factor = 1.5

            for s in readout_spikes:
                if s >= self.total_readout_size:
                    continue

                synapses = self.readout_synapses[s]
                current_w = synapses.get(target_id, 0.0)
                new_w = min(_SYNAPSE_MAX_WEIGHT, current_w + (1.5 * reward_factor * lr_scale))
                synapses[target_id] = new_w

                # 修正箇所: マイナス引き算の破壊的な忘却を廃止し、0.999掛けによる自然な減衰(Weight Decay)に変更
                for t_id in list(synapses.keys()):
                    if t_id != target_id:
                        synapses[t_id] *= 0.999
                        if synapses[t_id] < 0.05:
                            del synapses[t_id]

                # 誤予測したトークンに対してのみ、個別にペナルティを与える
                if not is_correct and predicted_id in synapses:
                    synapses[predicted_id] -= punish_factor * 0.8
                    if synapses[predicted_id] <= 0.0:
                        del synapses[predicted_id]

                self._prune_synapses(synapses, protect_id=target_id)

        return predicted_id

    def learn_sequence(self, input_ids: List[int]) -> None:
        sequence = input_ids + [0]
        for _ in range(self.config.replay_count):
            self.reset_state()
            for i in range(len(sequence) - 1):
                self.forward_step(
                    sequence[i],
                    learning=True,
                    target_id=sequence[i + 1],
                )

    def generate(
        self,
        input_ids: List[int],
        max_length: int = 150,
        temperature: float = 0.3,
        fire_threshold: float = 10.0,
    ) -> List[int]:
        self.reset_state()

        first_pred = 0
        for token_id in input_ids:
            first_pred = self.forward_step(
                token_id,
                learning=False,
                temperature=temperature,
                fire_threshold=fire_threshold,
            )

        generated_ids: List[int] = []
        current_token = first_pred
        refractory_buffer: List[int] = []

        for _ in range(max_length):
            if current_token == 0:
                break

            generated_ids.append(current_token)
            refractory_buffer.append(current_token)

            if len(refractory_buffer) > 6:
                refractory_buffer.pop(0)

            current_token = self.forward_step(
                current_token,
                learning=False,
                refractory_tokens=refractory_buffer,
                temperature=temperature,
                fire_threshold=fire_threshold,
            )

        return generated_ids

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)

        config_path = os.path.join(save_directory, "config.json")
        config_dict = self.config.to_dict()
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

        state_path = os.path.join(save_directory, "model_state.pkl")
        with open(state_path, "wb") as f:
            pickle.dump(self.state_dict(), f)

        print(f"[SpikingTransformerModel] Saved to '{save_directory}' (v{_MODEL_VERSION})")

    @classmethod
    def from_pretrained(cls, save_directory: str) -> "SpikingTransformerModel":
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            raw: Dict[str, object] = json.load(f)

        config = SNNTransformerConfig.from_dict(raw)
        model = cls(config)

        state_path = os.path.join(save_directory, "model_state.pkl")
        if os.path.exists(state_path):
            with open(state_path, "rb") as f:
                state = pickle.load(f)
            model.load_state_dict(state)

        saved_version = raw.get("model_version", "unknown")
        print(
            f"[SpikingTransformerModel] Loaded from '{save_directory}' "
            f"(saved version: {saved_version}, current: {_MODEL_VERSION})"
        )
        return model