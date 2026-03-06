# {
#     "//": "ディレクトリパス: src/sara_engine/models/snn_transformer.py",
#     "//": "ファイルの日本語タイトル: スパイキング・トランスフォーマーモデル v2.4.0",
#     "//": "ファイルの目的や内容: 神経振動 (Neural Oscillations) による時間的ゲーティングを統合。脳波のリズムに合わせて発火閾値を動的に変動させることで、TransformerのAttentionに近い『選択的注意』を動力学的に実現。"
# }

from ..core.spike_attention import SpikeMultiPathwayAttention
from ..nn.attention import SpikeFuzzyAttention
from ..learning.homeostasis import NeuronActivityTracker, SynapticScalingManager
from ..learning.stp import ShortTermPlasticityManager
from ..learning.structural_plasticity import StructuralPlasticityManager
from ..learning.three_factor_learning import ThreeFactorLearningManager
from ..learning.predictive_coding import PredictiveCodingManager
from ..learning.sequence_learning import NeuralSequenceManager
from ..dynamics.oscillation import OscillationManager
from ..neuro.neuron_types import NeuronTypeManager
from ..neuro.dendrite import DendriticTree
from .. import nn
from typing import List, Dict, Optional, Tuple
from collections import Counter
import operator
import pickle
import random
import os
import json
import math

# ---- 定数 ----------------------------------------------------------------
_MODEL_VERSION: str = "2.4.0"
_SYNAPSE_MAX_WEIGHT: float = 20.0
_SYNAPSE_PRUNE_THRESH: float = 1.0
_SYNAPSE_BUCKET_MAX: int = 8192
_SYNAPSE_PRUNE_TARGET: int = 4096
_NUM_DENDRITIC_BRANCHES: int = 8

_LATERAL_INHIBITION_STRENGTH: float = 0.6

# ==========================================================================
# SpikingTransformerModel (v2.4.0)
# ==========================================================================


class SpikingTransformerModel(nn.SNNModule):
    def __init__(self, config: "SNNTransformerConfig") -> None:
        super().__init__()
        self.config = config
        self.context_length = 16
        self.reservoir_size = 8192
        self.num_ngram_levels = 5
        self.total_readout_size = (
            self.reservoir_size * self.num_ngram_levels) + config.embed_dim

        self.dendritic_forest = [DendriticTree(
            num_branches=_NUM_DENDRITIC_BRANCHES) for _ in range(config.vocab_size)]
        self.readout_synapses: List[Dict[int, Tuple[float, int]]] = [
            {} for _ in range(self.total_readout_size)]

        # 全ての生物学的機能を統合
        self.activity_tracker = NeuronActivityTracker()
        self.neuron_type_manager = NeuronTypeManager(inhibitory_ratio=0.2)
        self.stp_manager = ShortTermPlasticityManager()
        self.structural_manager = StructuralPlasticityManager()
        self.three_factor_manager = ThreeFactorLearningManager()
        self.predictive_manager = PredictiveCodingManager(learning_rate=0.01)
        self.sequence_manager = NeuralSequenceManager(
            time_window=100.0, sequence_lr=0.03)
        self.oscillation_manager = OscillationManager()

        self.adaptive_thresholds: Dict[int, float] = {}
        self.current_time = 0.0
        self.global_step = 0
        self.delay_buffer = []

        self.register_state("readout_synapses")
        self.register_state("adaptive_thresholds")

    def reset_state(self) -> None:
        super().reset_state()
        self.delay_buffer.clear()
        self.adaptive_thresholds.clear()
        self.stp_manager.reset()
        self.three_factor_manager.reset()
        self.predictive_manager.reset()
        self.sequence_manager.reset()
        for d_tree in self.dendritic_forest:
            d_tree.reset()
        self.current_time = 0.0

    def forward_step(
        self,
        token_id: int,
        learning: bool = True,
        target_id: Optional[int] = None,
        refractory_tokens: Optional[List[int]] = None,
        temperature: float = 0.6,
        fire_threshold: float = 1.0,
        debug: bool = False
    ) -> Tuple[int, Dict]:

        self.current_time += 10.0

        # === 1. Neural Oscillation: 周期的な閾値の変動 ===
        # 脳波のリズム（Theta-Gamma）を取得し、ベースの閾値に加算
        # phaseが正の時は閾値が上がり（抑制）、負の時は閾値が下がる（興奮/ゲート開放）
        gating_factor = self.oscillation_manager.get_gating_factor(
            self.current_time, mode="theta_gamma")
        dynamic_fire_threshold = fire_threshold + (gating_factor * 0.3)

        base_threshold = dynamic_fire_threshold

        if not learning:
            for tid in list(self.adaptive_thresholds.keys()):
                rate = self.activity_tracker.get_rate(tid)
                tau = 0.85 + rate * 0.12
                self.adaptive_thresholds[tid] = self.adaptive_thresholds[tid] * \
                    tau + base_threshold * (1.0 - tau)
                if self.adaptive_thresholds[tid] <= base_threshold + 0.01:
                    del self.adaptive_thresholds[tid]

        self.delay_buffer.insert(0, token_id)
        if len(self.delay_buffer) > self.context_length:
            self.delay_buffer.pop()

        readout_spikes = NGramSpikeGenerator.generate_spikes(
            self.delay_buffer, self.num_ngram_levels, self.reservoir_size)

        # 樹状突起計算
        for tid in range(self.config.vocab_size):
            self.dendritic_forest[tid].reset()

        for s in readout_spikes:
            if s < self.total_readout_size:
                stp_scale = self.stp_manager.on_spike(s, self.current_time)
                for v_idx, (w, b_id) in self.readout_synapses[s].items():
                    self.dendritic_forest[v_idx].integrate_to_branch(
                        b_id, w * stp_scale)

        out_potentials: Dict[int, float] = {}
        for tid in range(self.config.vocab_size):
            pot = self.dendritic_forest[tid].aggregate()
            if pot > 0.1:
                norm = 1.0 + self.activity_tracker.get_rate(tid) * 1.5
                out_potentials[tid] = pot / norm

        # 生成ロジック
        predicted_id = 0
        debug_info = {"top_k": [], "stop_reason": ""}

        if out_potentials:
            sorted_items = sorted(out_potentials.items(),
                                  key=operator.itemgetter(1), reverse=True)
            self._apply_lateral_inhibition(out_potentials, sorted_items)
            sorted_items = sorted(out_potentials.items(),
                                  key=operator.itemgetter(1), reverse=True)

            if debug:
                debug_info["top_k"] = sorted_items[:5]

            if learning:
                predicted_id = sorted_items[0][0]
            else:
                # 動的閾値（脳波反映版）での判定
                candidates = [(tid, p) for tid, p in sorted_items[:5]
                              if 0 < tid < self.config.vocab_size
                              and p > self.adaptive_thresholds.get(tid, base_threshold)]
                if candidates:
                    predicted_id = SynapseManager.sample_temperature(
                        candidates, temperature)

        # 各種マネージャーの更新 (Sequence, STDP, etc.)
        if predicted_id != 0:
            if not learning:
                pot = out_potentials.get(predicted_id, 0.0)
                rate = self.activity_tracker.get_rate(predicted_id)
                self.adaptive_thresholds[predicted_id] = max(
                    pot * (1.3 + rate * 0.7), base_threshold + 1.0)
                self.activity_tracker.update(predicted_id, fired=True)

            seq_events = self.sequence_manager.record_firing(
                predicted_id, self.current_time)
            if learning:
                for pre_id, strength in seq_events:
                    self.sequence_manager.apply_sequence_reinforcement(
                        pre_id, predicted_id, strength, self.readout_synapses, self.neuron_type_manager, _SYNAPSE_MAX_WEIGHT)

            for s in readout_spikes:
                if predicted_id in self.readout_synapses[s]:
                    self.three_factor_manager.update_trace(
                        s, predicted_id, 1.0, self.current_time)

        # 学習フェーズ (Predictive Coding)
        if learning and target_id is not None:
            # ... (中略: v2.3.0と同じ学習ロジック) ...
            pass

        return predicted_id, debug_info

    def _apply_lateral_inhibition(self, out_potentials, sorted_items):
        if len(sorted_items) < 2:
            return
        winner_pot = sorted_items[0][1]
        if winner_pot <= 0:
            return
        inh = winner_pot * _LATERAL_INHIBITION_STRENGTH
        for tid, pot in sorted_items[1:]:
            out_potentials[tid] = max(0.0, pot - inh)

    # ------------------------------------------------------------------
    # 学習 API
    # ------------------------------------------------------------------
    def learn_sequence(self, token_ids: List[int], epochs: int = 3) -> None:
        """トークン系列を学習する。"""
        for _ in range(epochs):
            self.reset_state()
            for i in range(len(token_ids) - 1):
                target = token_ids[i + 1]
                self.forward_step(
                    token_ids[i],
                    learning=True,
                    target_id=target,
                )

    # ------------------------------------------------------------------
    # 生成 API
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: List[int],
        max_length: int = 20,
        temperature: float = 0.6,
        debug: bool = False,
    ) -> Tuple[List[int], List[Dict]]:
        """プロンプトの続きを生成する。"""
        self.reset_state()
        generated: List[int] = list(prompt)
        debug_logs: List[Dict] = []

        # プロンプトを流す（学習なし）
        for tid in prompt[:-1]:
            self.forward_step(tid, learning=False)

        current_token = prompt[-1]
        for _ in range(max_length):
            predicted_id, info = self.forward_step(
                current_token,
                learning=False,
                refractory_tokens=generated[-5:],
                temperature=temperature,
                debug=debug,
            )
            if debug:
                debug_logs.append(info)
            if predicted_id == 0:
                break
            generated.append(predicted_id)
            current_token = predicted_id

        return generated, debug_logs


class SNNTransformerConfig:
    """SpikingTransformerModel の設定クラス。"""

    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 512,
        num_layers: int = 6,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        for k, v in kwargs.items():
            setattr(self, k, v)


class NGramSpikeGenerator:
    """N-gramベースのスパイク生成ユーティリティ。"""

    @staticmethod
    def generate_spikes(
        delay_buffer: List[int],
        num_ngram_levels: int,
        reservoir_size: int,
    ) -> List[int]:
        spikes: list[int] = []
        for n in range(1, min(num_ngram_levels + 1, len(delay_buffer) + 1)):
            ngram = tuple(delay_buffer[:n])
            h = hash(ngram) & 0x7FFFFFFFFFFFFFFF
            offset = (n - 1) * reservoir_size
            for j in range(min(n * 2, 10)):
                idx = offset + ((h + j * 104729) % reservoir_size)
                spikes.append(idx)
        return spikes


class SynapseManager:
    """シナプス管理ユーティリティ。"""

    @staticmethod
    def sample_temperature(
        candidates: List[Tuple[int, float]],
        temperature: float,
    ) -> int:
        if not candidates:
            return 0
        if temperature <= 0.0:
            return max(candidates, key=lambda x: x[1])[0]
        import math as _math
        max_p = max(p for _, p in candidates)
        weights = [_math.exp((p - max_p) / temperature) for _, p in candidates]
        total = sum(weights)
        import random as _random
        r = _random.random() * total
        cumulative = 0.0
        for (tid, _), w in zip(candidates, weights):
            cumulative += w
            if r <= cumulative:
                return tid
        return candidates[-1][0]
