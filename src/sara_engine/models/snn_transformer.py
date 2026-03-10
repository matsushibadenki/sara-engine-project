# {
#     "//": "ディレクトリパス: src/sara_engine/models/snn_transformer.py",
#     "//": "ファイルの日本語タイトル: スパイキング・トランスフォーマーモデル v2.4.5",
#     "//": "ファイルの目的や内容: 樹状突起ブランチの決定論的割り当てによる空間的加算（Spatial Summation）の修復と、高頻度語（記号・助詞）の暴走を抑えるためのLTD（長期抑圧）ペナルティの改善。"
# }

from ..learning.homeostasis import NeuronActivityTracker, SynapticScalingManager
from ..learning.stp import ShortTermPlasticityManager
from ..learning.structural_plasticity import StructuralPlasticityManager
from ..learning.three_factor_learning import ThreeFactorLearningManager
from ..learning.predictive_coding import PredictiveCodingManager
from ..learning.sequence_learning import NeuralSequenceManager
from ..dynamics.oscillation import OscillationManager
from ..cognitive.global_workspace import GlobalWorkspace
from ..neuro.neuron_types import NeuronTypeManager
from ..neuro.dendrite import DendriticTree
from .. import nn
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import operator
import pickle
import os
import math


class _CompatibleModelUnpickler(pickle.Unpickler):
    """Load legacy checkpoints that were pickled with older module paths."""

    _MODULE_ALIASES = {
        "src.sara_engine": "sara_engine",
        "src": "",
    }

    def find_class(self, module: str, name: str):
        if module.startswith("src.sara_engine"):
            module = module.replace("src.sara_engine", "sara_engine", 1)
        elif module == "src":
            module = "__main__"
        return super().find_class(module, name)


# ---- 定数 ----------------------------------------------------------------
_MODEL_VERSION: str = "2.4.5"
_SYNAPSE_MAX_WEIGHT: float = 20.0
_SYNAPSE_PRUNE_THRESH: float = 0.1  # ノイズ刈り込みを少し強化
_SYNAPSE_BUCKET_MAX: int = 8192
_SYNAPSE_PRUNE_TARGET: int = 4096
_NUM_DENDRITIC_BRANCHES: int = 8

_LATERAL_INHIBITION_STRENGTH: float = 0.6
_SPECIAL_TOKEN_IDS = {0, 1, 2, 3}
_HOMEOSTATIC_SPARSE_WINNER_RATIO: float = 3.0
_HOMEOSTATIC_SPARSE_MIN_SCALE: float = 0.35

# ==========================================================================
# SpikingTransformerModel
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

        self.activity_tracker = NeuronActivityTracker()
        self.scaling_manager = SynapticScalingManager(
            target_rate=0.05,
            scaling_lr=0.02,
            min_scale=0.9,
            max_scale=1.1,
            deadband=0.03,
            global_weight=0.35,
        )
        self.neuron_type_manager = NeuronTypeManager(inhibitory_ratio=0.2)
        self.stp_manager = ShortTermPlasticityManager()
        self.structural_manager = StructuralPlasticityManager()
        self.three_factor_manager = ThreeFactorLearningManager()
        self.predictive_manager = PredictiveCodingManager(
            learning_rate=0.25)  # さらに少し学習率UP
        self.sequence_manager = NeuralSequenceManager(
            time_window=100.0, sequence_lr=0.05)
        self.oscillation_manager = OscillationManager()
        self.global_workspace = GlobalWorkspace(
            num_candidates=8,
            inhibition_factor=0.35,
            winner_threshold=0.8,
            decay=0.82,
        )

        self.adaptive_thresholds: Dict[int, float] = {}
        self.target_counts: Dict[int, int] = {}
        self.current_time = 0.0
        self.global_step = 0
        self.delay_buffer: List[int] = []

        self.register_state("readout_synapses")
        self.register_state("adaptive_thresholds")
        self.register_state("target_counts")

    def reset_state(self) -> None:
        super().reset_state()
        self.delay_buffer.clear()
        self.adaptive_thresholds.clear()
        self.activity_tracker.reset()
        self.stp_manager.reset()
        self.three_factor_manager.reset()
        self.predictive_manager.reset()
        self.sequence_manager.reset()
        self.global_workspace.reset()
        for d_tree in self.dendritic_forest:
            d_tree.reset()
        self.current_time = 0.0

    def _advance_dynamics(self, fire_threshold: float, learning: bool) -> Tuple[float, float]:
        self.current_time += 10.0
        self.activity_tracker.step()
        population_rate = self.activity_tracker.get_global_rate()

        gating_factor = self.oscillation_manager.get_gating_factor(
            self.current_time, mode="theta_gamma")
        dynamic_fire_threshold = fire_threshold + (gating_factor * 0.3)

        base_threshold = dynamic_fire_threshold
        if not learning:
            target_rate = max(1e-6, self.scaling_manager.target_rate)
            pop_rel_error = (population_rate - target_rate) / target_rate
            homeostatic_shift = max(-0.22, min(0.55, pop_rel_error * 0.18))
            base_threshold = dynamic_fire_threshold + homeostatic_shift

            for tid in list(self.adaptive_thresholds.keys()):
                rate = self.activity_tracker.get_rate(tid)
                tau = 0.85 + rate * 0.12
                self.adaptive_thresholds[tid] = (
                    self.adaptive_thresholds[tid] * tau
                    + base_threshold * (1.0 - tau)
                )
                if self.adaptive_thresholds[tid] <= base_threshold + 0.01:
                    del self.adaptive_thresholds[tid]

        return base_threshold, population_rate

    def _ingest_token_context(self, token_id: int, fire_threshold: float) -> List[int]:
        self._advance_dynamics(fire_threshold=fire_threshold, learning=False)
        self.delay_buffer.insert(0, token_id)
        if len(self.delay_buffer) > self.context_length:
            self.delay_buffer.pop()

        readout_spikes = NGramSpikeGenerator.generate_spikes(
            self.delay_buffer, self.num_ngram_levels, self.reservoir_size)
        for s in readout_spikes:
            self.stp_manager.on_spike(s, self.current_time)
        return readout_spikes

    def _is_valid_output_token(self, token_id: int) -> bool:
        return token_id not in _SPECIAL_TOKEN_IDS and token_id < self.config.vocab_size

    def _select_sparse_winner(
        self,
        sorted_items: List[Tuple[int, float]],
        base_threshold: float,
        population_rate: float,
    ) -> List[Tuple[int, float]]:
        if not sorted_items:
            return []
        target_rate = max(1e-6, self.scaling_manager.target_rate)
        underfire = max(0.0, target_rate - population_rate) / target_rate
        if underfire <= 0.0:
            return []

        top_tid, top_p = sorted_items[0]
        if not self._is_valid_output_token(top_tid):
            return []

        second_p = 0.0
        for cand_tid, cand_p in sorted_items[1:]:
            if self._is_valid_output_token(cand_tid):
                second_p = cand_p
                break

        min_sparse_threshold = base_threshold * max(
            0.15, _HOMEOSTATIC_SPARSE_MIN_SCALE - (0.15 * min(1.0, underfire))
        )
        dominates = second_p <= 0.0 or top_p >= second_p * _HOMEOSTATIC_SPARSE_WINNER_RATIO
        if dominates and top_p >= min_sparse_threshold:
            return [(top_tid, top_p)]
        return []

    def _workspace_select_candidate(
        self,
        candidates: List[Tuple[int, float]],
    ) -> List[Tuple[int, float]]:
        if not candidates:
            return []
        limited = candidates[:self.global_workspace.num_candidates]
        winner_idx = self.global_workspace.step([p for _, p in limited])
        if winner_idx < 0 or winner_idx >= len(limited):
            return limited
        winner = limited[winner_idx]
        return [winner] + [item for idx, item in enumerate(limited) if idx != winner_idx]

    def forward_step(
        self,
        token_id: int,
        learning: bool = True,
        target_id: Optional[int] = None,
        refractory_tokens: Optional[List[int]] = None,
        recent_tokens: Optional[List[int]] = None,
        temperature: float = 0.6,
        fire_threshold: float = 1.0,
        debug: bool = False
    ) -> Tuple[int, Dict]:

        base_threshold, population_rate = self._advance_dynamics(
            fire_threshold=fire_threshold,
            learning=learning,
        )

        self.delay_buffer.insert(0, token_id)
        if len(self.delay_buffer) > self.context_length:
            self.delay_buffer.pop()

        readout_spikes = NGramSpikeGenerator.generate_spikes(
            self.delay_buffer, self.num_ngram_levels, self.reservoir_size)

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
                norm = 1.0 + self.activity_tracker.get_rate(tid) * 2.0
                out_potentials[tid] = pot / norm

        if not learning and out_potentials:
            target_rate = max(1e-6, self.scaling_manager.target_rate)
            global_overfire = max(0.0, population_rate - target_rate)
            if global_overfire > 0.0:
                global_damp = 1.0 + (global_overfire / target_rate) * 0.6
                for tid in list(out_potentials.keys()):
                    out_potentials[tid] /= global_damp

        if not learning and refractory_tokens and out_potentials:
            for i, rt in enumerate(reversed(refractory_tokens)):
                if rt in out_potentials:
                    out_potentials[rt] *= (0.1 * i)

        if not learning and recent_tokens and out_potentials:
            recent_window = recent_tokens[-24:]
            token_counts = Counter(recent_window)
            for tid in list(out_potentials.keys()):
                count = token_counts.get(tid, 0)
                if count <= 0:
                    continue
                penalty = 1.0 + min(4.0, count * 0.45)
                if recent_window and tid == recent_window[-1]:
                    penalty *= 2.5
                elif len(recent_window) >= 2 and tid == recent_window[-2]:
                    penalty *= 1.5
                out_potentials[tid] /= penalty

        predicted_id = 0
        debug_info: Dict[str, Any] = {
            "top_k": [], "stop_reason": "", "candidates": []}

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
                candidates = [(tid, p) for tid, p in sorted_items[:5]
                              if self._is_valid_output_token(tid)
                              and p > self.adaptive_thresholds.get(tid, base_threshold)]

                if not candidates and sorted_items:
                    top_tid, top_p = sorted_items[0]
                    if (
                        self._is_valid_output_token(top_tid)
                        and top_p > (base_threshold * 1.25)
                    ):
                        candidates = [(top_tid, top_p)]
                if not candidates:
                    candidates = self._select_sparse_winner(
                        sorted_items=sorted_items[:5],
                        base_threshold=base_threshold,
                        population_rate=population_rate,
                    )
                candidates = self._workspace_select_candidate(candidates)
                debug_info["candidates"] = candidates[:5]

                if candidates:
                    predicted_id = SynapseManager.sample_temperature(
                        candidates, temperature)
                else:
                    debug_info["stop_reason"] = "no_candidates_after_threshold"

        elif not learning:
            fallback_potentials: Dict[int, float] = {}
            for s in readout_spikes:
                if s >= len(self.readout_synapses):
                    continue
                for v_idx, (w, _b_id) in self.readout_synapses[s].items():
                    if 0 < v_idx < self.config.vocab_size:
                        fallback_potentials[v_idx] = fallback_potentials.get(
                            v_idx, 0.0) + w

            if refractory_tokens and fallback_potentials:
                for i, rt in enumerate(reversed(refractory_tokens)):
                    if rt in fallback_potentials:
                        fallback_potentials[rt] *= (0.1 * i)

            if fallback_potentials:
                sorted_items = sorted(
                    fallback_potentials.items(), key=operator.itemgetter(1), reverse=True
                )

                # --- 修正箇所：フォールバック時にも動的閾値（恒常性）によるブロックを適用 ---
                candidates = [(tid, p) for tid, p in sorted_items[:5]
                              if self._is_valid_output_token(tid)
                              and p > self.adaptive_thresholds.get(tid, base_threshold)]

                # 閾値を満たすものが無い場合は一番強いものを妥協して選ぶ
                if not candidates and sorted_items:
                    top_tid, top_p = sorted_items[0]
                    if (
                        self._is_valid_output_token(top_tid)
                        and top_p > (base_threshold * 1.35)
                    ):
                        candidates = [(top_tid, top_p)]
                if not candidates:
                    candidates = self._select_sparse_winner(
                        sorted_items=sorted_items[:5],
                        base_threshold=base_threshold,
                        population_rate=population_rate,
                    )
                candidates = self._workspace_select_candidate(candidates)

                if debug:
                    debug_info["top_k"] = candidates
                debug_info["candidates"] = candidates[:5]

                if candidates:
                    predicted_id = SynapseManager.sample_temperature(
                        candidates, temperature=max(temperature, 0.2)
                    )
                else:
                    predicted_id = 0
                # -------------------------------------------------------------------

                debug_info["stop_reason"] = "fallback_linear_readout"
            else:
                debug_info["stop_reason"] = "no_out_potentials"

        if predicted_id != 0:
            if not learning:
                pot = out_potentials.get(predicted_id, 0.0)
                rate = self.activity_tracker.get_rate(predicted_id)
                current_thr = self.adaptive_thresholds.get(
                    predicted_id, base_threshold)
                target_thr = max(base_threshold + 0.35,
                                 pot * (1.2 + rate * 0.7))
                # 閾値を急激に跳ね上げず、EMAで滑らかに更新する
                self.adaptive_thresholds[predicted_id] = (
                    current_thr * 0.65 + target_thr * 0.35
                )

            self.activity_tracker.update(predicted_id, fired=True)

            seq_events = self.sequence_manager.record_firing(
                predicted_id, self.current_time)

            if learning:
                for pre_id, strength in seq_events:
                    self.sequence_manager.apply_sequence_reinforcement(
                        pre_id, predicted_id, strength, self.readout_synapses, self.neuron_type_manager, _SYNAPSE_MAX_WEIGHT)

                current_rate = self.activity_tracker.get_rate(predicted_id)
                scaling_factor = self.scaling_manager.compute_scaling_factor(
                    current_rate=current_rate,
                    population_rate=population_rate,
                )

                if abs(1.0 - scaling_factor) > 0.005:
                    for s in readout_spikes:
                        if predicted_id in self.readout_synapses[s]:
                            w, b_id = self.readout_synapses[s][predicted_id]
                            new_w = min(w * scaling_factor,
                                        _SYNAPSE_MAX_WEIGHT)

                            if new_w < _SYNAPSE_PRUNE_THRESH:
                                del self.readout_synapses[s][predicted_id]
                            else:
                                self.readout_synapses[s][predicted_id] = (
                                    new_w, b_id)

            for s in readout_spikes:
                if predicted_id in self.readout_synapses[s]:
                    self.three_factor_manager.update_trace(
                        s, predicted_id, 1.0, self.current_time)

        # === 修正箇所：Predictive Codingの学習最適化 ===
        if learning and target_id is not None:
            self.target_counts[target_id] = self.target_counts.get(
                target_id, 0) + 1
            count = self.target_counts[target_id]

            # 高頻度語への学習率低下をマイルドにし、全く学習されなくなるのを防ぐ
            freq_penalty = max(
                0.2, 1.0 / math.log(count + 1.5)) if count > 2 else 1.0
            freq_norm_lr = self.predictive_manager.learning_rate * freq_penalty

            target_pot = out_potentials.get(target_id, 0.0)
            prediction_error = max(0.0, 1.0 - target_pot)

            if prediction_error > 0.05:
                actual_lr = freq_norm_lr * prediction_error
                for s in readout_spikes:
                    if target_id not in self.readout_synapses[s]:
                        # 【重要】ランダム割り当てを廃止。トークンとスパイクの組み合わせから決定論的にブランチを決定
                        branch_id = (s + target_id) % _NUM_DENDRITIC_BRANCHES
                        self.readout_synapses[s][target_id] = (1.5, branch_id)

                    w, b_id = self.readout_synapses[s][target_id]
                    new_w = min(w + actual_lr, _SYNAPSE_MAX_WEIGHT)
                    self.readout_synapses[s][target_id] = (new_w, b_id)

            # 間違って発火した（しそうになった）ノイズトークンへのペナルティ
            for false_id, false_pot in out_potentials.items():
                if false_id != target_id and false_pot > 0.3:
                    # 頻度が高い（「の」「は」など）間違った発火ほど強くペナルティを与え、暴走を抑える
                    f_count = self.target_counts.get(false_id, 1)
                    boost = min(3.0, math.log(f_count + 1.0))
                    penalty = (
                        self.predictive_manager.learning_rate * 0.4) * boost

                    for s in readout_spikes:
                        if false_id in self.readout_synapses[s]:
                            w, b_id = self.readout_synapses[s][false_id]
                            new_w = w - penalty

                            if new_w < _SYNAPSE_PRUNE_THRESH:
                                del self.readout_synapses[s][false_id]
                            else:
                                self.readout_synapses[s][false_id] = (
                                    new_w, b_id)

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

    def learn_sequence(self, token_ids: List[int], epochs: int = 3) -> None:
        for _ in range(epochs):
            self.reset_state()
            for i in range(len(token_ids) - 1):
                target = token_ids[i + 1]
                self.forward_step(
                    token_ids[i],
                    learning=True,
                    target_id=target,
                )

    def generate(
        self,
        prompt: List[int],
        max_length: int = 20,
        temperature: float = 0.6,
        fire_threshold: float = 1.0,
        debug: bool = False,
    ) -> Tuple[List[int], List[Dict]]:
        self.reset_state()
        generated: List[int] = list(prompt)
        debug_logs: List[Dict] = []

        for tid in prompt[:-1]:
            self._ingest_token_context(tid, fire_threshold=fire_threshold)

        current_token = prompt[-1]
        for _ in range(max_length):
            retry_blocked: set[int] = set()
            predicted_id = 0
            info: Dict[str, Any] = {}
            for _attempt in range(3):
                predicted_id, info = self.forward_step(
                    current_token,
                    learning=False,
                    refractory_tokens=generated[-5:],
                    recent_tokens=generated[-32:],
                    temperature=temperature,
                    fire_threshold=fire_threshold,
                    debug=debug,
                )
                if predicted_id in retry_blocked:
                    predicted_id = 0
                    break
                if predicted_id == 0:
                    break
                if self._would_repeat_ngram(generated, predicted_id, n=4):
                    retry_blocked.add(predicted_id)
                    current_thr = self.adaptive_thresholds.get(
                        predicted_id, fire_threshold)
                    self.adaptive_thresholds[predicted_id] = max(
                        current_thr, current_thr + 0.8)
                    info["stop_reason"] = "blocked_repeat_4gram"
                    predicted_id = 0
                    continue
                if self._would_repeat_ngram(generated, predicted_id, n=3):
                    retry_blocked.add(predicted_id)
                    current_thr = self.adaptive_thresholds.get(
                        predicted_id, fire_threshold)
                    self.adaptive_thresholds[predicted_id] = max(
                        current_thr, current_thr + 0.45)
                    info["stop_reason"] = "blocked_repeat_3gram"
                    predicted_id = 0
                    continue
                break
            if debug:
                debug_logs.append(info)
            if predicted_id == 0 or predicted_id == 3:
                break
            generated.append(predicted_id)
            current_token = predicted_id

        return generated, debug_logs

    def _would_repeat_ngram(self, generated: List[int], next_token: int, n: int) -> bool:
        if n <= 1 or (len(generated) + 1) < n:
            return False
        candidate = tuple((generated + [next_token])[-n:])
        for i in range(len(generated) - n + 1):
            if tuple(generated[i:i + n]) == candidate:
                return True
        return False

    def save_pretrained(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "snn_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "config": self.config,
                "readout_synapses": self.readout_synapses,
                "adaptive_thresholds": self.adaptive_thresholds,
                "target_counts": getattr(self, "target_counts", {})
            }, f)
        print(f"Model successfully saved to {model_path}")

    @classmethod
    def from_pretrained(cls, save_dir: str) -> "SpikingTransformerModel":
        model_path = os.path.join(save_dir, "snn_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pre-trained model not found at {model_path}")

        with open(model_path, "rb") as f:
            state = _CompatibleModelUnpickler(f).load()

        model = cls(state["config"])
        model.readout_synapses = state["readout_synapses"]
        model.adaptive_thresholds = state["adaptive_thresholds"]
        if "target_counts" in state:
            model.target_counts = state["target_counts"]
        print(f"Model successfully loaded from {model_path}")
        return model


class SNNTransformerConfig:
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
    @staticmethod
    def sample_temperature(
        candidates: List[Tuple[int, float]],
        temperature: float,
    ) -> int:
        if not candidates:
            return 0
        if temperature <= 0.0:
            return max(candidates, key=lambda x: x[1])[0]
        max_p = max(p for _, p in candidates)
        weights = [math.exp((p - max_p) / temperature) for _, p in candidates]
        total = sum(weights)
        import random as _random
        r = _random.random() * total
        cumulative = 0.0
        for (tid, _), w in zip(candidates, weights):
            cumulative += w
            if r <= cumulative:
                return tid
        return candidates[-1][0]
