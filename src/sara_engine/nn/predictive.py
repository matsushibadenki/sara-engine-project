# {
#     "//": "ディレクトリパス: src/sara_engine/nn/predictive.py",
#     "//": "ファイルの日本語タイトル: 予測符号化スパイキング層",
#     "//": "ファイルの目的や内容: 強いAI向けに動的閾値(Homeostasis)と予測消去(Cancellation)を強化。Target Propagationによる上位層からの局所教示を実装。PredictiveCodingManagerによるbackward weights学習と予測精度メトリクスを統合。"
# }

from .module import SNNModule
from typing import List, Dict, Tuple, Set, Optional
import random

# 新設した学習モジュールのインポート
from ..learning.predictive_coding import PredictiveCodingManager, TargetPropagationManager

try:
    from ..sara_rust_core import CausalSynapses
except ImportError:
    CausalSynapses = None

# --- 定数 ---
_WEIGHT_CAP: float = 3.0

# =====================================================================
# [1] 既存の双方向・空間的予測レイヤー (Legacy / Spatial)
# =====================================================================


class PredictiveSpikeLayer(SNNModule):
    def __init__(self, in_features: int, out_features: int, density: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.forward_weights: List[Dict[int, float]] = [{}
                                                        for _ in range(in_features)]
        self.backward_weights: List[Dict[int, float]] = [
            {} for _ in range(out_features)]

        for i in range(in_features):
            num_conn = max(1, int(out_features * density))
            targets = random.sample(range(out_features), num_conn)
            for t in targets:
                self.forward_weights[i][t] = random.uniform(0.3, 0.8)

        for i in range(out_features):
            num_conn = max(1, int(in_features * density))
            targets = random.sample(range(in_features), num_conn)
            for t in targets:
                self.backward_weights[i][t] = random.uniform(0.1, 0.5)

        self.register_state("forward_weights")
        self.register_state("backward_weights")

        # 予測元となる「内部状態」と、次層へ伝える「出力スパイク」を分離
        self.internal_state: List[int] = []
        self.recent_out_spikes: List[int] = []
        # 予測精度メトリクス
        self._total_inputs: int = 0
        self._total_predicted: int = 0

    def reset_state(self) -> None:
        super().reset_state()
        self.internal_state.clear()
        self.recent_out_spikes.clear()
        self._total_inputs = 0
        self._total_predicted = 0

    def get_prediction_metrics(self) -> Dict[str, float]:
        """予測精度のメトリクスを返す。"""
        if self._total_inputs == 0:
            return {"prediction_rate": 0.0, "total_inputs": 0, "total_predicted": 0}
        rate = self._total_predicted / self._total_inputs
        return {
            "prediction_rate": rate,
            "total_inputs": float(self._total_inputs),
            "total_predicted": float(self._total_predicted),
        }

    def forward(self, in_spikes: List[int], learning: bool = False, reward: float = 1.0, target_spikes: Optional[List[int]] = None) -> List[int]:
        # 1. 現在の入力からの純粋な状態表現を計算 (予測に依存せず安定させる)
        raw_potentials = [0.0] * self.out_features
        for s in in_spikes:
            if s < self.in_features:
                for t, w in self.forward_weights[s].items():
                    raw_potentials[t] += w
        current_state = [i for i, p in enumerate(raw_potentials) if p > 0.8]

        # 2. 1ステップ前の状態から、現在の入力を予測する
        pred_potentials = [0.0] * self.in_features
        for s in self.internal_state:
            if s < self.out_features:
                for t, w in self.backward_weights[s].items():
                    pred_potentials[t] += w
        predicted_in_spikes = set(
            [i for i, p in enumerate(pred_potentials) if p > 1.0])

        # 3. 予測誤差(Surprise)の抽出。予測できた入力は消去(Cancellation)される
        error_spikes = [s for s in in_spikes if s not in predicted_in_spikes]

        # メトリクス更新
        self._total_inputs += len(in_spikes)
        self._total_predicted += len(in_spikes) - len(error_spikes)

        # 4. 次の層へ送出するスパイクは「予測できなかった誤差の特徴」のみ
        out_potentials = [0.0] * self.out_features
        for s in error_spikes:
            if s < self.in_features:
                for t, w in self.forward_weights[s].items():
                    out_potentials[t] += w

        out_spikes = [i for i, p in enumerate(out_potentials) if p > 0.8]
        max_spikes = max(1, int(self.out_features * 0.25))

        if len(out_spikes) > max_spikes:
            out_spikes = sorted(out_spikes, key=lambda x: out_potentials[x], reverse=True)[
                :max_spikes]
        if len(current_state) > max_spikes:
            current_state = sorted(
                current_state, key=lambda x: raw_potentials[x], reverse=True)[:max_spikes]

        # 5. 学習フェーズ (Target Propagation & Predictive Coding)
        if learning:
            if target_spikes is not None:
                # Target Propagation: 上位層から目標が与えられた場合、それに従い強制学習
                TargetPropagationManager.apply_target(
                    self.forward_weights, in_spikes, current_state, target_spikes, lr=0.1 * reward
                )
            else:
                # Predictive Coding: 教師なし自己組織化
                PredictiveCodingManager.update_forward(
                    self.forward_weights, in_spikes, current_state, error_spikes, lr=0.05 * reward
                )

            # 予測精度の向上
            PredictiveCodingManager.update_backward(
                self.backward_weights, self.internal_state, in_spikes, predicted_in_spikes, lr=0.1 * reward
            )

        # 状態の更新
        self.internal_state = current_state
        self.recent_out_spikes = out_spikes
        return out_spikes

# =====================================================================
# [2] フェーズ2 時系列・誤差主導予測レイヤー (Phase 2 / Temporal)
# =====================================================================


class PredictiveCodingLayer(SNNModule):
    def __init__(self, max_delay: int = 10, learning_rate: float = 0.05, threshold: float = 0.5):
        super().__init__()
        self.max_delay, self.learning_rate, self.threshold = max_delay, learning_rate, threshold
        if CausalSynapses is None:
            raise RuntimeError("sara_rust_core is not available.")
        self.synapses = CausalSynapses(max_delay=max_delay)
        self.spike_history: List[List[int]] = []
        self.register_state("spike_history")

    def forward(self, actual_spikes: List[int], learning: bool = True) -> Tuple[List[int], float]:
        if not self.spike_history:
            self.spike_history.insert(0, actual_spikes)
            return actual_spikes, 1.0
        if learning:
            error_spikes, error_rate = self.synapses.predict_and_learn(
                self.spike_history, actual_spikes, self.learning_rate, self.threshold)
            self.spike_history.insert(0, actual_spikes)
            if len(self.spike_history) > self.max_delay:
                self.spike_history.pop()
            return error_spikes, error_rate
        else:
            pot = self.synapses.calculate_potentials(self.spike_history)
            pred = [t for t, p in pot.items() if p >= self.threshold]
            err = list(set(actual_spikes) - set(pred))
            self.spike_history.insert(0, actual_spikes)
            if len(self.spike_history) > self.max_delay:
                self.spike_history.pop()
            return err, len(err)/max(1, len(actual_spikes))

    def reset_state(self) -> None:
        self.spike_history.clear()
        super().reset_state()

# =====================================================================
# [3] 強いAI向け 階層的予測符号化レイヤー (Phase 3 / Hierarchical)
# =====================================================================


class SpikingPredictiveLayer(SNNModule):
    def __init__(self, in_features: int, out_features: int, density: float = 0.1, threshold: float = 0.2):
        super().__init__()
        self.in_features, self.out_features, self.threshold = in_features, out_features, threshold
        self.forward_weights: List[Dict[int, float]] = [{}
                                                        for _ in range(in_features)]
        self.backward_weights: List[Dict[int, float]] = [
            {} for _ in range(out_features)]
        self._init_sparse(self.forward_weights, in_features,
                          out_features, density)
        self._init_sparse(self.backward_weights,
                          out_features, in_features, density)
        self.register_state("forward_weights")
        self.register_state("backward_weights")
        self.last_state: Set[int] = set()
        self.adaptation: Dict[int, float] = {}
        # 予測精度メトリクス
        self._total_inputs: int = 0
        self._total_predicted: int = 0

    def _init_sparse(self, weights: List[Dict[int, float]], src: int, dst: int, dens: float) -> None:
        for i in range(src):
            for t in random.sample(range(dst), max(1, int(dst * dens))):
                weights[i][t] = random.uniform(0.1, 0.5)

    def get_prediction_metrics(self) -> Dict[str, float]:
        """予測精度のメトリクスを返す。"""
        if self._total_inputs == 0:
            return {"prediction_rate": 0.0, "total_inputs": 0, "total_predicted": 0}
        rate = self._total_predicted / self._total_inputs
        return {
            "prediction_rate": rate,
            "total_inputs": float(self._total_inputs),
            "total_predicted": float(self._total_predicted),
        }

    def forward(self, bottom_up: List[int], top_down: Optional[List[int]] = None, learning: bool = True, target_spikes: Optional[List[int]] = None) -> Tuple[List[int], List[int]]:
        bu_set = set(bottom_up)
        td_set = set(top_down) if top_down else set()

        predicted_bu: Set[int] = set()
        if self.last_state:
            pot: Dict[int, float] = {}
            for s in self.last_state:
                if s < len(self.backward_weights):
                    for t, w in self.backward_weights[s].items():
                        pot[t] = pot.get(t, 0.0) + w
            predicted_bu = {t for t, p in pot.items() if p > self.threshold}

        positive_error = bu_set - predicted_bu

        # メトリクス更新
        self._total_inputs += len(bu_set)
        self._total_predicted += len(bu_set) - len(positive_error)

        forward_input = positive_error if learning else bu_set

        out_pot: Dict[int, float] = {}
        for s in forward_input:
            if s < len(self.forward_weights):
                for t, w in self.forward_weights[s].items():
                    effective_w = w - self.adaptation.get(t, 0.0)
                    out_pot[t] = out_pot.get(t, 0.0) + max(0.0, effective_w)

        for td_s in td_set:
            if td_s < self.out_features:
                out_pot[td_s] = out_pot.get(td_s, 0.0) + 0.1

        state_spikes: Set[int] = set()
        if out_pot:
            k = max(1, int(self.out_features * 0.02))
            sorted_s = sorted(
                out_pot.items(), key=lambda x: x[1], reverse=True)
            state_spikes = {t for t, p in sorted_s[:k] if p > self.threshold}

        if learning and not state_spikes and positive_error:
            state_spikes = set(random.sample(range(self.out_features), 2))

        if learning:
            if target_spikes is not None:
                # Target Propagation: 上位層からの目標に従い強制学習
                TargetPropagationManager.apply_target(
                    self.forward_weights, bottom_up, list(state_spikes), target_spikes)
            else:
                # Predictive Coding: 正の予測誤差からの自己組織化学習
                for pre in positive_error:
                    if pre < len(self.forward_weights):
                        for post in state_spikes:
                            self.forward_weights[pre][post] = min(
                                _WEIGHT_CAP, self.forward_weights[pre].get(
                                    post, 0.0) + 0.2
                            )

            # backward weights の学習に PredictiveCodingManager を使用
            PredictiveCodingManager.update_backward(
                self.backward_weights,
                list(self.last_state),
                bottom_up,
                predicted_bu,
                lr=0.2,
            )

            for pre in state_spikes:
                self.adaptation[pre] = self.adaptation.get(pre, 0.0) + 0.05

        for k_ad in list(self.adaptation.keys()):
            self.adaptation[k_ad] *= 0.9
            if self.adaptation[k_ad] < 0.01:
                del self.adaptation[k_ad]

        self.last_state = state_spikes
        return list(state_spikes), list(predicted_bu)

    def reset_state(self) -> None:
        super().reset_state()
        self.last_state.clear()
        self.adaptation.clear()
        self._total_inputs = 0
        self._total_predicted = 0
