from .module import SNNModule
from typing import List, Dict, Tuple, Set, Optional
import random
{
    "//": "ディレクトリパス: src/sara_engine/nn/predictive.py",
    "//": "ファイルの日本語タイトル: 予測符号化スパイキング層",
    "//": "ファイルの目的や内容: 強いAI向けに動的閾値(Homeostasis)と予測消去(Cancellation)を強化。SyntaxErrorを修正。"
}


try:
    from sara_engine.sara_rust_core import CausalSynapses
except ImportError:
    CausalSynapses = None

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
        self.recent_out_spikes: List[int] = []

    def reset_state(self):
        super().reset_state()
        self.recent_out_spikes.clear()

    def forward(self, in_spikes: List[int], learning: bool = False, reward: float = 1.0) -> List[int]:
        pred_potentials = [0.0] * self.in_features
        for s in self.recent_out_spikes:
            if s < self.out_features:
                for t, w in self.backward_weights[s].items():
                    if t < self.in_features:
                        pred_potentials[t] += w
        predicted_in_spikes = set(
            [i for i, p in enumerate(pred_potentials) if p > 1.0])
        error_spikes = [s for s in in_spikes if s not in predicted_in_spikes]
        out_potentials = [0.0] * self.out_features
        for s in error_spikes:
            if s < self.in_features:
                for t, w in self.forward_weights[s].items():
                    if t < self.out_features:
                        out_potentials[t] += w
        out_spikes = [i for i, p in enumerate(out_potentials) if p > 0.8]
        if len(out_spikes) > max(1, int(self.out_features * 0.25)):
            out_spikes = sorted(out_spikes, key=lambda x: out_potentials[x], reverse=True)[
                :max(1, int(self.out_features * 0.25))]
        if learning:
            out_set = set(out_spikes)
            in_set = set(in_spikes)
            for s in error_spikes:
                if s < self.in_features:
                    for t in list(self.forward_weights[s].keys()):
                        if t in out_set:
                            self.forward_weights[s][t] = min(
                                3.0, self.forward_weights[s][t] + 0.15 * reward)
                        else:
                            self.forward_weights[s][t] -= 0.05
                            if self.forward_weights[s][t] <= 0:
                                del self.forward_weights[s][t]
            for s in self.recent_out_spikes:
                if s < self.out_features:
                    for t in in_set:
                        if t not in self.backward_weights[s] and random.random() < 0.3:
                            self.backward_weights[s][t] = 0.5
                    for t in list(self.backward_weights[s].keys()):
                        if t in in_set:
                            self.backward_weights[s][t] = min(
                                3.0, self.backward_weights[s][t] + 0.3 * reward)
                        else:
                            self.backward_weights[s][t] -= 0.1
                            if self.backward_weights[s][t] <= 0:
                                del self.backward_weights[s][t]
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

    def reset_state(self):
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

    def _init_sparse(self, weights: List[Dict[int, float]], src: int, dst: int, dens: float):
        for i in range(src):
            for t in random.sample(range(dst), max(1, int(dst * dens))):
                weights[i][t] = random.uniform(0.1, 0.5)

    def forward(self, bottom_up: List[int], top_down: Optional[List[int]] = None, learning: bool = True) -> Tuple[List[int], List[int]]:
        bu_set = set(bottom_up)
        td_set = set(top_down) if top_down else set()

        # 前ステップの状態からボトムアップ入力を予測
        predicted_bu: Set[int] = set()
        if self.last_state:
            pot: Dict[int, float] = {}
            for s in self.last_state:
                if s < len(self.backward_weights):
                    for t, w in self.backward_weights[s].items():
                        pot[t] = pot.get(t, 0.0) + w
            predicted_bu = {t for t, p in pot.items() if p > self.threshold}

        positive_error = bu_set - predicted_bu
        forward_input = positive_error if learning else bu_set

        # 出力ポテンシャルの計算
        out_pot: Dict[int, float] = {}
        for s in forward_input:
            if s < len(self.forward_weights):
                for t, w in self.forward_weights[s].items():
                    effective_w = w - self.adaptation.get(t, 0.0)
                    out_pot[t] = out_pot.get(t, 0.0) + max(0.0, effective_w)

        # トップダウン信号を調整的バイアスとして適用（直接注入ではなく）
        for td_s in td_set:
            if td_s < self.out_features:
                out_pot[td_s] = out_pot.get(td_s, 0.0) + 0.1

        # スパーシティを高めて弁別力を向上（2%のアクティブ率）
        state_spikes: Set[int] = set()
        if out_pot:
            k = max(1, int(self.out_features * 0.02))
            sorted_s = sorted(
                out_pot.items(), key=lambda x: x[1], reverse=True)
            state_spikes = {t for t, p in sorted_s[:k] if p > self.threshold}

        if learning and not state_spikes and positive_error:
            state_spikes = set(random.sample(range(self.out_features), 2))

        # STDP学習
        if learning:
            for pre in positive_error:
                if pre < len(self.forward_weights):
                    for post in state_spikes:
                        self.forward_weights[pre][post] = min(
                            1.0, self.forward_weights[pre].get(post, 0.0) + 0.2)
            for pre in state_spikes:
                if pre < len(self.backward_weights):
                    for post in positive_error:
                        self.backward_weights[pre][post] = min(
                            1.0, self.backward_weights[pre].get(post, 0.0) + 0.2)
                self.adaptation[pre] = self.adaptation.get(pre, 0.0) + 0.05

        # 適応値の減衰
        for k_ad in list(self.adaptation.keys()):
            self.adaptation[k_ad] *= 0.9
            if self.adaptation[k_ad] < 0.01:
                del self.adaptation[k_ad]

        self.last_state = state_spikes
        return list(state_spikes), list(predicted_bu)

    def reset_state(self):
        super().reset_state()
        self.last_state.clear()
        self.adaptation.clear()
