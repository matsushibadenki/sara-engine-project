# {
#     "//": "ディレクトリパス: src/sara_engine/dynamics/oscillation.py",
#     "//": "ファイルの日本語タイトル: 神経振動と短期可塑性(STP)の統合ダイナミクス",
#     "//": "ファイルの目的や内容: 脳波リズムによる時間的ゲーティングと、Tsodyks-Markramモデルによる短期シナプス可塑性(STP)を統合。誤差逆伝播なし・行列演算なしで、高度な時系列パターン認識を可能にする脳型コアエンジン。"
# }

import math
import random
from typing import Dict, List, Optional, Tuple

# ==========================================================================
# 1. OscillationManager (神経振動 / 時間的ゲーティング)
# ==========================================================================

class OscillationManager:
    """
    神経振動（Neural Oscillations）による時間的ゲーティングの管理。
    """
    def __init__(self):
        # 脳波の周波数定義 (Hz)
        self.frequencies = {
            "theta": 6.0,   # 文脈・記憶保持 (sentence level)
            "alpha": 10.0,  # 抑制・注意の切り替え
            "beta":  20.0,  # 運動・予測の更新
            "gamma": 40.0   # 知覚・特徴結合 (word level)
        }

    def get_phase_effects(self, current_time_ms: float) -> Dict[str, float]:
        """
        現在の時刻における各脳波の位相（-1.0 〜 1.0）を取得する。
        """
        t_sec = current_time_ms / 1000.0
        effects = {}
        for name, freq in self.frequencies.items():
            phase = math.sin(2 * math.pi * freq * t_sec)
            effects[name] = phase
        return effects

    def get_gating_factor(self, current_time_ms: float, mode: str = "theta_gamma") -> float:
        """
        特定のモードに基づいた閾値変動係数（Gating Factor）を計算。
        """
        t_sec = current_time_ms / 1000.0
        if mode == "theta_gamma":
            theta = math.sin(2 * math.pi * self.frequencies["theta"] * t_sec)
            gamma = math.sin(2 * math.pi * self.frequencies["gamma"] * t_sec)
            return (theta * 0.5) + (gamma * 0.2)
        return math.sin(2 * math.pi * self.frequencies["alpha"] * t_sec)

# ==========================================================================
# 2. STPSynapse (Tsodyks–Markram モデル / 短期可塑性)
# ==========================================================================

class STPSynapse:
    """
    短期シナプス可塑性（Short-Term Plasticity）を実装したシナプス。
    u: 利用率 (release probability / facilitation)
    x: 利用可能資源 (available resources / depression)
    """
    def __init__(
        self, 
        weight: float, 
        U: float = 0.2, 
        tau_f: float = 600.0, 
        tau_d: float = 200.0
    ):
        self.w = weight  # 長期的な基底重み
        self.U = U       # 基本放出率
        self.tau_f = tau_f  # 促通の回復時間 (ms)
        self.tau_d = tau_d  # 抑圧の回復時間 (ms)

        self.u = self.U  # 現在の放出確率
        self.x = 1.0     # 現在の資源量
        self.last_update_t = 0.0

    def update(self, current_time: float, pre_spike: bool) -> float:
        """
        シナプス状態を更新し、有効重み（w_eff）を返す。
        """
        dt = current_time - self.last_update_t
        
        # 1. スパイクが無い時間の回復 (指数減衰・回復)
        if dt > 0:
            self.u = self.U + (self.u - self.U) * math.exp(-dt / self.tau_f)
            self.x = 1.0 + (self.x - 1.0) * math.exp(-dt / self.tau_d)

        # 2. 有効重みの計算 (w * u * x)
        effective_w = self.w * self.u * self.x

        # 3. スパイク発生時の状態遷移
        if pre_spike:
            # 放出確率の増加 (Facilitation)
            self.u += self.U * (1.0 - self.u)
            # 資源の消費 (Depression)
            self.x *= (1.0 - self.u)

        self.last_update_t = current_time
        return effective_w

# ==========================================================================
# 3. LIFNeuron (神経振動による動的閾値制御付き)
# ==========================================================================

class LIFNeuron:
    """
    Leaky Integrate-and-Fire (LIF) ニューロン。
    神経振動（Oscillation）によって発火閾値が変動する。
    """
    def __init__(self, base_threshold: float = 1.0, leak: float = 0.95):
        self.v = 0.0
        self.base_threshold = base_threshold
        self.leak = leak
        self.spike = False
        self.refractory_period = 0  # 不応期カウンタ

    def step(self, input_current: float, gating_factor: float = 0.0):
        """
        1ステップ更新。gating_factor によって閾値が上下する。
        """
        if self.refractory_period > 0:
            self.refractory_period -= 1
            self.v = 0.0
            self.spike = False
            return False

        # 1. 膜電位の蓄積とリーク
        self.v = (self.v + input_current) * self.leak

        # 2. 動的閾値の計算 (神経振動の位相を反映)
        # gating_factorが正の時、閾値が上がり発火しにくくなる
        dynamic_threshold = self.base_threshold + (gating_factor * 0.3)

        # 3. 発火判定
        if self.v > dynamic_threshold:
            self.spike = True
            self.v = 0.0
            self.refractory_period = 2 # 簡易的な不応期
        else:
            self.spike = False

        return self.spike

# ==========================================================================
# 4. DynamicSpikingNetwork (統合ネットワーク)
# ==========================================================================

class DynamicSpikingNetwork:
    """
    神経振動 + STP + LIF を統合した自律学習型リザバー。
    """
    def __init__(self, n_in: int, n_res: int, n_out: int):
        self.oscillation = OscillationManager()
        self.neurons_in = [LIFNeuron() for _ in range(n_in)]
        self.neurons_res = [LIFNeuron(base_threshold=1.2) for _ in range(n_res)]
        self.neurons_out = [LIFNeuron() for _ in range(n_out)]

        # シナプスリスト: (pre_neuron_index, post_neuron_index, stp_synapse)
        self.synapses_in_to_res = []
        self.synapses_res_recurrent = []
        self.synapses_res_to_out = []

        self._init_synapses(n_in, n_res, n_out)
        self.current_time_ms = 0.0

    def _init_synapses(self, n_in, n_res, n_out):
        # 入力 -> リザバー (スパース)
        for i in range(n_in):
            for j in range(n_res):
                if random.random() < 0.3:
                    w = random.uniform(0.1, 0.5)
                    self.synapses_in_to_res.append((i, j, STPSynapse(w)))

        # リザバー内再帰結合 (カオスの縁)
        for i in range(n_res):
            for j in range(n_res):
                if i != j and random.random() < 0.1:
                    w = random.uniform(-0.2, 0.4) # 抑制性結合も混ぜる
                    self.synapses_res_recurrent.append((i, j, STPSynapse(w)))

        # リザバー -> 出力
        for i in range(n_res):
            for j in range(n_out):
                w = random.uniform(0.0, 0.3)
                self.synapses_res_to_out.append((i, j, STPSynapse(w)))

    def step(self, input_signals: List[bool]) -> List[bool]:
        """
        ネットワーク全体を1ミリ秒進める。
        """
        self.current_time_ms += 1.0
        gating = self.oscillation.get_gating_factor(self.current_time_ms)

        # 1. 入力層の更新
        for i, fired in enumerate(input_signals):
            # 外部入力は電流として流し込む
            self.neurons_in[i].step(1.5 if fired else 0.0, gating)

        # 2. 電流の集計バッファ
        res_currents = [0.0] * len(self.neurons_res)
        out_currents = [0.0] * len(self.neurons_out)

        # 3. 入力 -> リザバーへの伝達 (STP適用)
        for pre_idx, post_idx, syn in self.synapses_in_to_res:
            pre_spike = self.neurons_in[pre_idx].spike
            w_eff = syn.update(self.current_time_ms, pre_spike)
            if pre_spike:
                res_currents[post_idx] += w_eff

        # 4. リザバー再帰結合の伝達
        # 前のステップのスパイク状態を使用
        res_spikes_prev = [n.spike for n in self.neurons_res]
        for pre_idx, post_idx, syn in self.synapses_res_recurrent:
            pre_spike = res_spikes_prev[pre_idx]
            w_eff = syn.update(self.current_time_ms, pre_spike)
            if pre_spike:
                res_currents[post_idx] += w_eff

        # 5. リザバーニューロンの更新
        for j, current in enumerate(res_currents):
            self.neurons_res[j].step(current, gating)

        # 6. リザバー -> 出力への伝達
        for pre_idx, post_idx, syn in self.synapses_res_to_out:
            pre_spike = self.neurons_res[pre_idx].spike
            w_eff = syn.update(self.current_time_ms, pre_spike)
            if pre_spike:
                out_currents[post_idx] += w_eff

        # 7. 出力ニューロンの更新
        out_results = []
        for k, current in enumerate(out_currents):
            spike = self.neurons_out[k].step(current, gating)
            out_results.append(spike)

        return out_results