# Directory Path: src/sara_engine/dynamics/oscillation.py
# English Title: Neural Oscillations and Event-Driven STP Dynamics
# Purpose/Content: 脳波リズムのLUT（ルックアップテーブル）による高速化と、スパース隣接リストを用いた完全イベント駆動型の短期可塑性(STP)。非発火時の無駄な演算を完全にスキップし、超低消費電力を実現する。多言語対応。

import math
import random
from typing import Dict, List, Tuple

class OscillationManager:
    """
    神経振動（Neural Oscillations）による時間的ゲーティングの管理。
    計算負荷を下げるため Fast Sine LUT (Look-Up Table) を使用。
    """

    def __init__(self):
        self.frequencies = {
            "theta": 6.0,   
            "alpha": 10.0,  
            "beta":  20.0,  
            "gamma": 40.0   
        }
        # 1000分割の高速参照用LUT
        self._sin_lut = [math.sin(2 * math.pi * i / 1000.0) for i in range(1000)]

    def get_status(self, lang: str = "en") -> str:
        """多言語対応ステータス出力"""
        messages = {
            "en": "OscillationManager: Fast Sine LUT active.",
            "ja": "OscillationManager: 高速サイン波LUTが有効です。",
            "fr": "OscillationManager: LUT sinus rapide actif."
        }
        return messages.get(lang, messages["en"])

    def _fast_sin(self, freq: float, t_sec: float) -> float:
        """LUTを用いた O(1) のサイン波計算"""
        idx = int((t_sec * freq * 1000.0)) % 1000
        return self._sin_lut[idx]

    def get_phase_effects(self, current_time_ms: float) -> Dict[str, float]:
        t_sec = current_time_ms / 1000.0
        effects = {}
        for name, freq in self.frequencies.items():
            effects[name] = self._fast_sin(freq, t_sec)
        return effects

    def get_gating_factor(self, current_time_ms: float, mode: str = "theta_gamma") -> float:
        t_sec = current_time_ms / 1000.0
        if mode == "theta_gamma":
            theta = self._fast_sin(self.frequencies["theta"], t_sec)
            gamma = self._fast_sin(self.frequencies["gamma"], t_sec)
            return (theta * 0.5) + (gamma * 0.2)
        return self._fast_sin(self.frequencies["alpha"], t_sec)


class STPSynapse:
    """
    短期シナプス可塑性（Short-Term Plasticity）。
    完全なイベント駆動型に最適化され、スパイク到達時のみ時間を進めて状態を更新する。
    """

    def __init__(
        self,
        weight: float,
        U: float = 0.2,
        tau_f: float = 600.0,
        tau_d: float = 200.0
    ):
        self.w = weight  
        self.U = U       
        self.tau_f = tau_f  
        self.tau_d = tau_d  

        self.u = self.U  
        self.x = 1.0     
        self.last_update_t = 0.0

    def compute_on_event(self, current_time: float) -> float:
        """
        プレニューロンが発火した時のみ呼び出される O(1) 更新メソッド。
        発火しなかった期間の減衰を一括で計算し、直ちに新しい重みを返す。
        """
        dt = current_time - self.last_update_t

        if dt > 0:
            # 簡略化された指数減衰近似により math.exp の負荷を軽減
            exp_f = 1.0 / (1.0 + dt / self.tau_f)
            exp_d = 1.0 / (1.0 + dt / self.tau_d)
            self.u = self.U + (self.u - self.U) * exp_f
            self.x = 1.0 + (self.x - 1.0) * exp_d

        effective_w = self.w * self.u * self.x

        self.u += self.U * (1.0 - self.u)
        self.x *= (1.0 - self.u)
        self.last_update_t = current_time
        
        return effective_w


class LIFNeuron:
    """神経振動による動的閾値制御付きLIFニューロン"""

    def __init__(self, base_threshold: float = 1.0, leak: float = 0.95):
        self.v = 0.0
        self.base_threshold = base_threshold
        self.leak = leak
        self.spike = False
        self.refractory_period = 0  

    def step(self, input_current: float, gating_factor: float = 0.0):
        if self.refractory_period > 0:
            self.refractory_period -= 1
            self.v = 0.0
            self.spike = False
            return False

        if input_current == 0.0 and self.v < 0.01:
            self.v = 0.0
            self.spike = False
            return False

        self.v = (self.v + input_current) * self.leak
        dynamic_threshold = self.base_threshold + (gating_factor * 0.3)

        if self.v > dynamic_threshold:
            self.spike = True
            self.v = 0.0
            self.refractory_period = 2  
        else:
            self.spike = False

        return self.spike


class DynamicSpikingNetwork:
    """
    イベント駆動によるスパース隣接リストを採用した超省エネ・自律学習型リザバー。
    全シナプスをループする仕様を廃止。
    """

    def __init__(self, n_in: int, n_res: int, n_out: int):
        self.oscillation = OscillationManager()
        self.neurons_in = [LIFNeuron() for _ in range(n_in)]
        self.neurons_res = [LIFNeuron(base_threshold=1.2) for _ in range(n_res)]
        self.neurons_out = [LIFNeuron() for _ in range(n_out)]

        # スパース隣接リスト構造 {pre_id: [(post_id, syn_obj), ...]}
        self.adj_in_to_res: Dict[int, List[Tuple[int, STPSynapse]]] = {i: [] for i in range(n_in)}
        self.adj_res_recurrent: Dict[int, List[Tuple[int, STPSynapse]]] = {i: [] for i in range(n_res)}
        self.adj_res_to_out: Dict[int, List[Tuple[int, STPSynapse]]] = {i: [] for i in range(n_res)}

        self._init_synapses(n_in, n_res, n_out)
        self.current_time_ms = 0.0

    def _init_synapses(self, n_in, n_res, n_out):
        for i in range(n_in):
            targets = random.sample(range(n_res), max(1, int(n_res * 0.3)))
            for j in targets:
                self.adj_in_to_res[i].append((j, STPSynapse(random.uniform(0.1, 0.5))))

        for i in range(n_res):
            targets = random.sample(range(n_res), max(1, int(n_res * 0.1)))
            for j in targets:
                if i != j:
                    self.adj_res_recurrent[i].append((j, STPSynapse(random.uniform(-0.2, 0.4))))

        for i in range(n_res):
            targets = random.sample(range(n_out), max(1, int(n_out * 0.5)))
            for j in targets:
                self.adj_res_to_out[i].append((j, STPSynapse(random.uniform(0.0, 0.3))))

    def step(self, input_signals: List[bool]) -> List[bool]:
        self.current_time_ms += 1.0
        gating = self.oscillation.get_gating_factor(self.current_time_ms)

        for i, fired in enumerate(input_signals):
            self.neurons_in[i].step(1.5 if fired else 0.0, gating)

        res_currents = [0.0] * len(self.neurons_res)
        out_currents = [0.0] * len(self.neurons_out)

        # イベント駆動: 発火したニューロンの枝のみを計算 (劇的な省エネ)
        for pre_idx, n_in in enumerate(self.neurons_in):
            if n_in.spike:
                for post_idx, syn in self.adj_in_to_res[pre_idx]:
                    res_currents[post_idx] += syn.compute_on_event(self.current_time_ms)

        res_spikes_prev = [n.spike for n in self.neurons_res]
        for pre_idx, spiked in enumerate(res_spikes_prev):
            if spiked:
                for post_idx, syn in self.adj_res_recurrent[pre_idx]:
                    res_currents[post_idx] += syn.compute_on_event(self.current_time_ms)

        for j, current in enumerate(res_currents):
            self.neurons_res[j].step(current, gating)

        for pre_idx, n_res in enumerate(self.neurons_res):
            if n_res.spike:
                for post_idx, syn in self.adj_res_to_out[pre_idx]:
                    out_currents[post_idx] += syn.compute_on_event(self.current_time_ms)

        out_results = []
        for k, current in enumerate(out_currents):
            spike = self.neurons_out[k].step(current, gating)
            out_results.append(spike)

        return out_results