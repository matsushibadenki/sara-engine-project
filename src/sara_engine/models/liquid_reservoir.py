# {
#     "//": "ディレクトリパス: src/sara_engine/models/liquid_reservoir.py",
#     "//": "ファイルの日本語タイトル: Liquid Reservoir (樹状突起計算対応版)",
#     "//": "ファイルの目的や内容: Izhikevichニューロンに樹状突起(Dendrite)コンパートメントを導入。シナプス入力を枝ごとに非線形統合することで、単一ニューロンの表現力をMLPレベルに引き上げる。"
# }

import random
import math
from typing import List, Dict, Any, Tuple
from ..neuro.dendrite import DendriticTree
from ..metrics.branching_ratio import BranchingRatioEstimator


class LiquidReservoir:
    def __init__(
        self,
        n_neurons: int = 200,
        p_connect: float = 0.1,
        dt: float = 1.0,
        max_weight: float = 2.0,
        max_delay_limit: int = 50
    ):
        self.n = n_neurons
        self.dt = dt
        self.max_weight = max_weight
        self.max_delay_limit = max_delay_limit
        self.current_time = 0.0

        # ニューロンごとの樹状突起ツリーを初期化
        self.dendritic_trees = [DendriticTree(
            num_branches=4) for _ in range(n_neurons)]

        self.is_inhibitory = [(i % 5 == 0) for i in range(n_neurons)]
        self.synapses: List[Dict[int, float]] = [{} for _ in range(n_neurons)]

        self.buffer_size = int(max_delay_limit / dt) + 1
        # arrival_buffer 内に「どの pre_id からの信号か」を保持するように拡張
        self.arrival_buffer: List[List[List[Tuple[int, float]]]] = [
            [[] for _ in range(n_neurons)] for _ in range(self.buffer_size)
        ]

        for pre in range(n_neurons):
            for post in range(n_neurons):
                if pre != post and random.random() < p_connect:
                    w = random.uniform(0.1, 0.5)
                    if self.is_inhibitory[pre]:
                        w = -abs(w)
                    else:
                        w = abs(w)
                    self.synapses[pre][post] = w

        self.U_base = 0.2
        self.tau_f, self.tau_d = 600.0, 200.0
        self.u, self.x = [self.U_base] * n_neurons, [1.0] * n_neurons
        self.pre_trace, self.post_trace = [0.0] * n_neurons, [0.0] * n_neurons
        self.tau_pre, self.tau_post = 20.0, 20.0
        self.A_plus, self.A_minus = 0.01, 0.012

        self.v = [-65.0 for _ in range(n_neurons)]
        self.u_izh = [0.2 * val for val in self.v]
        self.a = [0.1 if self.is_inhibitory[i]
                  else 0.02 for i in range(n_neurons)]
        self.b = [0.2 for _ in range(n_neurons)]
        self.c = [-65.0 for _ in range(n_neurons)]
        self.d = [2.0 if self.is_inhibitory[i]
                  else 8.0 for i in range(n_neurons)]
        self.v_thresh = [30.0 for _ in range(n_neurons)]
        self.branching_estimator = BranchingRatioEstimator(
            smoothing_alpha=0.08)
        self.target_sigma = 1.0
        self.critical_gain = 1.0

    def step(self, external_currents: List[float], delay_manager: Any = None) -> List[int]:
        self.current_time += self.dt
        current_slot = int((self.current_time / self.dt) % self.buffer_size)

        # 1. 減衰処理
        for i in range(self.n):
            self.u[i] += (-self.u[i] / self.tau_f) * self.dt
            self.x[i] += ((1.0 - self.x[i]) / self.tau_d) * self.dt
            self.pre_trace[i] *= math.exp(-self.dt / self.tau_pre)
            self.post_trace[i] *= math.exp(-self.dt / self.tau_post)

        # 2. 樹状突起による非線形統合
        # このステップで届いた信号(pre_id, effective_weight_base)を各ニューロンの枝に流し込む
        current_arrivals = self.arrival_buffer[current_slot]
        I_syn_total = [0.0] * self.n

        for post_id in range(self.n):
            for pre_id, eff_w in current_arrivals[post_id]:
                # 各ニューロンの樹状突起が信号を枝ごとに受容
                self.dendritic_trees[post_id].integrate_synapse(
                    pre_id, eff_w, 1.0)

            # 細胞体(Soma)に届く総電流を計算（ここで非線形なDendritic Spikeが発生）
            I_syn_total[post_id] = self.dendritic_trees[post_id].aggregate_to_soma()

        # 3. Izhikevich 更新と発火判定
        fired_neurons = []
        for i in range(self.n):
            I_ext = external_currents[i] if i < len(external_currents) else 0.0
            I_total = (I_ext + I_syn_total[i]) * self.critical_gain

            v, u_i = self.v[i], self.u_izh[i]
            v_new = v + self.dt * (0.04 * v**2 + 5.0 *
                                   v + 140.0 - u_i + I_total)
            u_new = u_i + self.dt * self.a[i] * (self.b[i] * v - u_i)

            if v_new >= self.v_thresh[i]:
                fired_neurons.append(i)
                self.v[i], self.u_izh[i] = self.c[i], u_new + self.d[i]
                if delay_manager:
                    delay_manager.update_delays(i, self.current_time)
            else:
                self.v[i], self.u_izh[i] = v_new, u_new

        # 使用済みスロットのクリア
        self.arrival_buffer[current_slot] = [[] for _ in range(self.n)]

        # 4. 発火による未来へのスパイク予約
        for pre_id in fired_neurons:
            u_pre = self.u[pre_id]
            u_pre += self.U_base * (1.0 - u_pre)
            self.x[pre_id] *= (1.0 - u_pre)
            self.u[pre_id] = u_pre
            self.pre_trace[pre_id] += 1.0

            w_eff_base = u_pre * self.x[pre_id]
            for post_id, w in self.synapses[pre_id].items():
                d = delay_manager.get_delay(
                    pre_id, post_id) if delay_manager else 5.0
                arrival_time = self.current_time + d
                target_slot = int((arrival_time / self.dt) % self.buffer_size)

                # 未来のバッファに (発火元ID, 実効的な重み) を予約
                self.arrival_buffer[target_slot][post_id].append(
                    (pre_id, w * w_eff_base))

                if delay_manager:
                    delay_manager.record_arrival(pre_id, post_id, arrival_time)
                self._update_weight(pre_id, post_id, -
                                    self.A_minus * self.post_trace[post_id])

        for post_id in fired_neurons:
            self.post_trace[post_id] += 1.0
            for pre_id in range(self.n):
                if post_id in self.synapses[pre_id]:
                    self._update_weight(
                        pre_id, post_id, self.A_plus * self.pre_trace[pre_id])

        sigma = self.branching_estimator.update(len(fired_neurons))
        self._apply_criticality_control(sigma)

        return fired_neurons

    def _apply_criticality_control(self, sigma: float) -> None:
        sigma_error = sigma - self.target_sigma
        self.critical_gain = max(0.7, min(1.3, 1.0 - sigma_error * 0.15))

        if abs(sigma_error) < 0.05:
            return

        excit_scale = max(0.92, min(1.08, 1.0 - sigma_error * 0.05))
        inhib_scale = max(0.92, min(1.08, 1.0 + sigma_error * 0.05))
        for pre_id in range(self.n):
            if self.is_inhibitory[pre_id]:
                scale = inhib_scale
            else:
                scale = excit_scale
            for post_id in list(self.synapses[pre_id].keys()):
                self.synapses[pre_id][post_id] *= scale
                if self.is_inhibitory[pre_id]:
                    self.synapses[pre_id][post_id] = max(
                        -self.max_weight, min(0.0, self.synapses[pre_id][post_id]))
                else:
                    self.synapses[pre_id][post_id] = max(
                        0.0, min(self.max_weight, self.synapses[pre_id][post_id]))

    def _update_weight(self, pre_id: int, post_id: int, dw: float):
        w = self.synapses[pre_id][post_id]
        if self.is_inhibitory[pre_id]:
            w = max(-self.max_weight, min(0.0, w - dw))
        else:
            w = max(0.0, min(self.max_weight, w + dw))
        self.synapses[pre_id][post_id] = w
