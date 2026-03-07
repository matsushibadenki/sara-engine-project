# {
#     "//": "ディレクトリパス: src/sara_engine/neuro/synapse.py",
#     "//": "ファイルの日本語タイトル: 生体模倣シナプス (修正版)",
#     "//": "ファイルの目的や内容: プレニューロンの発火時に、ポストニューロンの DendriticTree に正しく電流を流し込むようにAPIを修正。"
# }

import random
import math
from typing import Any


class Synapse:
    def __init__(self, pre_neuron: Any, post_neuron: Any, post_branch_idx: int, is_inhibitory: bool = False):
        self.pre = pre_neuron
        self.post = post_neuron
        self.post_branch_idx = post_branch_idx
        self.is_inhibitory = is_inhibitory

        base_w = random.uniform(0.01, 0.5)
        self.weight = -base_w if self.is_inhibitory else base_w

        self.U = 0.2
        self.tau_f = 100.0 if self.is_inhibitory else 600.0
        self.tau_d = 200.0
        self.u = self.U
        self.x = 1.0

        self.pre_trace = 0.0
        self.post_trace = 0.0
        self.tau_pre = 20.0
        self.tau_post = 20.0
        self.A_plus = 0.01
        self.A_minus = 0.012

    def step(self, dt: float = 1.0) -> float:
        self.u += (self.U - self.u) * dt / self.tau_f
        self.x += (1.0 - self.x) * dt / self.tau_d
        self.pre_trace *= math.exp(-dt / self.tau_pre)
        self.post_trace *= math.exp(-dt / self.tau_post)

        if getattr(self.pre, 'spike', False):
            self.u += self.U * (1.0 - self.u)
            self.x *= (1.0 - self.u)
            self.pre_trace += 1.0

            dw = self.A_plus * self.post_trace
            if self.is_inhibitory:
                self.weight -= dw
            else:
                self.weight += dw

            # 実効電流を計算し、対象の枝へ注入 (API適合)
            current = self.weight * self.u * self.x
            if hasattr(self.post, 'dendritic_tree'):
                self.post.dendritic_tree.integrate_to_branch(
                    self.post_branch_idx, current)

        if getattr(self.post, 'spike', False):
            self.post_trace += 1.0
            dw = self.A_minus * self.pre_trace
            if self.is_inhibitory:
                self.weight += dw
            else:
                self.weight -= dw

        if self.is_inhibitory:
            self.weight = max(-2.0, min(0.0, self.weight))
        else:
            self.weight = max(0.0, min(2.0, self.weight))

        return 0.0
