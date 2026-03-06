# {
#     "//": "ディレクトリパス: src/sara_engine/neuro/synapse_rl.py",
#     "//": "ファイルの日本語タイトル: 強化学習対応シナプス (修正版)",
#     "//": "ファイルの目的や内容: 枝(Branch)の概念を追加し、DendriticTree APIに適合するように修正。"
# }

import math
import random
from typing import Any


class RLSynapse:
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

        self.eligibility_trace = 0.0
        self.tau_e = 1000.0

    def step(self, dt: float = 1.0) -> float:
        self.u += (self.U - self.u) * dt / self.tau_f
        self.x += (1.0 - self.x) * dt / self.tau_d
        self.eligibility_trace *= math.exp(-dt / self.tau_e)

        pre_spike = getattr(self.pre, 'spike', False)
        post_spike = getattr(self.post, 'spike', False)

        if pre_spike:
            self.u += self.U * (1.0 - self.u)
            self.x *= (1.0 - self.u)

            # API適合: 対象の枝へ注入
            current = self.weight * self.u * self.x
            if hasattr(self.post, 'branches') and self.post_branch_idx < len(self.post.branches):
                self.post.branches[self.post_branch_idx].add_current(current)

        if post_spike and getattr(self.pre, 'v', 0.0) > 0.1:
            self.eligibility_trace += 1.0
        elif pre_spike and getattr(self.post, 'v', 0.0) > 0.1:
            self.eligibility_trace += 0.5

        if self.is_inhibitory:
            self.weight = max(-2.0, min(0.0, self.weight))
        else:
            self.weight = max(0.0, min(2.0, self.weight))

        return 0.0

    def apply_dopamine(self, dopamine_delta: float, learning_rate: float = 0.01):
        dw = learning_rate * dopamine_delta * self.eligibility_trace
        if self.is_inhibitory:
            self.weight -= dw
            self.weight = max(-2.0, min(0.0, self.weight))
        else:
            self.weight += dw
            self.weight = max(0.0, min(2.0, self.weight))
