# {
#     "//": "ディレクトリパス: src/sara_engine/models/lsm_network.py",
#     "//": "ファイルの日本語タイトル: Liquid State Machine ネットワーク",
#     "//": "ファイルの目的や内容: オブジェクト指向のLSM。樹状突起ネットワークによる空間計算と、STPによる時間計算を統合。純粋なメッセージパッシングで $O(N)$ の動的システムを駆動する。"
# }

import random
from typing import List
from ..neuro.neuron import Neuron
from ..neuro.synapse import Synapse


class LSMNetwork:
    """
    樹状突起計算(Dendritic Computation) と メタ可塑性 を備えた Liquid Transformer 代替コア。
    """

    def __init__(self, n_input: int = 50, n_liquid: int = 500, n_output: int = 100, branches_per_neuron: int = 4):
        self.inputs = [Neuron(i, is_inhibitory=False, num_branches=1)
                       for i in range(n_input)]
        self.liquid = [Neuron(i, is_inhibitory=(
            i % 5 == 0), num_branches=branches_per_neuron) for i in range(n_liquid)]
        self.outputs = [Neuron(
            i, is_inhibitory=False, num_branches=branches_per_neuron) for i in range(n_output)]

        self.synapses: List[Synapse] = []
        self.global_step = 0

        self.target_rate = 0.05
        self.firing_rates = {
            n: self.target_rate for n in self.liquid + self.outputs}

        self.prune_threshold = 0.01
        self.growth_prob = 0.1

        # ネットワークの配線 (枝をランダムに指定)
        self._wire_population(self.inputs, self.liquid, prob=0.2)
        self._wire_population(self.liquid, self.liquid,
                              prob=0.1, avoid_self=True)
        self._wire_population(self.liquid, self.outputs, prob=0.5)

    def _wire_population(self, pre_pop: List[Neuron], post_pop: List[Neuron], prob: float, avoid_self: bool = False):
        for pre_n in pre_pop:
            for post_n in post_pop:
                if avoid_self and pre_n.id == post_n.id:
                    continue
                if random.random() < prob:
                    # ランダムな枝に接続 (シナプスクラスタリングの初期状態)
                    b_idx = random.randint(0, len(post_n.branches) - 1)
                    self.synapses.append(
                        Synapse(pre_n, post_n, b_idx, is_inhibitory=pre_n.is_inhibitory))

    def step(self, input_spikes: List[bool]) -> List[bool]:
        """
        1ステップのシミュレーション。行列演算 O(N^2) を使わず、
        スパイクが発生したシナプスのみが信号を伝播するグラフ処理ライクな O(E) の計算。
        """
        self.global_step += 1

        # 1. 外部入力の強制発火
        for i, fired in enumerate(input_spikes):
            if i < len(self.inputs):
                self.inputs[i].spike = fired

        # 2. シナプス経由の電流伝達 (純粋なメッセージパッシング)
        # 各シナプスが pre の発火を確認し、post の特定の枝へ電流を注入する
        for syn in self.synapses:
            syn.step(dt=1.0)

        # 3. ニューロン状態の更新 (Dendrite統合とSoma発火)
        for n in self.liquid:
            n.step()
            self.firing_rates[n] = self.firing_rates[n] * \
                0.99 + (1.0 if n.spike else 0.0) * 0.01

        # 4. 出力層の更新
        out_spikes = []
        for n in self.outputs:
            spike = n.step()
            out_spikes.append(spike)
            self.firing_rates[n] = self.firing_rates[n] * \
                0.99 + (1.0 if n.spike else 0.0) * 0.01

        # メタ可塑性の適用
        if self.global_step % 100 == 0:
            self._apply_homeostasis()

        if self.global_step % 1000 == 0:
            self._apply_structural_plasticity()

        return out_spikes

    def _apply_homeostasis(self):
        scale_factors = {}
        for n, rate in self.firing_rates.items():
            scale = self.target_rate / (rate + 1e-6)
            scale_factors[n] = max(0.8, min(scale, 1.2))

        for syn in self.synapses:
            if syn.post in scale_factors:
                syn.weight *= scale_factors[syn.post]
                if syn.is_inhibitory:
                    syn.weight = max(-2.0, min(0.0, syn.weight))
                else:
                    syn.weight = max(0.0, min(2.0, syn.weight))

    def _apply_structural_plasticity(self):
        self.synapses = [s for s in self.synapses if abs(
            s.weight) > self.prune_threshold]

        if random.random() < self.growth_prob:
            pre_n = random.choice(self.liquid)
            post_n = random.choice(self.liquid)
            if pre_n.id != post_n.id:
                # Synaptic Clustering: 他の強いシナプスがある枝を優先的に選ぶことで
                # 特徴検出器（ANDゲート）としての能力を高めることが可能（将来の拡張）
                b_idx = random.randint(0, len(post_n.branches) - 1)
                self.synapses.append(
                    Synapse(pre_n, post_n, b_idx, is_inhibitory=pre_n.is_inhibitory))
