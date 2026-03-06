# {
#     "//": "ディレクトリパス: src/sara_engine/models/cognitive_architecture.py",
#     "//": "ファイルの日本語タイトル: 認知アーキテクチャ (引数修正版)",
#     "//": "ファイルの目的や内容: シナプス生成時にランダムな枝(branch_idx)を割り当てるように修正し、完全なパイプラインを確立。"
# }

import random
from typing import List, Dict
from ..neuro.neuron import Neuron
from ..neuro.synapse_rl import RLSynapse
from ..cognitive.global_workspace import GlobalWorkspace


class CognitiveArchitecture:
    def __init__(self, n_sensory: int = 20, n_liquid: int = 100, n_actions: int = 4):
        self.n_actions = n_actions

        self.sensory = [Neuron(i, is_inhibitory=False, num_branches=1)
                        for i in range(n_sensory)]
        self.liquid = [Neuron(i, is_inhibitory=(
            i % 5 == 0), num_branches=4) for i in range(n_liquid)]
        self.action_neurons = [
            Neuron(i, is_inhibitory=False, num_branches=4) for i in range(n_actions)]

        self.synapses: List[RLSynapse] = []

        # 結線の際にランダムな枝番号を渡す
        for pre_n in self.sensory:
            for post_n in self.liquid:
                if random.random() < 0.3:
                    b_idx = random.randint(
                        0, len(post_n.branches) - 1)
                    self.synapses.append(
                        RLSynapse(pre_n, post_n, b_idx, is_inhibitory=pre_n.is_inhibitory))

        for pre_n in self.liquid:
            for post_n in self.liquid:
                if pre_n.id != post_n.id and random.random() < 0.15:
                    b_idx = random.randint(
                        0, len(post_n.branches) - 1)
                    self.synapses.append(
                        RLSynapse(pre_n, post_n, b_idx, is_inhibitory=pre_n.is_inhibitory))

        for pre_n in self.liquid:
            for post_n in self.action_neurons:
                if random.random() < 0.4:
                    b_idx = random.randint(
                        0, len(post_n.branches) - 1)
                    self.synapses.append(
                        RLSynapse(pre_n, post_n, b_idx, is_inhibitory=pre_n.is_inhibitory))

        self.workspace = GlobalWorkspace(num_candidates=n_actions)
        self.expected_reward = 0.0
        self.value_lr = 0.1

    def step_environment(self, sensory_input: List[bool]) -> int:
        for i, fired in enumerate(sensory_input):
            if i < len(self.sensory):
                # 感覚ニューロンの枝に電流を注入して発火させる
                if fired:
                    self.sensory[i].branches[0].add_current(2.0)
                self.sensory[i].step()

        # シナプス経由のメッセージパッシング
        for syn in self.synapses:
            syn.step(dt=1.0)

        for n in self.liquid:
            n.step()

        action_potentials = []
        for n in self.action_neurons:
            n.step()
            action_potentials.append(n.v if n.v > 0 else 0.0)

        selected_action = self.workspace.step(action_potentials)
        return selected_action

    def apply_reward(self, actual_reward: float):
        dopamine_delta = actual_reward - self.expected_reward
        self.expected_reward += self.value_lr * dopamine_delta
        for syn in self.synapses:
            syn.apply_dopamine(
                dopamine_delta=dopamine_delta, learning_rate=0.02)
