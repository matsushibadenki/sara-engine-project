# ディレクトリパス: src/sara_engine/models/cognitive_architecture.py
# ファイルの日本語タイトル: 認知アーキテクチャ
# ファイルの目的や内容: SNNベースのLiquid State Machine、Global Workspace、Dopamine RLを統合した次世代脳型認知アーキテクチャの実装。行列演算および誤差逆伝播法を不使用。

# {
#     "//": "シナプス生成時にランダムな枝(branch_idx)を割り当てるように修正し、完全なパイプラインを確立"
# }

import random
from typing import List
from ..neuro.neuron import Neuron
from ..neuro.synapse_rl import RLSynapse
from ..cognitive.global_workspace import GlobalWorkspace
from .spiking_jepa import SpikingJEPA


class CognitiveArchitecture:
    def __init__(self, n_sensory: int = 20, n_liquid: int = 100, n_actions: int = 4):
        self.n_actions = n_actions

        # Daleの法則に基づき、Liquid層の20%を抑制性ニューロンとする (i % 5 == 0)
        self.sensory = [Neuron(i, is_inhibitory=False, num_branches=1)
                        for i in range(n_sensory)]
        self.liquid = [Neuron(i, is_inhibitory=(
            i % 5 == 0), num_branches=4) for i in range(n_liquid)]
        self.action_neurons = [
            Neuron(i, is_inhibitory=False, num_branches=4) for i in range(n_actions)]

        self.synapses: List[RLSynapse] = []

        # Sensory -> Liquid の結線 (樹状突起計算のためのランダムな枝番号割り当て)
        for pre_n in self.sensory:
            for post_n in self.liquid:
                if random.random() < 0.3:
                    b_idx = random.randint(
                        0, len(post_n.branches) - 1)
                    self.synapses.append(
                        RLSynapse(pre_n, post_n, b_idx, is_inhibitory=pre_n.is_inhibitory))

        # Liquid -> Liquid の再帰結合 (Reservoir層の構築)
        for pre_n in self.liquid:
            for post_n in self.liquid:
                if pre_n.id != post_n.id and random.random() < 0.15:
                    b_idx = random.randint(
                        0, len(post_n.branches) - 1)
                    self.synapses.append(
                        RLSynapse(pre_n, post_n, b_idx, is_inhibitory=pre_n.is_inhibitory))

        # Liquid -> Action の結線
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

        # [NEW] World Model for Active Inference
        # In Active Inference, the internal state (liquid) predicts the sensory input
        self.world_model = SpikingJEPA(
            layer_configs=[{"input_dim": n_liquid, "embed_dim": n_liquid}], # Internal auto-encoder/predictor
            learning_rate=0.05
        )
        # We need a separate predictor top-layer if we want to predict sensory directly,
        # but for surprise, we can compare JEPA's internal prediction with sensory-derived SDRs.
        # For simplicity, let's make JEPA layers match liquid, and we will use n_liquid as base.
        self.last_surprise = 0.0
        self.prev_liquid_spikes: List[int] = []

    def step_environment(self, sensory_input: List[bool]) -> int:
        """
        最終認知ループ: Perception -> Reservoir dynamics -> Workspace competition -> Action selection
        """
        # 1. Perception: 感覚入力の処理
        for i, fired in enumerate(sensory_input):
            if i < len(self.sensory):
                if fired:
                    self.sensory[i].branches[0].add_current(2.0)
                self.sensory[i].step()

        # シナプス経由のメッセージパッシング (イベント駆動計算)
        for syn in self.synapses:
            syn.step(dt=1.0)

        # 2. Reservoir dynamics: 内部状態空間の更新
        current_liquid_spikes = []
        for i, n in enumerate(self.liquid):
            n.step()
            if n.spike:
                current_liquid_spikes.append(i)

        # [NEW] Active Inference: Compute surprise from prediction error in internal state
        if self.prev_liquid_spikes and current_liquid_spikes:
            _, surprise = self.world_model.forward(
                x_spikes=self.prev_liquid_spikes,
                y_spikes=current_liquid_spikes,
                learning=True
            )
            self.last_surprise = surprise
        
        self.prev_liquid_spikes = current_liquid_spikes

        # 3. Action activity の収集
        action_potentials = []
        for n in self.action_neurons:
            n.step()
            action_potentials.append(n.v if n.v > 0 else 0.0)

        # 4 & 5. Workspace competition & Action selection
        selected_action = self.workspace.step(action_potentials)
        return selected_action

    def apply_reward(self, actual_reward: float):
        """
        Reward -> Dopamine -> Synaptic plasticity
        報酬予測誤差(RPE)にJEPA由来のサプライズを加算し、能動的推論を実現
        """
        # [MODIFIED] Include last_surprise for Active Inference
        dopamine_delta = actual_reward + self.last_surprise - self.expected_reward
        self.expected_reward += self.value_lr * dopamine_delta
        
        # 誤差逆伝播法を使わず、各シナプスが持つ資格トレース(Eligibility trace)とDopamine信号で更新
        for syn in self.synapses:
            syn.apply_dopamine(
                dopamine_delta=dopamine_delta, learning_rate=0.02)

    def consolidate_memory(self):
        """
        [NEW] Hippocampal Consolidation: Transfers short-term knowledge to long-term structures.
        Simulates "sleep" or reflection phases by stabilizing synapses.
        """
        print("[CognitiveArchitecture] Starting memory consolidation phase...")
        for syn in self.synapses:
            syn.consolidate()
        print("[CognitiveArchitecture] Consolidation complete.")