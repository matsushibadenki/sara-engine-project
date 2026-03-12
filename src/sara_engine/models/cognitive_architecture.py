# パス: src/sara_engine/models/cognitive_architecture.py
# 英語タイトル: Cognitive Architecture
# 目的や内容: SNNベースのLiquid State Machineに、Spiking JEPA、乗算ゼロのSpike Attention、および皮質-海馬連動メモリ(CorticoHippocampalSystem)を統合した高効率・高精度な認知アーキテクチャ。行列演算、誤差逆伝播法、Numpyを一切排除して実装。

import random
from typing import List, Dict, Tuple, Set
from ..neuro.neuron import Neuron
from ..neuro.synapse_rl import RLSynapse
from ..cognitive.global_workspace import GlobalWorkspace
from .spiking_jepa import SpikingJEPA
from ..core.coincidence_attention import SpikeDrivenAttention
from ..core.cortex import CorticalColumn
from ..memory.hippocampus import CorticoHippocampalSystem

class CognitiveArchitecture:
    def __init__(self, n_sensory: int = 20, n_liquid: int = 100, n_actions: int = 4, memory_capacity: int = 1000000, supported_languages: List[str] = None):
        self.n_actions = n_actions
        self.supported_languages = supported_languages if supported_languages else ["en", "ja", "zh", "fr"]

        # Daleの法則に基づくSensory/Liquid/Actionニューロン初期化
        self.sensory = [Neuron(i, is_inhibitory=False, num_branches=1) for i in range(n_sensory)]
        self.liquid = [Neuron(i, is_inhibitory=(i % 5 == 0), num_branches=4) for i in range(n_liquid)]
        self.action_neurons = [Neuron(i, is_inhibitory=False, num_branches=4) for i in range(n_actions)]

        self.synapses: List[RLSynapse] = []

        # Sensory -> Liquid (樹状突起計算のためのランダムな枝番号割り当て)
        for pre_n in self.sensory:
            for post_n in self.liquid:
                if random.random() < 0.3:
                    b_idx = random.randint(0, len(post_n.branches) - 1)
                    self.synapses.append(RLSynapse(pre_n, post_n, b_idx, is_inhibitory=pre_n.is_inhibitory))

        # Liquid -> Liquid (Reservoir層の構築)
        for pre_n in self.liquid:
            for post_n in self.liquid:
                if pre_n.id != post_n.id and random.random() < 0.15:
                    b_idx = random.randint(0, len(post_n.branches) - 1)
                    self.synapses.append(RLSynapse(pre_n, post_n, b_idx, is_inhibitory=pre_n.is_inhibitory))

        # Liquid -> Action
        for pre_n in self.liquid:
            for post_n in self.action_neurons:
                if random.random() < 0.4:
                    b_idx = random.randint(0, len(post_n.branches) - 1)
                    self.synapses.append(RLSynapse(pre_n, post_n, b_idx, is_inhibitory=pre_n.is_inhibitory))

        self.workspace = GlobalWorkspace(num_candidates=n_actions)
        self.expected_reward = 0.0
        self.value_lr = 0.1

        # Active InferenceのためのSpiking JEPA
        self.world_model = SpikingJEPA(
            layer_configs=[{"input_dim": n_liquid, "embed_dim": n_liquid}], 
            learning_rate=0.05
        )
        self.last_surprise = 0.0
        self.prev_liquid_spikes: List[int] = []

        # 乗算ゼロの Spike-Driven Attention
        self.spike_attention = SpikeDrivenAttention(context_size=128, threshold=2.0)

        # [修正] 実在する CorticoHippocampalSystem を正しく初期化
        self.cortex = CorticalColumn()
        self.memory_system = CorticoHippocampalSystem(
            cortex=self.cortex,
            ltm_filepath="cognitive_arch_ltm.pkl",
            max_working_memory_size=15,
            snn_input_size=n_liquid
        )
        
    def _encode_language_context(self, lang: str) -> List[bool]:
        """多言語対応: 言語ごとのコンテキストをスパイクとしてSensory層に付与"""
        if lang not in self.supported_languages:
            lang = self.supported_languages[0]
        
        lang_hash = hash(lang) % max(1, len(self.sensory) // 4)
        lang_spikes = [False] * len(self.sensory)
        lang_spikes[lang_hash] = True
        lang_spikes[(lang_hash + 1) % len(self.sensory)] = True
        return lang_spikes

    def step_environment(self, sensory_input: List[bool], lang: str = "en") -> int:
        """最終認知ループ: Perception -> H-JEPA -> Attention -> Hippocampus -> Workspace -> Action"""
        # 1. Perception & Language Context
        lang_context = self._encode_language_context(lang)
        combined_sensory = [s or l for s, l in zip(sensory_input, lang_context)]
        
        for i, fired in enumerate(combined_sensory):
            if i < len(self.sensory):
                if fired:
                    self.sensory[i].branches[0].add_current(2.0)
                self.sensory[i].step()

        for syn in self.synapses:
            syn.step(dt=1.0)

        # 2. Reservoir dynamics
        current_liquid_spikes = []
        for i, n in enumerate(self.liquid):
            n.step()
            if n.spike:
                current_liquid_spikes.append(i)

        # 3. Active Inference via Spiking JEPA
        latent_spikes, surprise = self.world_model.forward(
            x_spikes=self.prev_liquid_spikes,
            y_spikes=current_liquid_spikes,
            learning=True
        )
        self.last_surprise = surprise
        self.prev_liquid_spikes = current_liquid_spikes

        # 4. Spike-Driven Attention
        current_set = set(current_liquid_spikes)
        attended_spikes_set = self.spike_attention.forward(q_spikes=current_set, k_spikes=current_set, v_spikes=current_set)
        attended_spikes = list(attended_spikes_set)

        # 5. Hippocampal Memory (CorticoHippocampalSystem) の参照と統合
        # 現在のスパイク状態(SDR)から過去の文脈を検索
        retrieved_contexts = self.memory_system.in_context_inference(
            current_sensory_sdr=attended_spikes, 
            context=lang
        )

        # 現在の体験を記憶層へ書き込み
        self.memory_system.experience_and_memorize(
            sensory_sdr=attended_spikes, 
            content="liquid_state_activation", 
            context=lang, 
            learning=True
        )

        # 検索された海馬記憶を用いてAction層を刺激し、文脈を強化
        for mem_dict in retrieved_contexts:
            mem_sdr = mem_dict.get('sdr', [])
            for mem_spike_idx in mem_sdr:
                if mem_spike_idx < len(self.action_neurons):
                    self.action_neurons[mem_spike_idx].branches[0].add_current(1.0)

        # 6. Action activity の収集
        action_potentials = []
        for n in self.action_neurons:
            n.step()
            action_potentials.append(n.v if n.v > 0 else 0.0)

        # 7. Workspace competition & Action selection
        selected_action = self.workspace.step(action_potentials)
        return selected_action

    def apply_reward(self, actual_reward: float):
        dopamine_delta = actual_reward + self.last_surprise - self.expected_reward
        self.expected_reward += self.value_lr * dopamine_delta
        
        for syn in self.synapses:
            syn.apply_dopamine(dopamine_delta=dopamine_delta, learning_rate=0.02)

    def consolidate_memory(self):
        print("[CognitiveArchitecture] Starting memory consolidation phase...")
        for syn in self.synapses:
            syn.consolidate()
        
        # 睡眠フェーズの模倣：言語コンテキストごとに記憶のリプレイ(固定化)を行う
        for lang in self.supported_languages:
            self.memory_system.consolidate_memories(context=lang, replay_count=5)
            
        print("[CognitiveArchitecture] Consolidation complete.")