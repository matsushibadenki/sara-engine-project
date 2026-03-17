# Directory Path: src/sara_engine/core/cortex.py
# English Title: Cortex Layer with Structural Plasticity
# Purpose/Content: 毎ステップのランダム接続計算を廃止し、構造的可塑性(Structural Plasticity)を備えた明示的な動的スパース結合を導入。イベント駆動化により計算量を劇的に削減し、破滅的忘却を防ぐコンテキストルーティングを強化。多言語対応。

import random
from typing import List, Dict, Set

class CortexLayer:
    """
    構造的可塑性(Structural Plasticity)を備えた動的コンパートメント。
    事前定義されたスパースなシナプスのみをイベント駆動で計算し、
    活動状態に応じてシナプスの刈り込み(Pruning)と新生(Regeneration)を行う。
    """
    def __init__(self, input_size: int, hidden_size: int, layer_type: str = "L2/3"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_type = layer_type
        
        if layer_type == "L2/3":
            decay = 0.95
            density = 0.05
        elif layer_type == "L4":
            decay = 0.8
            density = 0.1
        elif layer_type == "L5/6":
            decay = 0.98
            density = 0.02
        else:
            decay = 0.9
            density = 0.05
            
        self.decay = decay
        self.target_density = density
        self.v: List[float] = [0.0] * hidden_size
        self.activity_ema: List[float] = [0.0] * hidden_size
        self.target_rate = 0.02
        self.dynamic_thresh: List[float] = [1.0] * hidden_size
        self.base_thresh = 1.0
        
        # O(1)アクセス可能なスパースシナプス結合 [プレニューロン -> {ポストニューロン: 重み}]
        self.synapses: Dict[int, Dict[int, float]] = {i: {} for i in range(input_size)}
        self._initialize_topology()

    def get_status(self, lang: str = "en") -> str:
        messages = {
            "en": f"CortexLayer {self.layer_type}: Structural Plasticity active. Target Density {self.target_density}",
            "ja": f"大脳皮質層 {self.layer_type}: 構造的可塑性が有効。目標結合密度 {self.target_density}",
            "fr": f"Couche Cortex {self.layer_type}: Plasticité structurelle active. Densité cible {self.target_density}"
        }
        return messages.get(lang, messages["en"])

    def _initialize_topology(self):
        for i in range(self.input_size):
            num_targets = max(1, int(self.hidden_size * self.target_density))
            targets = random.sample(range(self.hidden_size), num_targets)
            for t in targets:
                self.synapses[i][t] = random.uniform(0.5, 1.0)

    def forward(self, input_indices: List[int], learning: bool = True) -> List[int]:
        fired_indices = []
        active_targets = set()
        
        # 1. イベント駆動型フォワードパス: 入力があったスパース接続のみ計算
        for i in input_indices:
            if i in self.synapses:
                for target, weight in self.synapses[i].items():
                    self.v[target] += weight
                    active_targets.add(target)
                    
        # 2. 膜電位の減衰と発火判定
        for i in range(self.hidden_size):
            if self.v[i] > 0.0:
                self.v[i] *= self.decay
                if self.v[i] >= self.dynamic_thresh[i]:
                    fired_indices.append(i)
                    self.v[i] = 0.0
                elif self.v[i] < 0.01:
                    self.v[i] = 0.0
                    
        # 3. ホメオスタシスと構造的可塑性 (学習時)
        if learning:
            ema_decay = 0.1
            fired_set = set(fired_indices)
            
            for i in range(self.hidden_size):
                is_fired = 1.0 if i in fired_set else 0.0
                self.activity_ema[i] = (1 - ema_decay) * self.activity_ema[i] + ema_decay * is_fired
                
                diff = self.activity_ema[i] - self.target_rate
                self.dynamic_thresh[i] += diff * 0.1
                if self.dynamic_thresh[i] < self.base_thresh:
                    self.dynamic_thresh[i] = self.base_thresh

            # 計算負荷軽減のため5%の確率でシナプスの刈り込みと新生を実行
            if random.random() < 0.05:
                self._structural_plasticity(input_indices, fired_set)

        return fired_indices

    def _structural_plasticity(self, active_inputs: List[int], fired_outputs: Set[int]):
        for pre in active_inputs:
            if pre in self.synapses:
                targets = list(self.synapses[pre].keys())
                for target in targets:
                    if target not in fired_outputs:
                        # LTDと刈り込み
                        self.synapses[pre][target] -= 0.01
                        if self.synapses[pre][target] < 0.1:
                            del self.synapses[pre][target]
                    else:
                        # LTP
                        self.synapses[pre][target] = min(2.0, self.synapses[pre][target] + 0.05)
                        
                # 結合の新生 (Rewiring)
                current_fanout = len(self.synapses[pre])
                target_fanout = max(1, int(self.hidden_size * self.target_density))
                if current_fanout < target_fanout and fired_outputs:
                    new_target = random.choice(list(fired_outputs))
                    if new_target not in self.synapses[pre]:
                        self.synapses[pre][new_target] = 0.5

    def reset_state(self):
        self.v = [0.0] * self.hidden_size


class CorticalColumn:
    """
    複数のコンパートメント（サブネットワーク）を持ち、コンテキストに応じて
    発火経路を切り替える大脳皮質カラムのモデル。
    破滅的忘却を防ぐためのモジュール化された構造を持つ。
    """
    def __init__(self, input_size: int, hidden_size_per_comp: int, compartment_names: List[str], target_rate: float = 0.05):
        self.input_size = input_size
        self.hidden_size_per_comp = hidden_size_per_comp
        self.compartments: Dict[str, CortexLayer] = {}
        
        for name in compartment_names:
            layer = CortexLayer(input_size=input_size, hidden_size=hidden_size_per_comp)
            layer.target_rate = target_rate
            self.compartments[name] = layer
            
    def get_status(self, lang: str = "en") -> str:
        messages = {
            "en": f"CorticalColumn active with {len(self.compartments)} compartments.",
            "ja": f"皮質カラムは {len(self.compartments)} 個のコンパートメントでアクティブです。",
            "fr": f"Colonne corticale active avec {len(self.compartments)} compartiments."
        }
        return messages.get(lang, messages["en"])

    def forward_latent_chain(self, active_inputs: List[int], prev_active_hidden: List[int], 
                             current_context: str, learning: bool = False, 
                             reward_signal: float = 0.0) -> List[int]:
        if current_context not in self.compartments:
            return []
            
        target_layer = self.compartments[current_context]
        fired_indices = target_layer.forward(active_inputs, learning=learning)
        
        return fired_indices

    def reset_short_term_memory(self):
        for layer in self.compartments.values():
            layer.reset_state()
        
    def get_compartment_states(self) -> Dict[str, Dict[str, int]]:
        states = {}
        for name, layer in self.compartments.items():
            active_count = sum(1 for v in layer.v if v > 0.0)
            states[name] = {
                "active_neurons": active_count
            }
        return states