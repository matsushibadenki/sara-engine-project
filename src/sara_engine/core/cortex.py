_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/cortex.py",
    "//": "タイトル: 皮質カラム (Cortical Column) と神経修飾",
    "//": "目的: コンパートメント化された局所学習と、SDRベースの潜在空間協調(Latent Chain)を実現する。"
}

import numpy as np
from typing import List, Dict, Optional
from .layers import DynamicLiquidLayer

class CorticalColumn:
    """
    大脳皮質カラムを模倣したコンパートメント化レイヤー群。
    文脈(Context)に応じて特定のDynamicLiquidLayerのみを活性化・学習させ、
    破滅的忘却を防ぎながらSDR(スパース発火インデックス)を次へ連鎖させる。
    """
    def __init__(self, input_size: int, hidden_size_per_comp: int, 
                 compartment_names: List[str], target_rate: float = 0.05):
        self.input_size = input_size
        self.hidden_size = hidden_size_per_comp
        self.compartments: Dict[str, DynamicLiquidLayer] = {}
        
        self.compartments["core_shared"] = DynamicLiquidLayer(
            input_size=input_size, 
            hidden_size=hidden_size_per_comp, 
            decay=0.9, 
            target_rate=target_rate
        )
        
        for name in compartment_names:
            self.compartments[name] = DynamicLiquidLayer(
                input_size=input_size, 
                hidden_size=hidden_size_per_comp, 
                decay=0.9, 
                target_rate=target_rate
            )
            
        self.neuromodulator_levels: Dict[str, float] = {name: 0.0 for name in self.compartments.keys()}

    def _apply_context_gating(self, active_context: str):
        """
        文脈シグナルを受け取り、各コンパートメントの活性化度合いを設定。
        無関係なレイヤーのホメオスタシス暴走を防ぐため、対象以外は完全に閉じる。
        """
        for name in self.neuromodulator_levels.keys():
            if name == active_context:
                self.neuromodulator_levels[name] = 1.0
            else:
                self.neuromodulator_levels[name] = 0.0

    def forward_latent_chain(self, active_inputs: List[int], prev_active_hidden: List[int], 
                             current_context: str, learning: bool = False,
                             reward_signal: float = 1.0) -> List[int]:
        self._apply_context_gating(current_context)
        
        combined_fired_indices = []
        
        for name, layer in self.compartments.items():
            modulator_level = self.neuromodulator_levels[name]
            
            if modulator_level <= 0.0:
                continue
            
            compartment_learning = learning and (modulator_level * reward_signal > 0.5)
            
            fired = layer.forward_with_feedback(
                active_inputs=active_inputs,
                prev_active_hidden=prev_active_hidden,
                learning=compartment_learning
            )
            combined_fired_indices.extend(fired)
            
        return list(set(combined_fired_indices))

    def reset_short_term_memory(self):
        """
        タスク切り替え時に呼び出し、過渡的な状態（膜電位や不応期）のみをクリアする。
        学習で獲得した長期記憶（STDPの重みや、ホメオスタシスの動的閾値）は保持される。
        """
        for layer in self.compartments.values():
            if hasattr(layer, 'v'):
                layer.v.fill(0)
            if hasattr(layer, 'refractory'):
                layer.refractory.fill(0)
            # Rustコアを使用している場合はRust側のリセット処理を呼ぶ（状態全クリアになる可能性があるため注意）
            if getattr(layer, 'use_rust', False) and hasattr(layer, 'core'):
                 layer.core.reset()

    def get_compartment_states(self) -> Dict[str, dict]:
        states = {}
        for name, layer in self.compartments.items():
            v, thresh = layer.get_state()
            states[name] = {
                "active_neurons": int(np.sum(v > 0)),
                "avg_threshold": float(np.mean(thresh))
            }
        return states