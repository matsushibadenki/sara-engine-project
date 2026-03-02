_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/spiking_multimodal_hub.py",
    "//": "ファイルの日本語タイトル: マルチモーダルSNN連想ハブ",
    "//": "ファイルの目的や内容: 複数のモダリティ（視覚、テキスト、音声など）からのスパイクを同一の連想記憶空間で結びつける。行列演算を排除し、純粋なスパイクの共起（Hebbian Learning）によるクロスモーダル検索を実現。"
}

from typing import Dict, List
from sara_engine.nn.module import SNNModule

class SpikingMultimodalHub(SNNModule):
    """
    Phase 2: Multi-source Integration Hub
    テキスト、画像、音声などの異なる感覚入力をスパイクのレベルで関連付けるモデル。
    """
    def __init__(self, modalities: List[str], learning_rate: float = 0.5, decay_rate: float = 0.95, max_weight: float = 3.0):
        super().__init__()
        self.modalities = modalities
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.max_weight = max_weight
        
        # 疎な結合を表現する辞書。行列は使わない。
        # 構造: {source_mod: {target_mod: {src_spike: {tgt_spike: weight}}}}
        self.associative_weights: Dict[str, Dict[str, Dict[int, Dict[int, float]]]] = {}
        
        for src in modalities:
            self.associative_weights[src] = {}
            for tgt in modalities:
                if src != tgt:
                    self.associative_weights[src][tgt] = {}
                    
        self.register_state("associative_weights")

    def associate(self, inputs: Dict[str, List[int]]) -> None:
        """
        異なるモダリティ間で同時に発火したスパイクの結合を強める（Hebbian Learning）。
        同時に、使われていない結合は減衰（Decay）させることで破綻を防ぐ。
        """
        # 1. 全体のシナプスを減衰（不要な記憶の忘却と省エネ）
        for src_mod in self.modalities:
            for tgt_mod in self.modalities:
                if src_mod == tgt_mod: continue
                weights = self.associative_weights[src_mod][tgt_mod]
                for s in list(weights.keys()):
                    for t in list(weights[s].keys()):
                        weights[s][t] *= self.decay_rate
                        # 重みが微小になったらシナプスを刈り込む（Pruning）
                        if weights[s][t] < 0.05:
                            del weights[s][t]
                    if not weights[s]:
                        del weights[s]

        # 2. 今回共起したスパイク間の結合を強化（LTP: 長期増強）
        for src_mod, src_spikes in inputs.items():
            for tgt_mod, tgt_spikes in inputs.items():
                if src_mod == tgt_mod: continue
                weights = self.associative_weights[src_mod][tgt_mod]
                for s in src_spikes:
                    if s not in weights:
                        weights[s] = {}
                    for t in tgt_spikes:
                        current_w = weights[s].get(t, 0.0)
                        weights[s][t] = min(self.max_weight, current_w + self.learning_rate)

    def retrieve(self, source_modality: str, source_spikes: List[int], target_modality: str, threshold: float = 1.0) -> List[int]:
        """
        あるモダリティのスパイク入力から、別のモダリティのスパイク表現を連想・想起する。
        """
        if source_modality not in self.associative_weights or target_modality not in self.associative_weights[source_modality]:
            return []

        weights = self.associative_weights[source_modality][target_modality]
        target_potentials: Dict[int, float] = {}

        # ソーススパイクからターゲットへの刺激を伝播
        for s in source_spikes:
            if s in weights:
                for t, w in weights[s].items():
                    target_potentials[t] = target_potentials.get(t, 0.0) + w

        # 閾値を超えたものが発火（想起成功）
        retrieved_spikes = [t for t, pot in target_potentials.items() if pot >= threshold]
        
        # 電位が高い順にソートして返す
        retrieved_spikes.sort(key=lambda x: target_potentials[x], reverse=True)
        return retrieved_spikes