# {
#     "//": "ディレクトリパス: src/sara_engine/learning/sequence_learning.py",
#     "//": "ファイルの日本語タイトル: 神経シーケンス・マネージャー",
#     "//": "ファイルの目的や内容: ニューロンの発火順序（A→B→C）を時間の連鎖として記憶する。非対称STDPにより順方向の結合を優先的に強化し、思考の遷移や文脈の再生（Replay）を可能にする。"
# }

import math
from typing import Any, Dict, List, Tuple


class NeuralSequenceManager:
    """
    Neural Sequence Dynamics の管理。
    発火の時間的な前後関係を記録し、シーケンス・アトラクターの形成を促す。
    """

    def __init__(self, time_window: float = 50.0, sequence_lr: float = 0.02):
        """
        Args:
            time_window: シーケンスとして認める最大時間差 (ms)
            sequence_lr: シーケンス強化の学習率
        """
        self.time_window = time_window
        self.sequence_lr = sequence_lr
        # last_fired[neuron_id] = timestamp
        self.last_fired: Dict[int, float] = {}

    def reset(self):
        self.last_fired.clear()

    def record_firing(self, neuron_id: int, current_time: float) -> List[Tuple[int, float]]:
        """
        ニューロンの発火を記録し、直前に発火したニューロンとの連鎖（順序）を特定する。
        """
        sequence_events = []

        # 直近で発火した他のニューロンを探す
        for pre_id, pre_time in self.last_arrivals(current_time):
            if pre_id == neuron_id:
                continue

            dt = current_time - pre_time
            if 0 < dt <= self.time_window:
                # 順方向の連鎖 (pre_id -> neuron_id) を検出
                # dtが小さいほど（＝直後に発火したほど）強い連鎖とみなす
                strength = math.exp(-dt / 20.0)
                sequence_events.append((pre_id, strength))

        self.last_fired[neuron_id] = current_time
        return sequence_events

    def last_arrivals(self, current_time: float):
        """時間窓内の発火履歴を返す"""
        return [(nid, t) for nid, t in self.last_fired.items()
                if current_time - t <= self.time_window]

    def apply_sequence_reinforcement(
        self,
        pre_id: int,
        post_id: int,
        strength: float,
        synapses_list: List[Dict[int, Tuple[float, int]]],
        neuron_type_manager: Any,
        max_weight: float
    ):
        """
        シーケンスの順方向結合 (pre -> post) を非対称に強化する。
        """
        if pre_id >= len(synapses_list):
            return

        synapses = synapses_list[pre_id]
        if post_id in synapses:
            w, b_id = synapses[post_id]

            # 非対称な強化: pre -> post の重みを一方的に増やす
            delta = self.sequence_lr * strength
            w_abs = min(max_weight, abs(w) + delta)

            # Dale's law を維持
            synapses[post_id] = (
                neuron_type_manager.enforce_dales_law(pre_id, w_abs), b_id)
