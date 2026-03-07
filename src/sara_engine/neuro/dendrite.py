# {
#     "//": "ディレクトリパス: src/sara_engine/neuro/dendrite.py",
#     "//": "ファイルの日本語タイトル: 樹状突起マネージャー (v2.2.0)",
#     "//": "ファイルの目的や内容: 1ニューロンをMLP化する樹状突起構造を実装。各枝での局所的な非線形統合（Dendritic Spike）に加え、関連するシナプスを同じ枝に集約させる「Synaptic Clustering」機能を搭載。"
# }

import math
import random

class DendriticBranch:
    """
    1つの樹状突起の枝。独立した非線形計算ユニットとして機能する。
    """
    def __init__(self, branch_id: int, threshold: float = 1.0, gain: float = 2.0):
        self.branch_id = branch_id
        self.threshold = threshold
        self.gain = gain
        self.current_sum = 0.0
        # クラスタリングのための「活動の余韻」
        self.recent_activity = 0.0

    def reset(self):
        self.current_sum = 0.0
        self.recent_activity *= 0.9 # 指数減衰

    def integrate(self, input_value: float):
        """シナプス入力を蓄積"""
        self.current_sum += input_value

    def get_output(self) -> float:
        """
        局所的な非線形出力を計算 (Dendritic Spikeの模倣)
        y = g(Σ w * x)
        """
        if self.current_sum <= 0:
            return 0.0
            
        # シグモイド曲線による超線形統合
        # 入力が閾値を超えると急激に細胞体(Soma)へ伝達される
        activation = 1.0 / (1.0 + math.exp(-8.0 * (self.current_sum - self.threshold)))
        output = activation * self.gain * self.current_sum
        
        if output > 0.5:
            self.recent_activity = 1.0 # 着火記録
            
        return output

class DendriticTree:
    """
    ニューロン内部の樹状突起構造（森）。複数の枝を統括する。
    """
    def __init__(self, num_branches: int = 8):
        self.branches = [DendriticBranch(i) for i in range(num_branches)]

    def reset(self):
        for b in self.branches:
            b.reset()

    def integrate_to_branch(self, branch_id: int, input_value: float):
        """特定の枝に信号を入力"""
        idx = branch_id % len(self.branches)
        self.branches[idx].integrate(input_value)

    def aggregate(self) -> float:
        """全枝の出力を集計し、Soma(細胞体)へ送る"""
        return sum(b.get_output() for b in self.branches)

    def find_best_branch(self) -> int:
        """
        最も活動が高い枝のIDを返す。
        Synaptic Clusteringにおいて、新しいシナプスを割り当てる際のヒントにする。
        """
        best_id = 0
        max_val = -1.0
        for i, b in enumerate(self.branches):
            if b.current_sum > max_val:
                max_val = b.current_sum
                best_id = i
        return best_id

    def get_clustering_target(self) -> int:
        """最近活動した（着火した）枝を優先的に割り当て先として選ぶ"""
        active_branches = [i for i, b in enumerate(self.branches) if b.recent_activity > 0.5]
        if active_branches:
            return random.choice(active_branches)
        return random.randint(0, len(self.branches) - 1)