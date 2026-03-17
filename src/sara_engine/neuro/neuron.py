# Directory Path: src/sara_engine/neuro/neuron.py
# English Title: Event-Driven Dendritic Neuron
# Purpose/Content: 複数の樹状突起(Branch)を持つLIFニューロン。無入力時の計算をスキップするイベント駆動化（省エネ）と、同時入力の厳密な非線形統合（精度向上）を両立。多言語対応機能を含む。

import math

class DendriticBranch:
    """
    ニューロンの1つの枝。シナプス入力を受け取り、非線形に統合する。
    """
    def __init__(self, branch_id: int):
        self.id = branch_id
        self.current_input = 0.0
        self.threshold = 0.5  # 局所的なスパイクの閾値
        self.gain = 2.0       # 非線形増幅のゲイン
        self.is_active = False # 省エネ用のアクティブフラグ

    def get_status(self, lang: str = "en") -> str:
        """多言語対応: 枝の状態を取得"""
        if lang == "ja":
            return f"枝 {self.id}: アクティブ={self.is_active}, 入力={self.current_input}"
        return f"Branch {self.id}: Active={self.is_active}, Input={self.current_input}"

    def add_current(self, current: float):
        """シナプスからの電流を蓄積する"""
        if current > 0:
            self.current_input += current
            self.is_active = True

    def compute_output(self) -> float:
        """
        局所的な入力を非線形に変換し、細胞体(Soma)へ送る。
        同時入力(Coincidence)があった場合のみ超線形に増幅される。
        入力がない場合は演算を完全にスキップする。
        """
        if not self.is_active or self.current_input <= 0:
            self.current_input = 0.0
            self.is_active = False
            return 0.0
            
        # 局所的な樹状突起スパイクをシグモイド関数で模倣
        activation = 1.0 / (1.0 + math.exp(-5.0 * (self.current_input - self.threshold)))
        out = activation * self.gain * self.current_input
            
        self.current_input = 0.0  # 計算後にリセット
        self.is_active = False
        return out

class Neuron:
    """
    樹状突起コンパートメントを持つ Leaky Integrate-and-Fire ニューロン。
    """
    def __init__(self, neuron_id: int, is_inhibitory: bool = False, num_branches: int = 4):
        self.id = neuron_id
        self.is_inhibitory = is_inhibitory
        
        # 複数の樹状突起枝（計算ユニット）を保持
        self.branches = [DendriticBranch(i) for i in range(num_branches)]
        
        # 細胞体(Soma)のパラメータ
        self.v = 0.0
        self.threshold = 1.0
        self.leak = 0.95
        self.spike = False
        
        self.refractory_time = 0
        self.active_branches = set() # O(1)でアクティブな枝を管理

    def add_input_to_branch(self, branch_index: int, current: float):
        """外部から特定の枝へ電流を注入する"""
        if 0 <= branch_index < len(self.branches) and current > 0:
            self.branches[branch_index].add_current(current)
            self.active_branches.add(branch_index)

    def step(self) -> bool:
        """
        イベント駆動による省エネ演算。発火履歴と入力がない場合は膜電位減衰のみを計算。
        """
        if self.refractory_time > 0:
            self.refractory_time -= 1
            self.v = 0.0
            self.spike = False
            # 電流は消費されるが発火はしない
            for b_idx in list(self.active_branches):
                self.branches[b_idx].current_input = 0.0
                self.branches[b_idx].is_active = False
            self.active_branches.clear()
            return False

        soma_input = 0.0
        
        # アクティブな枝のみ計算を実行し、計算資源を劇的に削減
        if self.active_branches:
            for b_idx in list(self.active_branches):
                soma_input += self.branches[b_idx].compute_output()
            self.active_branches.clear()

        # 入力がない場合は乗算を1回行うだけで済む。極小値はゼロに丸めて演算を停止。
        if soma_input == 0.0 and self.v < 0.01:
            self.v = 0.0 
            self.spike = False
            return False

        # リークと入力の加算
        self.v = self.v * self.leak + soma_input

        # 発火判定
        if self.v > self.threshold:
            self.spike = True
            self.v = 0.0
            self.refractory_time = 2
        else:
            self.spike = False

        return self.spike