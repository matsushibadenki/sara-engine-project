# {
#     "//": "ディレクトリパス: src/sara_engine/neuro/neuron.py",
#     "//": "ファイルの日本語タイトル: 樹状突起ニューロン (オブジェクト指向版)",
#     "//": "ファイルの目的や内容: 複数の樹状突起(Branch)を持つLIFニューロン。枝ごとの非線形な局所統合（NMDAスパイクの模倣）により、1ニューロンを数層のニューラルネット(MLP)と同等の計算ユニットへと進化させる。"
# }

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

    def add_current(self, current: float):
        """シナプスからの電流を蓄積する"""
        self.current_input += current

    def compute_output(self) -> float:
        """
        局所的な入力を非線形に変換し、細胞体(Soma)へ送る。
        同時入力(Coincidence)があった場合のみ超線形に増幅される。
        """
        if self.current_input <= 0:
            out = 0.0
        else:
            # 局所的な樹状突起スパイクをシグモイド関数で模倣
            activation = 1.0 / (1.0 + math.exp(-5.0 * (self.current_input - self.threshold)))
            out = activation * self.gain * self.current_input
            
        self.current_input = 0.0  # 計算後にリセット
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

    def step(self) -> bool:
        """
        各枝からの非線形出力を細胞体で統合し、発火を判定する。
        """
        if self.refractory_time > 0:
            self.refractory_time -= 1
            self.v = 0.0
            self.spike = False
            # 電流は消費されるが発火はしない
            for branch in self.branches:
                branch.current_input = 0.0
            return False

        # 各枝の独立した計算結果を集約
        soma_input = sum(branch.compute_output() for branch in self.branches)

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