# SARA Engine: NeuroFEM Extension
_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/neurofem.py",
    "//": "タイトル: ニューロモルフィック有限要素法 (NeuroFEM) レイヤー",
    "//": "目的: 行列演算や誤差逆伝播法を用いずに、SNNのスパイク伝播によって物理的変形や熱伝導などの偏微分方程式(PDE)を解く。FEMのメッシュノードをニューロンに、剛性行列をスパースなシナプス結合にマッピングする。"
}

from typing import List, Dict

class NeuroFEMLayer:
    """
    有限要素法(FEM)のメッシュ構造をSNNでエミュレートするレイヤー。
    ノード間の物理的相互作用を行列ではなく、スパースなシナプスの発火伝達として計算する。
    """
    def __init__(self, num_nodes: int, threshold: float = 1.0, decay: float = 0.95):
        self.num_nodes = num_nodes
        self.threshold = threshold
        self.decay = decay
        
        # 膜電位（物理量：温度、変位などの近似値の蓄積）
        self.v = [0.0 for _ in range(num_nodes)]
        
        # スパースなシナプス結合（FEMにおける剛性行列の代わり）
        # 形式: { 自身のノードID: { 隣接ノードID: 伝達重み, ... }, ... }
        self.synapses: Dict[int, Dict[int, float]] = {i: {} for i in range(num_nodes)}
        
        # 外部からの定常的な入力（物理的な外力や境界条件）
        self.biases = [0.0 for _ in range(num_nodes)]

    def add_connection(self, node_i: int, node_j: int, weight: float) -> None:
        """
        メッシュ上の隣接するノード間にシナプス結合（剛性・熱伝達率）を設定する。
        """
        if node_i not in self.synapses:
            self.synapses[node_i] = {}
        if node_j not in self.synapses:
            self.synapses[node_j] = {}
            
        # 物理シミュレーションのため通常は対称的な相互作用を持つ
        self.synapses[node_i][node_j] = weight
        self.synapses[node_j][node_i] = weight

    def set_boundary_condition(self, node_id: int, bias_value: float) -> None:
        """
        特定のノードに外力や境界条件を定常バイアスとして設定する。
        """
        if 0 <= node_id < self.num_nodes:
            self.biases[node_id] = bias_value

    def forward_step(self, active_inputs: List[int]) -> List[int]:
        """
        行列演算を一切使わず、スパイクと局所的な伝播のみでネットワークの状態を1ステップ進める。
        """
        fired_indices = []
        
        # 1. 減衰、定常バイアスの印加、および外部入力スパイクの処理
        for i in range(self.num_nodes):
            self.v[i] *= self.decay
            self.v[i] += self.biases[i]
            
        for node_id in active_inputs:
            if 0 <= node_id < self.num_nodes:
                self.v[node_id] += 1.0
                
        # 2. 発火判定と隣接ノードへのエネルギー（スパイク）伝播
        for i in range(self.num_nodes):
            if self.v[i] >= self.threshold:
                fired_indices.append(i)
                # 発火したニューロンの電位をリセット
                self.v[i] -= self.threshold 
                
                # 行列計算を使わず、スパース結合の辞書を辿って直接隣接ニューロンの電位を更新
                for neighbor, weight in self.synapses[i].items():
                    self.v[neighbor] += weight
                    
        return fired_indices
        
    def get_steady_state(self) -> List[float]:
        """
        各ノードの現在の膜電位を取得する。
        十分な回数 forward_step を回した後のこの値が、FEMシミュレーションの近似解となる。
        """
        return self.v.copy()

    def reset_state(self) -> None:
        """
        シミュレーション状態を初期化する。結合構造や境界条件は保持される。
        """
        self.v = [0.0 for _ in range(self.num_nodes)]