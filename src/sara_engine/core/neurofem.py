_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/neurofem.py",
    "//": "タイトル: ニューロモルフィック有限要素法 (NeuroFEM) レイヤー",
    "//": "目的: 行列演算や誤差逆伝播法を用いずに、SNNのスパイク伝播によって物理シミュレーションを行う。"
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
        self.v: List[float] = [0.0] * num_nodes
        self.biases: List[float] = [0.0] * num_nodes
        
        # スパースなシナプス結合（FEMにおける剛性行列の代わり）
        # 形式: { 自身のノードID: { 隣接ノードID: 伝達重み, ... }, ... }
        self.synapses: Dict[int, Dict[int, float]] = {i: {} for i in range(num_nodes)}
    
    def add_connection(self, node_a: int, node_b: int, weight: float) -> None:
        """
        メッシュのエッジ（要素間の結合）を追加する。
        双方向の結合として定義し、作用・反作用を表現する。
        """
        if 0 <= node_a < self.num_nodes and 0 <= node_b < self.num_nodes:
            self.synapses[node_a][node_b] = weight
            self.synapses[node_b][node_a] = weight

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
        # 注: 同期的な更新を行うため、加算分は一時バッファに貯めるか、即時反映かを決める必要がある。
        # ここでは単純なLIFモデルとして即時反映させつつ、連鎖発火を防ぐためループ順序依存を受け入れる
        # （より厳密にするならダブルバッファリングが必要だが、軽量化のためこのままとする）
        
        # まず発火するノードを特定
        current_step_fired = []
        for i in range(self.num_nodes):
            if self.v[i] >= self.threshold:
                current_step_fired.append(i)
                self.v[i] -= self.threshold # リセット
        
        # 発火による伝播処理
        for src_node in current_step_fired:
            fired_indices.append(src_node)
            # 隣接ノードへ重みを加算
            for target_node, weight in self.synapses[src_node].items():
                self.v[target_node] += weight
                
        return fired_indices
    
    def get_state(self) -> List[float]:
        """現在の全ノードの物理量（膜電位）をリストで返す"""
        return list(self.v)

    def get_steady_state(self) -> List[float]:
        """定常状態の全ノードの物理量（膜電位）をリストで返す（テストコード用のエイリアス）"""
        return self.get_state()