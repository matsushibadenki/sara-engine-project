_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/nn/multimodal.py",
    "//": "ファイルの日本語タイトル: マルチモーダル連合学習層",
    "//": "ファイルの目的や内容: 異なる感覚（テキスト、画像等）からのスパイクを同期させ、STDPによって相関を学習することで、クロスモーダルな想起を可能にするモジュール。"
}

from typing import List, Dict, Set, Optional
from .module import SNNModule

class CrossModalAssociator(SNNModule):
    """
    異なる入力ソースからのスパイク集合を結合する連合記憶層。
    「リンゴ」という言葉と「赤い円」の画像が同時に提示された際に、
    そのスパイクパターン間の結合をSTDPで強化する。
    """
    def __init__(self, dim_a: int, dim_b: int, density: float = 0.3):
        super().__init__()
        self.dim_a = dim_a
        self.dim_b = dim_b
        
        # AからBへの結合、BからAへの結合（双方向連合）
        self.weights_a2b: Dict[int, Dict[int, float]] = {}
        self.weights_b2a: Dict[int, Dict[int, float]] = {}
        
        self.register_state("weights_a2b")
        self.register_state("weights_b2a")

    def forward(self, spikes_a: Optional[List[int]] = None, spikes_b: Optional[List[int]] = None, learning: bool = False) -> Dict[str, List[int]]:
        """
        片方の入力からもう一方を想起するか、両方の入力がある場合は連合を学習する。
        """
        spikes_a_list: List[int] = spikes_a or []
        spikes_b_list: List[int] = spikes_b or []
        
        recall_b: List[int] = []
        recall_a: List[int] = []

        # AからBを想起
        if spikes_a_list:
            potentials_b: Dict[int, float] = {}
            for s in spikes_a_list:
                if s in self.weights_a2b:
                    for target, w in self.weights_a2b[s].items():
                        potentials_b[target] = potentials_b.get(target, 0.0) + w
            recall_b = [k for k, v in sorted(potentials_b.items(), key=lambda x: x[1], reverse=True) if v > 0.5]

        # BからAを想起
        if spikes_b_list:
            potentials_a: Dict[int, float] = {}
            for s in spikes_b_list:
                if s in self.weights_b2a:
                    for target, w in self.weights_b2a[s].items():
                        potentials_a[target] = potentials_a.get(target, 0.0) + w
            recall_a = [k for k, v in sorted(potentials_a.items(), key=lambda x: x[1], reverse=True) if v > 0.5]

        # 学習 (STDP)
        if learning and spikes_a_list and spikes_b_list:
            self._update_associative_weights(spikes_a_list, spikes_b_list)

        return {"recall_a": recall_a, "recall_b": recall_b}

    def _update_associative_weights(self, spikes_a: List[int], spikes_b: List[int]) -> None:
        # A -> B の結合強化
        for a in spikes_a:
            if a not in self.weights_a2b: self.weights_a2b[a] = {}
            for b in spikes_b:
                self.weights_a2b[a][b] = min(2.0, self.weights_a2b[a].get(b, 0.0) + 0.2)
        
        # B -> A の結合強化
        for b in spikes_b:
            if b not in self.weights_b2a: self.weights_b2a[b] = {}
            for a in spikes_a:
                self.weights_b2a[b][a] = min(2.0, self.weights_b2a[b].get(a, 0.0) + 0.2)