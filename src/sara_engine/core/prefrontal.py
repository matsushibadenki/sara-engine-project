_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/prefrontal.py",
    "//": "タイトル: 前頭前野 (Prefrontal Cortex)",
    "//": "目的: 入力SDRと各概念のオーバーラップを計算し、自律的に適切な文脈(Context)を決定する。"
}

from typing import List, Dict
from ..memory.sdr import SDREncoder

class PrefrontalCortex:
    """
    前頭前野（Agentic Orchestrator / ルーター）
    入力SDRと各コンパートメントの「概念SDR」のオーバーラップを計算し、
    最も適切な文脈（Context）を自律的に決定・ルーティングする。
    """
    def __init__(self, encoder: SDREncoder, compartments: List[str]):
        self.encoder = encoder
        self.compartments = compartments
        
        # 各コンパートメントを代表する「概念SDR（アンカー）」を生成・保持
        self.context_anchors: Dict[str, List[int]] = {}
        for comp in compartments:
            # コンパートメント名そのものを概念としてエンコードする
            self.context_anchors[comp] = self.encoder.encode(comp)
            
    def determine_context(self, input_sdr: List[int]) -> str:
        """
        行列演算を使わず、単純な集合の積(Intersection)で文脈を決定する
        """
        best_context = self.compartments[0]
        max_overlap = -1
        
        input_set = set(input_sdr)
        if not input_set:
            return best_context
            
        for comp, anchor_sdr in self.context_anchors.items():
            anchor_set = set(anchor_sdr)
            overlap = len(input_set.intersection(anchor_set))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_context = comp
                
        return best_context