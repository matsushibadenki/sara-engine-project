_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/core/prefrontal.py",
    "//": "タイトル: 前頭前野 (Prefrontal Cortex)",
    "//": "目的: 行列演算を使わず、単純な集合の積(Intersection)で文脈を決定する。"
}

from typing import List, Dict, Set
# 注意: ここでのSDREncoderのインポート先は実際のファイル構成に合わせる必要がありますが、
# 循環参照を避けるため型ヒントのみにするか、インターフェースを想定します。
# 今回は動的インポート等は行わず、呼び出し側が正しいオブジェクトを渡す前提とします。

class PrefrontalCortex:
    def __init__(self, encoder, compartments: List[str]):
        """
        Args:
            encoder: 文字列をSDR(List[int])に変換できるエンコーダーインスタンス
            compartments: 管理するコンパートメント（文脈）のリスト
        """
        self.encoder = encoder
        self.compartments = compartments
        
        # 各コンパートメントを代表する「概念SDR（アンカー）」
        self.context_anchors: Dict[str, Set[int]] = {}
        for comp in compartments:
            sdr = self.encoder.encode(comp)
            self.context_anchors[comp] = set(sdr)
            
    def determine_context(self, input_sdr: List[int]) -> str:
        best_context = self.compartments[0]
        max_overlap = -1
        
        input_set = set(input_sdr)
        if not input_set:
            return best_context
            
        for comp in self.compartments:
            anchor_set = self.context_anchors[comp]
            # 集合積で計算
            overlap = len(input_set.intersection(anchor_set))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_context = comp
                
        return best_context