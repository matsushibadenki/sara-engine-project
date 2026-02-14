_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/memory/hippocampus.py",
    "//": "タイトル: 皮質-海馬 連動メモリシステム (Cortico-Hippocampal System)",
    "//": "目的: CorticalColumn(皮質)のパターン補完能力と、LTM(海馬)のエピソード記憶を統合する。"
}

from typing import List, Dict, Any
from ..core.cortex import CorticalColumn
from .ltm import SparseMemoryStore

class CorticoHippocampalSystem:
    """
    大脳皮質(Cortex)と海馬(Hippocampus/LTM)の連動システム。
    SDRベースの潜在空間のままエピソードを記銘・想起する。
    """
    def __init__(self, cortex: CorticalColumn, ltm_filepath: str = "sara_integrated_ltm.pkl"):
        self.cortex = cortex
        self.ltm = SparseMemoryStore(filepath=ltm_filepath)
        self.ltm.clear() # 初期化時にクリア（永続化する場合は削除可）

    def experience_and_memorize(self, sensory_sdr: List[int], content: str, context: str, learning: bool = True) -> List[int]:
        """
        [記銘] 体験を皮質で処理し、海馬(LTM)に刻み込む
        """
        self.cortex.reset_short_term_memory()
        
        # Step 1: 外部からの感覚入力
        cortical_t1 = self.cortex.forward_latent_chain(sensory_sdr, [], current_context=context, learning=learning)
        # Step 2: リカレント結合による自己想起（ここで抽象化される）
        cortical_t2 = self.cortex.forward_latent_chain([], cortical_t1, current_context=context, learning=learning)
        
        # 抽象化された皮質の状態(SDR)をキーとして、エピソード(content)を保存
        self.ltm.add(sdr=cortical_t2, content=content, memory_type=context)
        
        return cortical_t2

    def recall_with_pattern_completion(self, partial_sensory_sdr: List[int], context: str) -> List[Dict[str, Any]]:
        """
        [想起] 不完全な入力から皮質がパターンを補完し、海馬から記憶を引き出す
        """
        self.cortex.reset_short_term_memory()
        
        # Step 1: ノイズを含む、あるいは不完全な感覚入力
        cortical_t1 = self.cortex.forward_latent_chain(partial_sensory_sdr, [], current_context=context, learning=False)
        # Step 2: リカレント結合が「いつもの形」へ自己想起・補完(Pattern Completion)する
        cortical_t2 = self.cortex.forward_latent_chain([], cortical_t1, current_context=context, learning=False)
        
        # 補完された綺麗なSDRを使って、LTMから連想検索を行う
        # 閾値(threshold)は適宜調整可能
        return self.ltm.search(query_sdr=cortical_t2, top_k=3, threshold=0.1)