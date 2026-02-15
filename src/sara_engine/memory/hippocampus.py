_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/memory/hippocampus.py",
    "//": "タイトル: 皮質-海馬 連動メモリシステム (Cortico-Hippocampal System)",
    "//": "目的: CorticalColumn(皮質)とLTM(海馬)を統合し、大規模文脈内学習(ICL)における時間的推論(Recency BiasとLatent Chain)を実現する。"
}

from typing import List, Dict, Any, Set
from collections import deque
import random
from ..core.cortex import CorticalColumn
from .ltm import SparseMemoryStore

class CorticoHippocampalSystem:
    """
    大脳皮質(Cortex)と海馬(Hippocampus/LTM)の連動システム。
    ワーキングメモリを用いたSDR文脈内学習と、リプレイを通じた継続学習をサポート。
    """
    def __init__(self, cortex: CorticalColumn, ltm_filepath: str = "sara_integrated_ltm.pkl", max_working_memory_size: int = 15):
        self.cortex = cortex
        self.ltm = SparseMemoryStore(filepath=ltm_filepath)
        self.ltm.clear() 
        self.working_memory = deque(maxlen=max_working_memory_size)

    def experience_and_memorize(self, sensory_sdr: List[int], content: str, context: str, learning: bool = True) -> List[int]:
        self.cortex.reset_short_term_memory()
        
        cortical_t1 = self.cortex.forward_latent_chain(sensory_sdr, [], current_context=context, learning=learning)
        cortical_t2 = self.cortex.forward_latent_chain([], cortical_t1, current_context=context, learning=learning)
        
        self.ltm.add(sdr=cortical_t2, content=content, memory_type=context)
        self.working_memory.append(cortical_t2)
        
        return cortical_t2

    def recall_with_pattern_completion(self, partial_sensory_sdr: List[int], context: str) -> List[Dict[str, Any]]:
        self.cortex.reset_short_term_memory()
        
        cortical_t1 = self.cortex.forward_latent_chain(partial_sensory_sdr, [], current_context=context, learning=False)
        cortical_t2 = self.cortex.forward_latent_chain([], cortical_t1, current_context=context, learning=False)
        
        return self.ltm.search(query_sdr=cortical_t2, top_k=3, threshold=0.1)

    def in_context_inference(self, current_sensory_sdr: List[int], context: str) -> List[Dict[str, Any]]:
        """
        [大規模文脈内学習 (In-Context Learning)]
        VSAとHD Computingの知見に基づく時間減衰と多段連想を用いて、最新の状態推論を行う。
        """
        self.cortex.reset_short_term_memory()
        
        # 1. クエリから直接「起点となる過去の記憶(アンカー)」を連想
        initial_recall = self.ltm.search(query_sdr=current_sensory_sdr, top_k=1, threshold=0.05)
        anchor_sdr = initial_recall[0]['sdr'] if initial_recall and 'sdr' in initial_recall[0] else []
        
        # 2. ワーキングメモリのSDRに対し「時間減衰(Spike Decay)」を適用してマクロ文脈を生成
        macro_context_set: Set[int] = set(current_sensory_sdr)
        if anchor_sdr:
            macro_context_set.update(anchor_sdr)
            
        wm_list = list(self.working_memory)
        total_items = len(wm_list)
        for i, past_sdr in enumerate(wm_list):
            # iが大きい(末尾に近い)ほど新しい記憶。保持確率を線形に設定し、古い記憶のビットを間引く
            keep_prob = (i + 1) / (total_items + 1e-5) 
            for bit in past_sdr:
                if random.random() <= keep_prob:
                    macro_context_set.add(bit)
                    
        macro_context_sdr = list(macro_context_set)
        
        # 3. 起点記憶と時間減衰文脈を皮質に入力し、潜在空間で推論を「次の状態」へ遷移させる
        cortical_out = self.cortex.forward_latent_chain(
            active_inputs=macro_context_sdr, 
            prev_active_hidden=anchor_sdr, 
            current_context=context, 
            learning=False
        )
        
        self.working_memory.append(cortical_out)
        
        # 4. 遷移後の皮質状態(SDR)を用いてLTMを検索 (閾値を下げて広めに取得)
        final_results = self.ltm.search(query_sdr=cortical_out, top_k=5, threshold=0.01)
        
        # 5. Recency Bias (直近重視) による再ランク付け
        # 語彙の重複スコアが拮抗した場合、タイムスタンプが新しい事象(状態)を優先する
        if final_results:
            latest_time = max([res['timestamp'] for res in final_results])
            for res in final_results:
                time_diff = latest_time - res['timestamp']
                # 時間差が少ない(新しい)ほど最大1.5倍のボーナスを付与してスコアを底上げ
                recency_bonus = 1.0 + (0.5 / (1.0 + time_diff))
                res['score'] = res['score'] * recency_bonus
                
            final_results.sort(key=lambda x: x['score'], reverse=True)
            
        return final_results[:3]

    def consolidate_memories(self, context: str, replay_count: int = 5):
        if not self.ltm.memories:
            return
            
        target_memories = [m for m in self.ltm.memories if m.get('type') == context]
        if not target_memories:
            target_memories = self.ltm.memories
            
        replay_samples = random.choices(target_memories, k=min(replay_count, len(target_memories)))
        
        for mem in replay_samples:
            self.cortex.reset_short_term_memory()
            replay_sdr = mem['sdr']
            
            cortical_t1 = self.cortex.forward_latent_chain(
                active_inputs=replay_sdr, 
                prev_active_hidden=[], 
                current_context=context, 
                learning=True,
                reward_signal=0.6 
            )
            self.cortex.forward_latent_chain(
                active_inputs=[], 
                prev_active_hidden=cortical_t1, 
                current_context=context, 
                learning=True,
                reward_signal=0.6
            )