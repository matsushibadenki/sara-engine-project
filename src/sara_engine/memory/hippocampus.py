_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/memory/hippocampus.py",
    "//": "タイトル: 皮質-海馬 連動メモリシステム (Cortico-Hippocampal System)",
    "//": "目的: 嗅内皮質からの貫通線維(Perforant Path)を模倣し、皮質表現と生入力SDRのハイブリッドで海馬の検索精度(Recall)を極大化する。"
}

from typing import List, Dict, Any, Set, Deque
from collections import deque
import random
from ..core.cortex import CorticalColumn
from .ltm import SparseMemoryStore

class CorticoHippocampalSystem:
    def __init__(self, cortex: CorticalColumn, ltm_filepath: str = "sara_integrated_ltm.pkl", max_working_memory_size: int = 15):
        self.cortex = cortex
        self.ltm = SparseMemoryStore(filepath=ltm_filepath)
        self.ltm.clear() 
        self.working_memory: Deque[List[int]] = deque(maxlen=max_working_memory_size)

    def experience_and_memorize(self, sensory_sdr: List[int], content: str, context: str, learning: bool = True) -> List[int]:
        self.cortex.reset_short_term_memory()
        
        cortical_t1 = self.cortex.forward_latent_chain(sensory_sdr, [], current_context=context, learning=learning)
        cortical_t2 = self.cortex.forward_latent_chain([], cortical_t1, current_context=context, learning=learning)
        
        # 生物学的な貫通線維(Perforant Path)を模倣し、皮質表現(t2)と生の感覚SDRをブレンドする
        # これにより、皮質の学習が未熟でも単語レベルの直接的な重なりで確実に記憶できる
        hippocampal_input = list(set(cortical_t2) | set(sensory_sdr))
        
        self.ltm.add(sdr=hippocampal_input, content=content, memory_type=context)
        self.working_memory.append(hippocampal_input)
        
        return hippocampal_input

    def recall_with_pattern_completion(self, partial_sensory_sdr: List[int], context: str) -> List[Dict[str, Any]]:
        self.cortex.reset_short_term_memory()
        
        cortical_t1 = self.cortex.forward_latent_chain(partial_sensory_sdr, [], current_context=context, learning=False)
        cortical_t2 = self.cortex.forward_latent_chain([], cortical_t1, current_context=context, learning=False)
        
        hippocampal_query = list(set(cortical_t2) | set(partial_sensory_sdr))
        return self.ltm.search(query_sdr=hippocampal_query, top_k=3, threshold=0.1)

    def in_context_inference(self, current_sensory_sdr: List[int], context: str) -> List[Dict[str, Any]]:
        self.cortex.reset_short_term_memory()
        
        cortical_t1 = self.cortex.forward_latent_chain(current_sensory_sdr, [], current_context=context, learning=False)
        cortical_t2 = self.cortex.forward_latent_chain([], cortical_t1, current_context=context, learning=False)
        
        # 検索時もハイブリッドクエリを構築
        query_set = set(cortical_t2) | set(current_sensory_sdr)
        
        if self.working_memory:
            recent_wm = self.working_memory[-1]
            sample_size = int(len(recent_wm) * 0.1)
            if sample_size > 0:
                query_set.update(random.sample(recent_wm, min(sample_size, len(recent_wm))))
                
        final_results = self.ltm.search(query_sdr=list(query_set), top_k=3, threshold=0.01)
        
        if final_results:
            latest_time = max([res['timestamp'] for res in final_results])
            for res in final_results:
                time_diff = latest_time - res['timestamp']
                recency_bonus = 1.0 + (0.5 / (1.0 + time_diff))
                res['score'] = res['score'] * recency_bonus
                
            final_results.sort(key=lambda x: x['score'], reverse=True)
            
        return final_results

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