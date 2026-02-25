from snn_models.spatiotemporal_stdp import SpatioTemporalSNN
from sara_engine.memory.ltm import SparseMemoryStore
from sara_engine.core.cortex import CorticalColumn
import time
import random
from collections import deque
from typing import List, Dict, Any, Set, Deque
_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/memory/hippocampus.py",
    "//": "タイトル: 皮質-海馬 連動メモリシステム (Cortico-Hippocampal System)",
    "//": "目的: 嗅内皮質からの貫通線維を模倣し、皮質表現と生入力SDRのハイブリッドで検索精度を極大化。過度な数式ハックを排し、皮質(Cortex)が持つ自然な時間バイアスを尊重する。"
}


class CorticoHippocampalSystem:
    def __init__(self, cortex: CorticalColumn, ltm_filepath: str = "sara_integrated_ltm.pkl", max_working_memory_size: int = 15, snn_input_size: int = 2000):
        self.cortex = cortex
        self.ltm = SparseMemoryStore(filepath=ltm_filepath)
        self.ltm.clear()
        self.working_memory: Deque[List[int]] = deque(
            maxlen=max_working_memory_size)

        self.snn_input_size = snn_input_size
        self.st_snn = SpatioTemporalSNN(
            n_in=snn_input_size, n_sensory=200, n_cortex=100)

    def experience_and_memorize(self, sensory_sdr: List[int], content: str, context: str, learning: bool = True) -> List[int]:
        self.cortex.reset_short_term_memory()

        cortical_t1 = self.cortex.forward_latent_chain(
            sensory_sdr, [], current_context=context, learning=learning)
        cortical_t2 = self.cortex.forward_latent_chain(
            [], cortical_t1, current_context=context, learning=learning)

        hippocampal_input = list(set(cortical_t2) | set(sensory_sdr))

        self.ltm.add(sdr=hippocampal_input,
                     content=content, memory_type=context)
        self.working_memory.append(hippocampal_input)

        if learning:
            heat_data = [0.0] * self.snn_input_size
            for idx in sensory_sdr:
                if idx < self.snn_input_size:
                    heat_data[idx] = 1.0

            self.st_snn.step(heat_data)

        return hippocampal_input

    def recall_with_pattern_completion(self, partial_sensory_sdr: List[int], context: str) -> List[Dict[str, Any]]:
        self.cortex.reset_short_term_memory()

        cortical_t1 = self.cortex.forward_latent_chain(
            partial_sensory_sdr, [], current_context=context, learning=False)
        cortical_t2 = self.cortex.forward_latent_chain(
            [], cortical_t1, current_context=context, learning=False)

        hippocampal_query = list(set(cortical_t2) | set(partial_sensory_sdr))
        return self.ltm.search(query_sdr=hippocampal_query, top_k=3, threshold=0.1)

    def in_context_inference(self, current_sensory_sdr: List[int], context: str) -> List[Dict[str, Any]]:
        self.cortex.reset_short_term_memory()

        cortical_t1 = self.cortex.forward_latent_chain(
            current_sensory_sdr, [], current_context=context, learning=False)
        cortical_t2 = self.cortex.forward_latent_chain(
            [], cortical_t1, current_context=context, learning=False)

        # クエリの汚染（WMからのランダムなノイズ混入）を完全に削除し、純粋な関連度を計算する
        query_set = set(cortical_t2) | set(current_sensory_sdr)

        final_results = self.ltm.search(
            query_sdr=list(query_set), top_k=5, threshold=0.01)

        if final_results:
            # LTM内のすべての記憶のタイムスタンプを取得してソート（順位づけ用）
            all_timestamps = sorted([m['timestamp']
                                    for m in self.ltm.memories])

            for res in final_results:
                # Rank-based Recency: ミリ秒の差を無視し「何番目に新しいか」で評価
                rank = all_timestamps.index(res['timestamp'])
                rank_ratio = rank / \
                    max(1, len(all_timestamps) - 1)  # 0.0 ~ 1.0

                # ベーススコア（Cortexによる自然な意味的・時間的バイアス）を最優先する
                # 同スコア帯（競合する事実）でのみ順位が覆る程度の微小なボーナス（最大+10%）にとどめる
                recency_bonus = 1.0 + (rank_ratio * 0.1)
                res['score'] = res['score'] * recency_bonus

            final_results.sort(key=lambda x: x['score'], reverse=True)

        return final_results[:3]

    def consolidate_memories(self, context: str, replay_count: int = 5):
        if not self.ltm.memories:
            return

        target_memories = [
            m for m in self.ltm.memories if m.get('type') == context]
        if not target_memories:
            target_memories = self.ltm.memories

        target_memories.sort(key=lambda x: x['timestamp'])

        replay_samples = target_memories[-replay_count:] if len(
            target_memories) > replay_count else target_memories

        for mem in replay_samples:
            self.cortex.reset_short_term_memory()
            replay_sdr = mem['sdr']

            # リプレイによってCortexの結合重みが更新され、最新の事実が検索時に自然とバイアスされる
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

            heat_data = [0.0] * self.snn_input_size
            for idx in replay_sdr:
                if idx < self.snn_input_size:
                    heat_data[idx] = 1.0
            self.st_snn.step(heat_data)
