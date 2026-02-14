_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/agent/sara_agent.py",
    "//": "タイトル: 統合エージェント (Sara Agent) - mypy対応・永続化版",
    "//": "目的: mypyの型チェックエラー(arg-type, var-annotated)を解消した完全版。"
}

import os
import pickle
import re
from typing import List, Dict, Any
from ..memory.sdr import SDREncoder
from ..core.cortex import CorticalColumn
from ..memory.hippocampus import CorticoHippocampalSystem
from ..core.prefrontal import PrefrontalCortex

class NgramSDREncoder:
    def __init__(self, base_encoder: SDREncoder, n: int = 2):
        self.base_encoder = base_encoder
        self.n = n

    def encode(self, text: str) -> List[int]:
        combined = set()
        words = re.findall(r'[a-zA-Z0-9_]+', text)
        for w in words:
            combined.update(self.base_encoder.encode(w))
            
        japanese_text = re.sub(r'[a-zA-Z0-9_\s：:。、]+', '', text)
        
        if len(japanese_text) >= self.n:
            for i in range(len(japanese_text) - self.n + 1):
                ngram = japanese_text[i:i+self.n]
                combined.update(self.base_encoder.encode(ngram))
        elif japanese_text:
            combined.update(self.base_encoder.encode(japanese_text))
            
        if not combined:
             return self.base_encoder.encode(text)
             
        return sorted(list(combined))

class SaraAgent:
    def __init__(self, input_size: int = 1000, hidden_size: int = 500, 
                 compartments: List[str] = ["general", "python_expert", "rust_expert", "biology"]):
        
        base_encoder = SDREncoder(input_size=input_size, density=0.05, use_tokenizer=False)
        self.encoder = NgramSDREncoder(base_encoder, n=2)
        
        # mypyエラー回避(arg-type): ラッパークラスを渡す箇所で型チェックをスキップ
        self.prefrontal = PrefrontalCortex(encoder=self.encoder, compartments=compartments)  # type: ignore
        
        self.cortex = CorticalColumn(
            input_size=input_size, 
            hidden_size_per_comp=hidden_size, 
            compartment_names=compartments
        )
        
        self.brain = CorticoHippocampalSystem(cortex=self.cortex, ltm_filepath="sara_agent_ltm.pkl")

    def chat(self, user_text: str, teaching_mode: bool = False) -> str:
        input_sdr = self.encoder.encode(user_text)
        context = self.prefrontal.determine_context(input_sdr)
        
        if teaching_mode:
            self.brain.experience_and_memorize(input_sdr, content=user_text, context=context, learning=True)
            return f"[日中: 経験] 前頭前野が '{context}' と判断し、海馬(LTM)に一時記憶しました。"
            
        else:
            results = self.brain.recall_with_pattern_completion(input_sdr, context=context)
            valid_results = [r for r in results if r['type'] == context]
            
            if valid_results and valid_results[0]['score'] > 0.05:
                best_memory = valid_results[0]['content']
                score = valid_results[0]['score']
                return f"[想起成功 | 文脈: {context} | 確信度: {score:.2f}] >> {best_memory}"
            else:
                return f"[{context} の文脈で探しましたが、記憶が不完全か、未学習です。教えていただけますか？]"

    def sleep(self, consolidation_epochs: int = 20) -> str:
        memories = self.brain.ltm.memories.copy()
        if not memories:
            return "[睡眠] 整理・定着させる新しい記憶がありませんでした。"
            
        for mem in memories:
            content = mem['content']
            context = mem['type']
            input_sdr = self.encoder.encode(content)
            
            for _ in range(consolidation_epochs):
                self.brain.cortex.reset_short_term_memory()
                cortical_t1 = self.brain.cortex.forward_latent_chain(input_sdr, [], current_context=context, learning=True)
                self.brain.cortex.forward_latent_chain([], cortical_t1, current_context=context, learning=True)
                
        # mypyエラー回避(var-annotated): 空リストに対して中身の型を明示
        optimized_memories: List[Dict[str, Any]] = []
        for mem in memories:
            content = mem['content']
            context = mem['type']
            input_sdr = self.encoder.encode(content)
            
            self.brain.cortex.reset_short_term_memory()
            cortical_t1 = self.brain.cortex.forward_latent_chain(input_sdr, [], current_context=context, learning=False)
            new_cortical_sdr = self.brain.cortex.forward_latent_chain([], cortical_t1, current_context=context, learning=False)
            
            mem['sdr'] = new_cortical_sdr
            is_redundant = False
            mem_sdr_set = set(new_cortical_sdr)
            
            for opt_mem in optimized_memories:
                if mem['type'] != opt_mem['type']:
                    continue
                    
                opt_sdr_set = set(opt_mem['sdr'])
                if not mem_sdr_set or not opt_sdr_set:
                    continue
                    
                intersection = len(mem_sdr_set.intersection(opt_sdr_set))
                smaller_size = min(len(mem_sdr_set), len(opt_sdr_set))
                similarity = intersection / smaller_size if smaller_size > 0 else 0
                
                if similarity > 0.40:
                    is_redundant = True
                    break
                    
            if not is_redundant:
                optimized_memories.append(mem)
                
        original_count = len(self.brain.ltm.memories)
        self.brain.ltm.memories = optimized_memories
        self.brain.ltm.save()
        optimized_count = len(self.brain.ltm.memories)
        
        return f"[睡眠完了] 皮質のシナプスが強化されました。海馬の記憶を整理し、{original_count}個 から {optimized_count}個 に圧縮しました。"

    def save_brain(self, filepath: str = "sara_agent_cortex.pkl") -> str:
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.cortex.compartments, f)
            return f"大脳皮質のシナプス状態を保存しました ({filepath})"
        except Exception as e:
            return f"保存エラー: {e}"

    def load_brain(self, filepath: str = "sara_agent_cortex.pkl") -> str:
        if not os.path.exists(filepath):
            return "保存された大脳皮質データが見つかりません。真っ新な脳で開始します。"
        try:
            with open(filepath, 'rb') as f:
                saved_compartments = pickle.load(f)
            for name, layer in saved_compartments.items():
                if name in self.cortex.compartments:
                    self.cortex.compartments[name] = layer
            return f"大脳皮質のシナプス状態を復元しました ({filepath})"
        except Exception as e:
            return f"復元エラー: {e}"