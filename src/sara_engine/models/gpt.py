FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/gpt.py",
    "//": "タイトル: 自己回帰型SNN (SaraGPT) - 復帰抑制(IoR)導入版",
    "//": "目的: エピソードの無限ループを防ぐため、一度発火した概念のアテンションを減衰させるInhibition of Returnを実装し、自然な停止(EOS)を促す。"
}

import random
import math
from typing import List, Dict, Set, Optional, Tuple

class SaraGPT:
    def __init__(self, encoder):
        self.encoder = encoder
        self.sdr_size = encoder.input_size
        self.synapses: Dict[int, Dict[int, float]] = {}

    def learn_sequence(self, text: str, weight: float = 1.0):
        token_ids = self.encoder.tokenizer.encode(text)
        
        eos_id = self.encoder.tokenizer.vocab.get("<eos>")
        if eos_id is not None:
            if not token_ids or token_ids[-1] != eos_id:
                token_ids.append(eos_id)

        for i in range(len(token_ids) - 1):
            pre_sdr = self.encoder._get_token_sdr(token_ids[i])
            post_sdr = self.encoder._get_token_sdr(token_ids[i+1])
            
            for pre in pre_sdr:
                if pre not in self.synapses:
                    self.synapses[pre] = {}
                for post in post_sdr:
                    self.synapses[pre][post] = self.synapses[pre].get(post, 0.0) + weight

    def predict_next_sdr(self, current_sdr: List[int], context_sdr: List[int] = [], temperature: float = 1.0) -> List[int]:
        potentials: Dict[int, float] = {}
        
        for pre in current_sdr:
            if pre in self.synapses:
                for post, w in self.synapses[pre].items():
                    potentials[post] = potentials.get(post, 0.0) + w

        if context_sdr:
            context_set = set(context_sdr)
            for post in context_set:
                potentials[post] = potentials.get(post, 0.0) + 1.0
                
            for post in potentials:
                if post in context_set:
                    potentials[post] *= 2.0  

        if not potentials:
            return []

        if temperature > 0.0:
            noise_scale = temperature * 0.2 
            for post in potentials:
                noise = random.uniform(-noise_scale, noise_scale)
                potentials[post] += potentials[post] * noise

        activation_threshold = len(current_sdr) * 0.05 
        valid_potentials = {k: v for k, v in potentials.items() if v >= activation_threshold}

        if not valid_potentials:
            return []

        target_n = int(self.sdr_size * self.encoder.density)
        sorted_bits = sorted(valid_potentials.items(), key=lambda x: x[1], reverse=True)
        
        return sorted([b[0] for b in sorted_bits[:target_n]])

    def _sample_top_k_top_p(self, candidates: List[Tuple[int, float]], top_k: int, top_p: float, temperature: float) -> Optional[int]:
        if not candidates:
            return None

        adjusted_candidates = []
        max_score = candidates[0][1]
        
        for tid, score in candidates:
            try:
                weight = math.exp((score - max_score) / temperature)
            except OverflowError:
                weight = 0.0
            adjusted_candidates.append((tid, weight))

        if top_k > 0:
            adjusted_candidates = adjusted_candidates[:top_k]

        if top_p < 1.0:
            total_weight = sum(w for _, w in adjusted_candidates)
            cumulative_prob = 0.0
            filtered_candidates = []
            
            for tid, weight in adjusted_candidates:
                prob = weight / total_weight if total_weight > 0 else 0
                cumulative_prob += prob
                filtered_candidates.append((tid, weight))
                if cumulative_prob >= top_p:
                    break
            adjusted_candidates = filtered_candidates

        if not adjusted_candidates:
            return candidates[0][0]

        tids = [c[0] for c in adjusted_candidates]
        weights = [c[1] for c in adjusted_candidates]
        
        try:
            chosen_tid = random.choices(tids, weights=weights, k=1)[0]
            return chosen_tid
        except ValueError:
            return tids[0]

    def generate(self, prompt: str, context_sdr: List[int] = [], 
                 max_tokens: int = 15, 
                 temperature: float = 0.1, 
                 top_k: int = 5, 
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.2) -> str:
        
        token_ids = self.encoder.tokenizer.encode(prompt)
        if not token_ids: 
            return ""
            
        current_token = token_ids[-1]
        current_sdr = self.encoder._get_token_sdr(current_token)
        generated_words = []
        
        reverse_vocab = {v: k for k, v in self.encoder.tokenizer.vocab.items()}
        eos_id = self.encoder.tokenizer.vocab.get("<eos>", -1)
        unk_id = self.encoder.tokenizer.vocab.get("<unk>", -1)
        
        context_set = set(context_sdr) if context_sdr else set()
        
        # 【脳科学的アプローチ】プロンプトに含まれる単語（すでに発声済み）の概念を
        # アテンション(context_set)から差し引き、反復の動機を消滅させる
        for tid in token_ids:
            prompt_token_sdr = self.encoder.token_sdr_map.get(tid, [])
            context_set -= set(prompt_token_sdr)
        
        recent_tokens = token_ids.copy()
        if len(recent_tokens) > 10:
            recent_tokens = recent_tokens[-10:]

        for _ in range(max_tokens):
            # 動的に消費されていくcontext_setを用いて次を予測
            next_sdr = self.predict_next_sdr(current_sdr, list(context_set), temperature=temperature)
            if not next_sdr: 
                break

            next_set = set(next_sdr)
            candidates = []

            for tid, sdr in self.encoder.token_sdr_map.items():
                if tid == unk_id: continue
                
                overlap = len(next_set.intersection(sdr))
                context_overlap = len(set(sdr).intersection(context_set))
                
                if overlap < 2 and context_overlap < 5: 
                    continue
                
                score = float(overlap) + float(context_overlap) * 2.5
                
                if tid in recent_tokens:
                    if tid == recent_tokens[-1]:
                        score /= (repetition_penalty * 2.0)
                    else:
                        count = recent_tokens.count(tid)
                        score /= (repetition_penalty * count)
                
                candidates.append((tid, score))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[1], reverse=True)

            if temperature < 0.1:
                best_token = candidates[0][0]
            else:
                best_token = self._sample_top_k_top_p(candidates, top_k, top_p, temperature)

            if best_token is None or best_token == eos_id:
                break

            word = reverse_vocab.get(best_token, "")
            if not word: 
                break
                
            generated_words.append(word)
            
            recent_tokens.append(best_token)
            if len(recent_tokens) > 10: 
                recent_tokens.pop(0)
                
            current_sdr = self.encoder._get_token_sdr(best_token)
            
            # 【復帰抑制: Inhibition of Return】
            # 新たに生成された単語の概念をワーキングメモリのアテンションから消費する
            # これにより、目標の言葉をすべて言い終えるとアテンションが空になり、自然に<eos>へ向かう
            context_set -= set(current_sdr)

        return "".join(generated_words)