_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/gpt.py",
    "//": "タイトル: 自己回帰型SNN (SaraGPT)",
    "//": "目的: オンライン学習の強化と、海馬アテンションによる強制発火ロジックの改善。"
}

from typing import List, Dict, Set
from ..memory.sdr import SDREncoder

class SaraGPT:
    def __init__(self, encoder: SDREncoder):
        self.encoder = encoder
        self.sdr_size = encoder.input_size
        self.synapses: Dict[int, Dict[int, float]] = {}

    def learn_sequence(self, text: str, weight: float = 1.0):
        """文章のトークン遷移を学習する。weight引数で学習強度を調整可能。"""
        token_ids = self.encoder.tokenizer.encode(text)
        for i in range(len(token_ids) - 1):
            pre_sdr = self.encoder._get_token_sdr(token_ids[i])
            post_sdr = self.encoder._get_token_sdr(token_ids[i+1])
            
            for pre in pre_sdr:
                if pre not in self.synapses:
                    self.synapses[pre] = {}
                for post in post_sdr:
                    # 指定された重みで結合を強化
                    self.synapses[pre][post] = self.synapses[pre].get(post, 0.0) + weight

    def predict_next_sdr(self, current_sdr: List[int], context_sdr: List[int] = []) -> List[int]:
        potentials: Dict[int, float] = {}
        for pre in current_sdr:
            if pre in self.synapses:
                for post, w in self.synapses[pre].items():
                    potentials[post] = potentials.get(post, 0.0) + w

        # トップダウン・アテンション: 海馬の文脈ビットをさらに強力に(15倍)ブースト
        if context_sdr:
            context_set = set(context_sdr)
            for post in potentials:
                if post in context_set:
                    potentials[post] *= 15.0  

        if not potentials:
            return []

        # 絶対発火閾値
        activation_threshold = len(current_sdr) * 0.2
        valid_potentials = {k: v for k, v in potentials.items() if v >= activation_threshold}

        if not valid_potentials:
            return []

        target_n = int(self.sdr_size * self.encoder.density)
        sorted_bits = sorted(valid_potentials.items(), key=lambda x: x[1], reverse=True)
        return sorted([b[0] for b in sorted_bits[:target_n]])

    def generate(self, prompt: str, context_sdr: List[int] = [], max_tokens: int = 10) -> str:
        token_ids = self.encoder.tokenizer.encode(prompt)
        if not token_ids: 
            return ""
            
        current_token = token_ids[-1]
        current_sdr = self.encoder._get_token_sdr(current_token)
        generated_words = []
        reverse_vocab = {v: k for k, v in self.encoder.tokenizer.vocab.items()}
        
        recent_tokens = [current_token]

        for _ in range(max_tokens):
            next_sdr = self.predict_next_sdr(current_sdr, context_sdr)
            if not next_sdr: 
                break

            next_set = set(next_sdr)
            best_token = -1
            max_overlap = -1

            for tid, sdr in self.encoder.token_sdr_map.items():
                if tid in recent_tokens[-3:]:
                    continue
                    
                overlap = len(next_set.intersection(sdr))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_token = tid

            # デコード閾値を18ビットへさらに厳格化（誤発火を許さない）
            if best_token == -1 or max_overlap < 18: 
                break

            word = reverse_vocab.get(best_token, "")
            if not word: 
                break
                
            generated_words.append(word)
            recent_tokens.append(best_token)
            current_sdr = self.encoder._get_token_sdr(best_token)

            if word in ["です", "ます", "ました", "？", "。"]:
                break

        return "".join(generated_words)