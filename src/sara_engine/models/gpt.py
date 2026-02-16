_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/gpt.py",
    "//": "タイトル: 自己回帰型SNN (SaraGPT)",
    "//": "目的: 行列演算を用いずに、SNNの膜電位の揺らぎ（Stochastic Firing）を模倣した確率的デコーディング（Temperature, Top-K, Top-Pサンプリング）を実装する。"
}

import random
import math
from typing import List, Dict, Set, Optional, Tuple

class SaraGPT:
    def __init__(self, encoder):
        self.encoder = encoder
        self.sdr_size = encoder.input_size
        # シナプス結合: {pre_synaptic_bit: {post_synaptic_bit: weight}}
        self.synapses: Dict[int, Dict[int, float]] = {}

    def learn_sequence(self, text: str, weight: float = 1.0):
        """
        文章のトークン遷移を学習する。
        文末に自動的に<eos>を付与し、終了条件を学習させる。
        """
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
                    # ヘブ則的学習: 前シナプスと後シナプスが連続して発火した場合に結合を強化
                    self.synapses[pre][post] = self.synapses[pre].get(post, 0.0) + weight

    def predict_next_sdr(self, current_sdr: List[int], context_sdr: List[int] = [], temperature: float = 1.0) -> List[int]:
        """
        確率的発火モデル (Stochastic Leaky Integrate-and-Fire) の模倣。
        現在のSDRと文脈から次に発火すべきニューロン群を予測する。
        温度パラメータ(temperature)により、膜電位に微小なノイズ（揺らぎ）を注入し、生成の多様性を生み出す。
        """
        potentials: Dict[int, float] = {}
        
        # 1. ボトムアップ入力の積算
        for pre in current_sdr:
            if pre in self.synapses:
                for post, w in self.synapses[pre].items():
                    potentials[post] = potentials.get(post, 0.0) + w

        if not potentials:
            return []

        # 2. トップダウン・アテンションの適用
        if context_sdr:
            context_set = set(context_sdr)
            for post in potentials:
                if post in context_set:
                    potentials[post] *= 15.0  

        # 3. 確率的発火 (Stochastic Firing) のためのノイズ注入
        # 温度が0に近い場合は決定論的、高い場合はランダム性が増す
        if temperature > 0.0:
            noise_scale = temperature * 0.5 # ノイズの強さを調整
            for post in potentials:
                # 標準正規分布近似のノイズを注入 (ガウス分布の代用として一様分布から生成)
                noise = random.uniform(-noise_scale, noise_scale)
                potentials[post] += potentials[post] * noise

        # 4. 発火閾値によるフィルタリング
        activation_threshold = len(current_sdr) * 0.1
        valid_potentials = {k: v for k, v in potentials.items() if v >= activation_threshold}

        if not valid_potentials:
            return []

        # 5. k-Winner-Take-All (ノイズを含んだ電位での上位抽出)
        target_n = int(self.sdr_size * self.encoder.density)
        sorted_bits = sorted(valid_potentials.items(), key=lambda x: x[1], reverse=True)
        
        return sorted([b[0] for b in sorted_bits[:target_n]])

    def _sample_top_k_top_p(self, candidates: List[Tuple[int, float]], top_k: int, top_p: float, temperature: float) -> Optional[int]:
        """
        行列演算やSoftmaxを用いない、標準ライブラリによる Top-K & Top-P (Nucleus) サンプリング。
        """
        if not candidates:
            return None

        # 1. Temperatureによるスコアの調整 (ボルツマン分布の近似)
        # score = exp(score / temperature) の代わりに、オーバーフローを防ぐため相対的な重み付けを行う
        adjusted_candidates = []
        max_score = candidates[0][1] # 既にソート済み前提
        
        for tid, score in candidates:
            # ソフトマックスの代替として、最大スコアとの差に基づく指数関数を利用
            try:
                weight = math.exp((score - max_score) / temperature)
            except OverflowError:
                weight = 0.0
            adjusted_candidates.append((tid, weight))

        # 2. Top-K フィルタリング
        if top_k > 0:
            adjusted_candidates = adjusted_candidates[:top_k]

        # 3. Top-P (Nucleus) フィルタリング
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

        # 4. 重み付きランダムサンプリング
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
                 max_tokens: int = 50, 
                 temperature: float = 0.8, 
                 top_k: int = 40,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.2) -> str:
        """
        SNNの確率的振る舞いを模倣したデコーディング。
        """
        token_ids = self.encoder.tokenizer.encode(prompt)
        if not token_ids: 
            return ""
            
        current_token = token_ids[-1]
        current_sdr = self.encoder._get_token_sdr(current_token)
        generated_words = []
        
        reverse_vocab = {v: k for k, v in self.encoder.tokenizer.vocab.items()}
        eos_id = self.encoder.tokenizer.vocab.get("<eos>", -1)
        unk_id = self.encoder.tokenizer.vocab.get("<unk>", -1)
        
        recent_tokens = [current_token]
        if len(token_ids) > 1:
            recent_tokens.extend(token_ids[-3:])

        for _ in range(max_tokens):
            # 温度パラメータを渡して、微小なノイズを含んだ予測SDRを生成
            next_sdr = self.predict_next_sdr(current_sdr, context_sdr, temperature=temperature)
            if not next_sdr: 
                break

            next_set = set(next_sdr)
            candidates = []

            # 語彙全体のSDRと予測SDRのOverlap（重なり）を計算
            for tid, sdr in self.encoder.token_sdr_map.items():
                if tid == unk_id: continue
                
                overlap = len(next_set.intersection(sdr))
                
                # 最低限のスパイクの一致（閾値）を要求
                if overlap < 3: continue
                
                score = float(overlap)
                
                # 反復ペナルティ: 最近生成した単語のスコアを割り引く
                if tid in recent_tokens:
                    score /= repetition_penalty
                
                candidates.append((tid, score))

            if not candidates:
                break

            # スコア順にソート
            candidates.sort(key=lambda x: x[1], reverse=True)

            # 温度が極端に低い場合はGreedy Search
            if temperature < 0.1:
                best_token = candidates[0][0]
            else:
                # Top-K & Top-P サンプリングの実行
                best_token = self._sample_top_k_top_p(candidates, top_k, top_p, temperature)

            if best_token is None:
                break

            # 停止トークン
            if best_token == eos_id:
                break

            word = reverse_vocab.get(best_token, "")
            if not word: 
                break
                
            generated_words.append(word)
            
            recent_tokens.append(best_token)
            if len(recent_tokens) > 10: 
                recent_tokens.pop(0)
                
            current_sdr = self.encoder._get_token_sdr(best_token)

        return "".join(generated_words)