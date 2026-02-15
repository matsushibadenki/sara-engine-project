_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/models/gpt.py",
    "//": "タイトル: 自己回帰型SNN (SaraGPT)",
    "//": "目的: 生成停止条件の柔軟化（<eos>自動学習）と、Top-kサンプリング・反復ペナルティによるデコード精度の向上。"
}

import random
import math
from typing import List, Dict, Set, Optional
from ..memory.sdr import SDREncoder

class SaraGPT:
    def __init__(self, encoder: SDREncoder):
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
        
        # 明示的に<eos>を学習データに追加（終了状態の学習）
        eos_id = self.encoder.tokenizer.vocab.get("<eos>")
        if eos_id is not None:
            # すでに末尾にある場合を除き追加
            if not token_ids or token_ids[-1] != eos_id:
                token_ids.append(eos_id)

        for i in range(len(token_ids) - 1):
            pre_sdr = self.encoder._get_token_sdr(token_ids[i])
            post_sdr = self.encoder._get_token_sdr(token_ids[i+1])
            
            for pre in pre_sdr:
                if pre not in self.synapses:
                    self.synapses[pre] = {}
                for post in post_sdr:
                    # 指定された重みで結合を強化（ヘブ則的学習）
                    self.synapses[pre][post] = self.synapses[pre].get(post, 0.0) + weight

    def predict_next_sdr(self, current_sdr: List[int], context_sdr: List[int] = []) -> List[int]:
        """現在のSDRと文脈SDRから、次に発火すべきSDR（予測ビット群）を生成する"""
        potentials: Dict[int, float] = {}
        
        # 1. ボトムアップ入力（直前の単語からの遷移）
        for pre in current_sdr:
            if pre in self.synapses:
                for post, w in self.synapses[pre].items():
                    potentials[post] = potentials.get(post, 0.0) + w

        # 2. トップダウン・アテンション（海馬・短期記憶からのバイアス）
        # 文脈SDRに含まれるビットが予測候補にあれば、その電位を強力にブーストする
        if context_sdr:
            context_set = set(context_sdr)
            for post in potentials:
                if post in context_set:
                    # 文脈との一致は非常に強いヒントになるため係数を大きく設定
                    potentials[post] *= 15.0  

        if not potentials:
            return []

        # 3. 発火閾値によるノイズ除去
        # 入力ビット数に比例した閾値を設定
        activation_threshold = len(current_sdr) * 0.2
        valid_potentials = {k: v for k, v in potentials.items() if v >= activation_threshold}

        if not valid_potentials:
            return []

        # 4. 膜電位の高い順にk-WTA (k-Winner-Take-All)
        target_n = int(self.sdr_size * self.encoder.density)
        sorted_bits = sorted(valid_potentials.items(), key=lambda x: x[1], reverse=True)
        
        return sorted([b[0] for b in sorted_bits[:target_n]])

    def generate(self, prompt: str, context_sdr: List[int] = [], 
                 max_tokens: int = 20, 
                 temperature: float = 1.0, 
                 top_k: int = 5,
                 repetition_penalty: float = 1.2) -> str:
        """
        柔軟なデコード戦略を用いた文章生成
        Args:
            temperature: 1.0より大きいとランダム性が増し、小さいと決定的になる
            top_k: スコア上位k個の候補からサンプリング
            repetition_penalty: 既出の単語のスコアを割り引く係数（1.0で無効）
        """
        token_ids = self.encoder.tokenizer.encode(prompt)
        if not token_ids: 
            return ""
            
        current_token = token_ids[-1]
        current_sdr = self.encoder._get_token_sdr(current_token)
        generated_words = []
        
        # 逆引き辞書の準備
        reverse_vocab = {v: k for k, v in self.encoder.tokenizer.vocab.items()}
        eos_id = self.encoder.tokenizer.vocab.get("<eos>", -1)
        unk_id = self.encoder.tokenizer.vocab.get("<unk>", -1)
        
        # 生成履歴（反復ペナルティ用）
        recent_tokens = [current_token]
        if len(token_ids) > 1:
            recent_tokens.extend(token_ids[-3:]) # プロンプトの最後の方も履歴に含める

        for _ in range(max_tokens):
            # 次のSDRを予測
            next_sdr = self.predict_next_sdr(current_sdr, context_sdr)
            if not next_sdr: 
                break

            next_set = set(next_sdr)
            candidates = []

            # 全語彙に対して重複度（スコア）を計算
            # ※本来は重いが、語彙数数千程度ならリアルタイム動作可能
            for tid, sdr in self.encoder.token_sdr_map.items():
                if tid == unk_id: continue # <unk>は生成しない
                
                # SDRの重なり（Overlap）をスコアとする
                overlap = len(next_set.intersection(sdr))
                
                # 足切り: ほとんど重なりがないものは無視（高速化）
                if overlap < 5: continue
                
                score = float(overlap)
                
                # 反復ペナルティの適用
                if tid in recent_tokens:
                    score /= repetition_penalty
                
                candidates.append((tid, score))

            if not candidates:
                break

            # Top-k フィルタリング
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = candidates[:top_k]

            # サンプリングによるトークン決定
            if temperature < 0.1:
                # 温度が非常に低い場合はGreedy（スコア最大のもの）
                best_token = top_candidates[0][0]
                best_score = top_candidates[0][1]
            else:
                # 重み付きランダムサンプリング
                # スコアを温度で調整: weight = score ^ (1/T)
                weights = [c[1] ** (1.0 / temperature) for c in top_candidates]
                try:
                    chosen_pair = random.choices(top_candidates, weights=weights, k=1)[0]
                    best_token = chosen_pair[0]
                    best_score = chosen_pair[1]
                except ValueError:
                    break

            # 最終的な品質チェック（SDRの再現度が低すぎる場合は生成停止）
            # 閾値はビット数依存だが、ここでは経験則として設定
            if best_score < 10: 
                break

            # 停止トークン判定
            if best_token == eos_id:
                break

            word = reverse_vocab.get(best_token, "")
            if not word: 
                break
                
            generated_words.append(word)
            
            # 履歴更新
            recent_tokens.append(best_token)
            if len(recent_tokens) > 10: # 履歴が長くなりすぎないように維持
                recent_tokens.pop(0)
                
            current_sdr = self.encoder._get_token_sdr(best_token)

            # 補助的な停止条件（ハードコードは最小限にし、基本は<eos>に任せる）
            # ただし、学習不足時の無限ループ防止として句点での強制終了はオプションとして残す設計もアリ
            if word == "。" and len(generated_words) > 10:
                pass # ここでは強制停止せず、<eos>の予測を待つ（またはmax_tokensで停止）

        return "".join(generated_words)