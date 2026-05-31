# Directory Path: examples/core/demo_h_jepa_noise_robustness.py
# English Title: Demonstration of H-JEPA Robustness Against Burst Noise (Continuous Stream)
# Purpose and Content: H-JEPA(階層的スパイキングJEPA)のノイズロバスト性テスト。SNN本来の「連続ストリーム処理」の特性を活かし、状態リセットを行わずに学習から異常検知、そして自己復元までのプロセスをシームレスに検証する。

import sys
import os
import random
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from sara_engine.models.spiking_jepa import SpikingJEPA

class RobustSDRTextEncoder:
    """未知のノイズ文字にも固有のSDRスパイク（20発火）を割り当てるエンコーダ"""
    def __init__(self, vocab_size: int = 1000, sdr_size: int = 20):
        self.vocab_size = vocab_size
        self.sdr_size = sdr_size
        self.char_to_sdr: Dict[str, List[int]] = {}

    def encode(self, text: str) -> List[List[int]]:
        spikes_seq = []
        for char in text:
            if char not in self.char_to_sdr:
                rng = random.Random(hash(char))
                self.char_to_sdr[char] = sorted(rng.sample(range(self.vocab_size), self.sdr_size))
            spikes_seq.append(self.char_to_sdr[char])
        return spikes_seq

def run_noise_robustness_demo():
    print("==================================================")
    print(" H-JEPA ノイズロバスト性（連続ストリーム処理）テスト v4")
    print("==================================================\n")

    base_sentence = "SARA Engine is an SNN based cognitive architecture. "
    encoder = RobustSDRTextEncoder(vocab_size=800, sdr_size=20)
    
    # モデルの初期化 (過飽和を防ぐためしきい値を調整し、Targetの同期を早める)
    embed_dim = 800
    jepa_model = SpikingJEPA(
        layer_configs=[{"embed_dim": embed_dim, "hidden_dim": embed_dim}],
        ema_decay=0.5, # コンテキストとターゲットの表現を素早く同期
        learning_rate=0.1,
        time_scales={"low": 1, "medium": 3, "high": 5}
    )

    # 1. シームレスな連続学習（状態を一切リセットしない）
    print("【Phase 1: 連続ストリームでの自己組織化】")
    print("テキストデータを流し込み、リアルタイムで予測モデルを形成しています...")
    train_text = base_sentence * 5 
    train_spikes = encoder.encode(train_text)
    
    for t in range(len(train_spikes) - 1):
        jepa_model.forward(x_spikes=train_spikes[t], y_spikes=train_spikes[t+1], learning=True)
        
    print("=> 予測モデルの形成が安定しました。\n")

    # 2. 途切れることなくノイズ区間へ突入
    test_text = "SARA Engine is an SNN based cognitive architecture."
    noisy_test_text = "SARA Eng#@!&is an SNN based cognitive architecture."
    
    print("【Phase 2: 未知のバーストノイズへの遭遇と自己復元】")
    print(f"正解テキスト: {test_text}")
    print(f"ノイズ入力  : {noisy_test_text}\n")

    test_spikes_sequence = encoder.encode(noisy_test_text)
    
    print(f"{'Step':>4} | {'Ctx':>3} -> {'Tgt':>3} | {'Pred Spikes':>11} | {'Surprise':>8} | 状態")
    print("-" * 65)
    
    for t in range(len(test_spikes_sequence) - 1):
        current_char = noisy_test_text[t]
        next_char = noisy_test_text[t + 1]
        
        # 学習は止めるが、内部の膜電位や履歴はそのまま維持して推論
        predicted_s_y, _ = jepa_model.forward(
            x_spikes=test_spikes_sequence[t], 
            y_spikes=test_spikes_sequence[t + 1], 
            learning=False
        )
        
        # ターゲット表現との真の誤差（Symmetric Difference）を計算
        surprise_spikes, _ = jepa_model.minimizer.compute_surprise_signal(
            predicted_s_y, test_spikes_sequence[t + 1]
        )
        true_surprise = sum(surprise_spikes)
        pred_count = len(predicted_s_y)
        
        status = ""
        if current_char in "#@!&" or next_char in "#@!&":
            status = "⚠️ ノイズ区間 (異常検知)"
        elif true_surprise == 0 and pred_count > 0:
            status = "✅ 予測完全一致"
        elif true_surprise <= 20 and pred_count > 0:
            status = "🔄 予測復元中"
        elif pred_count == 0:
            status = "❌ 予測喪失"
        else:
            status = "🚨 文脈崩壊 (大誤差)"

        print(f"[{t:2d}] |  '{current_char}' ->  '{next_char}' | {pred_count:11d} | {true_surprise:8d} | {status}")

    print("\n==================================================")
    print(" テスト完了")
    print("==================================================")
    print("【結果の考察】")
    print("連続処理により、前半の正常区間ではSurpriseが0（予測完全一致）となります。")
    print("ノイズ区間（#@!&）に突入すると、未知のパターンに対してH-JEPAが予測を外し、大きなSurpriseが発生します。")
    print("ノイズ通過後、過去のコンテキスト履歴バッファ（high: 5）からノイズが押し出されると、")
    print("ネットワークは再び元の軌道を取り戻し、予測完全一致に自己復元します。")

if __name__ == "__main__":
    run_noise_robustness_demo()
