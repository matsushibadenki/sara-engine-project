_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_memory_consolidation.py",
    "//": "タイトル: 皮質-海馬連動システムによるパターン補完テスト",
    "//": "目的: STDPで練度が上がるほど、欠損データからの記憶検索(LTM)精度が上がることを証明する。"
}

import os
import sys
import random
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sara_engine.core.cortex import CorticalColumn
from sara_engine.memory.hippocampus import CorticoHippocampalSystem

def generate_mock_sdr(seed_str: str, size: int = 1000, density: float = 0.05) -> List[int]:
    random.seed(seed_str)
    return random.sample(range(size), int(size * density))

def inject_noise(sdr: List[int], keep_ratio: float = 0.5) -> List[int]:
    """SDRの情報を半分欠損させる（不完全な記憶の再現）"""
    return random.sample(sdr, int(len(sdr) * keep_ratio))

def run_test():
    input_size = 1000
    cortex = CorticalColumn(input_size=input_size, hidden_size_per_comp=500, compartment_names=["vision_expert"])
    brain = CorticoHippocampalSystem(cortex=cortex)
    
    # 視覚入力（完全な状態）
    full_apple_sdr = generate_mock_sdr("apple_image", size=input_size)
    # 暗闇で見えにくい状態（情報が50%欠損した入力）
    noisy_apple_sdr = inject_noise(full_apple_sdr, keep_ratio=0.5)
    
    print("=== [フェーズ1] 未学習（練度0）での記憶と想起 ===")
    # 1回だけ見てLTMに保存する
    brain.experience_and_memorize(full_apple_sdr, content="This is an Apple.", context="vision_expert", learning=True)
    
    # ノイズの入ったSDRで思い出せるか？
    results_untrained = brain.recall_with_pattern_completion(noisy_apple_sdr, context="vision_expert")
    score_untrained = results_untrained[0]['score'] if results_untrained else 0.0
    print(f"未学習時のノイズ入力からの検索スコア: {score_untrained:.3f}")
    
    print("\n=== [フェーズ2] STDPによる学習（練度向上） ===")
    # 何度もリンゴを見て、皮質のリカレント結合を成長（練度上げ）させる
    print("皮質のシナプスを強化中...")
    for _ in range(20):
        brain.experience_and_memorize(full_apple_sdr, content="This is an Apple.", context="vision_expert", learning=True)
        
    print("\n=== [フェーズ3] 学習後（練度MAX）での想起 ===")
    # 皮質が「リンゴのパターン」を完全に覚えた状態で、再度ノイズ入力から検索する
    results_trained = brain.recall_with_pattern_completion(noisy_apple_sdr, context="vision_expert")
    score_trained = results_trained[0]['score'] if results_trained else 0.0
    print(f"学習後のノイズ入力からの検索スコア: {score_trained:.3f}")
    
    print("\n--- 結論 ---")
    if score_trained > score_untrained:
        print(f">> 成功: 練度が上がったことで皮質のパターン補完が働き、LTMへの検索精度が {((score_trained - score_untrained) / score_untrained * 100):.1f}% 向上しました！")
    else:
        print(">> 失敗: 検索精度が向上していません。")

if __name__ == "__main__":
    run_test()