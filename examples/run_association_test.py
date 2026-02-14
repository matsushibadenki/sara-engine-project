_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_association_test.py",
    "//": "タイトル: SNN連想（予測）テスト",
    "//": "目的: 学習したシーケンスの一部から、モデルが続きを想起できるかを確認する。"
}

import os
import sys
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from sara_engine.core.transformer import PlasticTransformerBlock

def get_char_spikes(char: str, d_model: int):
    rng = np.random.RandomState(ord(char))
    return rng.choice(d_model, max(1, int(d_model * 0.05)), replace=False).tolist()

def calculate_overlap(list1, list2):
    set1, set2 = set(list1), set(list2)
    if not set1: return 0.0
    return len(set1.intersection(set2)) / len(set1)

def main():
    d_model = 256
    model = PlasticTransformerBlock(d_model=d_model, num_heads=4)
    
    chars = ["S", "A", "R", "A"]
    char_spikes = {c: get_char_spikes(c, d_model) for c in chars}
    
    print("--- Phase 1: Training ---")
    # シーケンス "SARA" を繰り返し学習
    for epoch in range(30):
        model.reset()
        for t, c in enumerate(chars):
            model.compute(char_spikes[c], pos=t, learning=True)
            
    print("--- Phase 2: Association (Prediction) ---")
    model.reset()
    # 「S」を入力して、次の文字を予測させる
    input_char = "S"
    print(f"Input: '{input_char}'")
    
    # 時刻 t=0 で 'S' を入力
    _ = model.compute(char_spikes[input_char], pos=0, learning=False)
    
    # 時刻 t=1 でモデルが何を連想するか確認
    prediction_spikes = model.associate([], pos=1)
    
    # 各文字の正解パターンとの一致度（Overlap）を計算
    print("\nOverlap with target patterns:")
    for c in chars:
        overlap = calculate_overlap(char_spikes[c], prediction_spikes)
        print(f"Pattern '{c}': {overlap:.2%}")

    # 最も一致度が高いものを予測結果とする
    best_char = max(chars, key=lambda c: calculate_overlap(char_spikes[c], prediction_spikes))
    print(f"\nResult: Predicted next character is likely '{best_char}'")

if __name__ == "__main__":
    main()