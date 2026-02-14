_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_autoregressive_generation.py",
    "//": "タイトル: 因果遷移・自己回帰生成デモ (恒常性・不応期版)",
    "//": "目的: 教師あり感覚クランピングで学習したモデルが S->A->R->A を正確に想起するか検証する。"
}

import os
import sys
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from sara_engine.core.transformer import PlasticTransformerBlock
from sara_engine.utils.encoder import SpikeEncoder

def calculate_overlap(list1, list2):
    s1, s2 = set(list1), set(list2)
    return len(s1.intersection(s2)) / len(s1) if s1 else 0

def main():
    d_model = 256
    model = PlasticTransformerBlock(d_model=d_model, num_heads=4)
    encoder = SpikeEncoder(d_model)
    
    target_text = "SARA"
    char_patterns = {c: encoder.text_to_temporal_spikes(c)[0] for c in set(target_text)}
    
    print(f"Phase 1: Learning sequence '{target_text}' (200 epochs)...")
    for epoch in range(200):
        model.reset()
        for t, char in enumerate(target_text):
            model.compute(char_patterns[char], pos=t, learning=True)

    print("\nPhase 2: Autoregressive Generation...")
    model.reset()
    
    current_input = char_patterns["S"]
    generated_sequence = ["S"]
    
    for t in range(1, 4):
        # 現在の「刺激(current_input)」から「次の状態」を連想する
        next_spikes = model.generate_next(current_input, pos=t)
        
        # デコード
        best_char = "?"
        max_ov = 0.0
        for char, pattern in char_patterns.items():
            ov = calculate_overlap(pattern, next_spikes)
            if ov > max_ov:
                max_ov = ov
                best_char = char
        
        generated_sequence.append(best_char)
        # 次のステップのために出力を入力へ戻す
        current_input = next_spikes
        
    print(f"Generated Result: {' -> '.join(generated_sequence)}")

if __name__ == "__main__":
    main()