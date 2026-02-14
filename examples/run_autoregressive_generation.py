# -*- coding: utf-8 -*-
_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_autoregressive_generation.py",
    "//": "タイトル: 整数演算モード 自己回帰生成デモ",
    "//": "目的: 浮動小数点を排除した固定小数点モデルで S->A->R->A が再現されるか検証する。"
}

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from sara_engine.core.transformer import PlasticTransformerBlock
from sara_engine.utils.encoder import SpikeEncoder

def calculate_overlap(list1, list2):
    s1, s2 = set(list1), set(list2)
    return len(s1.intersection(s2)) / len(s1) if s1 else 0

def decode_spikes(spikes, char_patterns):
    best_char = "?"
    max_ov = 0.0
    for char, pattern in char_patterns.items():
        ov = calculate_overlap(pattern, spikes)
        if ov > max_ov:
            max_ov = ov
            best_char = char
    return best_char

def main():
    d_model = 256
    model = PlasticTransformerBlock(d_model=d_model, num_heads=4)
    encoder = SpikeEncoder(d_model)
    
    target_text = "SARA"
    char_patterns = {c: encoder.text_to_temporal_spikes(c)[0] for c in set(target_text)}
    
    sequence_spikes = [char_patterns[char] for char in target_text]
    
    print("=== Integer Arithmetic (Fixed-Point) Mode ===")
    model.fit(sequence_spikes, epochs=200)

    save_path = os.path.join(project_root, "sara_model_weights_int.json")
    print(f"\nSaving INT model weights to {save_path}...")
    model.save(save_path)
    
    print("Resetting model states and reloading INT weights from JSON...")
    model.reset()
    model.load(save_path)

    print("\nAutoregressive Generation from 'S'...")
    initial_spikes = char_patterns["S"]
    
    # 修正内容:
    # model.predict で生のスパイク出力をそのまま次ステップに入力すると、
    # k_o の拡張による余剰スパイクやVetoの影響でノイズが蓄積し崩壊します。
    # そこで、毎ステップ「デコード(記号化)」して一番近い文字を確定させ、
    # その純粋な(クリーンな)文字スパイクを次の入力とするError Correctionループに置き換えます。
    
    model.reset()
    current_spikes = initial_spikes
    generated_sequence = ["S"]
    
    # S->A->R->A のため3ステップ実行
    for t in range(1, 4):
        # 1ステップ分の生成
        next_spikes = model.generate_next(current_spikes, pos=t)
        
        # 記号にデコードし、最も確からしい文字を確定
        best_char = decode_spikes(next_spikes, char_patterns)
        generated_sequence.append(best_char)
        
        # 次の推論入力には、ノイズを除去したクリーンな文字スパイクパターンを使用する
        current_spikes = char_patterns[best_char]
        
    print(f"Generated Result: {' -> '.join(generated_sequence)}")

if __name__ == "__main__":
    main()