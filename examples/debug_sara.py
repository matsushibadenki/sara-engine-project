# examples/debug_sara.py
# title: SARA Debug Tool
# description: エンジンの内部状態を診断するツール

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from sara_engine.sara_gpt_core import SaraGPT

def debug_engine():
    print("=" * 60)
    print("SARA Engine Debug Tool")
    print("=" * 60)
    
    engine = SaraGPT(sdr_size=1024)
    
    # テストコーパス
    test_words = ["hello", "world", "who", "are", "you", "i", "am", "sara"]
    
    print("\n[1] Testing Encoder Consistency...")
    print("-" * 60)
    for word in test_words:
        sdr1 = engine.encoder.encode(word)
        sdr2 = engine.encoder.encode(word)
        
        # 一貫性チェック
        if sdr1 == sdr2:
            print(f"✓ '{word}': {len(sdr1)} bits (consistent)")
        else:
            print(f"✗ '{word}': INCONSISTENT! {len(sdr1)} vs {len(sdr2)}")
        
        # 重複チェック
        overlap_count = 0
        for other_word in test_words:
            if other_word == word:
                continue
            other_sdr = engine.encoder.encode(other_word)
            overlap = len(set(sdr1).intersection(other_sdr))
            if overlap > 5:  # 5ビット以上の重複
                overlap_count += 1
                print(f"  ⚠ High overlap with '{other_word}': {overlap} bits")
        
        if overlap_count == 0:
            print(f"  → Good orthogonality")
    
    print("\n[2] Testing Reservoir Activity...")
    print("-" * 60)
    
    # "hello world" を処理
    engine.reset_state()
    test_sentence = ["hello", "world"]
    
    total_spikes_per_layer = [0, 0, 0]
    
    for word in test_sentence:
        sdr = engine.encoder.encode(word)
        print(f"\nProcessing: '{word}' (SDR: {len(sdr)} bits active)")
        
        predicted_sdr, all_spikes = engine.forward_step(sdr, training=False)
        
        # 層ごとのスパイク数をカウント
        spikes_l1 = len([s for s in all_spikes if s < 2000])
        spikes_l2 = len([s for s in all_spikes if 2000 <= s < 4000])
        spikes_l3 = len([s for s in all_spikes if 4000 <= s < 6000])
        
        total_spikes_per_layer[0] += spikes_l1
        total_spikes_per_layer[1] += spikes_l2
        total_spikes_per_layer[2] += spikes_l3
        
        print(f"  L1 spikes: {spikes_l1} / 2000 ({spikes_l1/2000*100:.1f}%)")
        print(f"  L2 spikes: {spikes_l2} / 2000 ({spikes_l2/2000*100:.1f}%)")
        print(f"  L3 spikes: {spikes_l3} / 2000 ({spikes_l3/2000*100:.1f}%)")
        print(f"  Total spikes: {len(all_spikes)}")
        print(f"  Readout prediction: {len(predicted_sdr)} bits")
    
    print("\n" + "-" * 60)
    print("Average spikes per step:")
    for i, count in enumerate(total_spikes_per_layer):
        avg = count / len(test_sentence)
        print(f"  L{i+1}: {avg:.1f} spikes/step ({avg/2000*100:.1f}%)")
    
    # アラート
    if total_spikes_per_layer[0] < 50:
        print("\n⚠ WARNING: L1 activity is too low! Input connections may be weak.")
    if total_spikes_per_layer[1] < 50:
        print("⚠ WARNING: L2 activity is too low! Recurrent connections may be weak.")
    if total_spikes_per_layer[2] < 50:
        print("⚠ WARNING: L3 activity is too low! Network may be too stable.")
    
    print("\n[3] Testing Readout Weights...")
    print("-" * 60)
    
    # ランダムに10個のReadoutニューロンをサンプリング
    sample_indices = np.random.choice(engine.sdr_size, 10, replace=False)
    
    for idx in sample_indices:
        weights = engine.readout_weights[idx]['w']
        
        mean_w = np.mean(weights)
        max_w = np.max(weights)
        min_w = np.min(weights)
        nonzero = np.sum(weights > 0.01)
        
        print(f"Readout[{idx}]: mean={mean_w:.3f}, max={max_w:.3f}, "
              f"min={min_w:.3f}, active={nonzero}/{len(weights)}")
    
    print("\n[4] Testing Learning...")
    print("-" * 60)
    
    # 簡単なペアを学習
    engine.reset_state()
    train_pairs = [("hello", "world"), ("who", "are"), ("i", "am")]
    
    print("Training 3 word pairs for 20 iterations...")
    for iteration in range(20):
        for word1, word2 in train_pairs:
            engine._learn_tokens([word1, word2], reset=False, boost=True, stdp=True)
    
    # テスト
    print("\nTesting learned associations:")
    for word1, word2 in train_pairs:
        engine.reset_state()
        sdr1 = engine.encoder.encode(word1)
        engine.forward_step(sdr1, training=False)
        
        predicted_sdr, _ = engine.forward_step([], training=False, force_output=True)
        predicted_word = engine.encoder.decode(predicted_sdr, [w for pair in train_pairs for w in pair])
        
        result = "✓" if predicted_word == word2 else "✗"
        print(f"  {result} '{word1}' → predicted: '{predicted_word}' (expected: '{word2}')")
    
    print("\n" + "=" * 60)
    print("Diagnosis complete.")
    print("=" * 60)

if __name__ == "__main__":
    debug_engine()