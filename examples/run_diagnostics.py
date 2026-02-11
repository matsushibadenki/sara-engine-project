# examples/run_diagnostics.py
# 診断・デバッグ・テストツール (統合版)

import sys
import os
import argparse
import numpy as np

# ユーティリティ
try:
    from utils import setup_path
except ImportError:
    # 修正: type: ignore 追加
    from .utils import setup_path # type: ignore

setup_path()

try:
    from sara_engine import SaraGPT
except ImportError:
    print("Error: 'sara_engine' module not found.")
    sys.exit(1)

def run_debug_tool():
    print("=" * 60)
    print("SARA Engine Debug Tool")
    print("=" * 60)
    
    engine = SaraGPT(sdr_size=1024)
    test_words = ["hello", "world", "who", "are", "you", "i", "am", "sara"]
    
    print("\n[1] Testing Encoder Consistency...")
    print("-" * 60)
    for word in test_words:
        sdr1 = engine.encoder.encode(word)
        sdr2 = engine.encoder.encode(word)
        if sdr1 == sdr2:
            print(f"✓ '{word}': {len(sdr1)} bits (consistent)")
        else:
            print(f"✗ '{word}': INCONSISTENT!")
            
    print("\n[2] Testing Reservoir Activity (Layer Spikes)...")
    print("-" * 60)
    engine.reset_state()
    test_sentence = ["hello", "world"]
    
    for word in test_sentence:
        sdr = engine.encoder.encode(word)
        _, all_spikes = engine.forward_step(sdr, training=False)
        
        l1 = len([s for s in all_spikes if s < 2000])
        l2 = len([s for s in all_spikes if 2000 <= s < 4000])
        l3 = len([s for s in all_spikes if 4000 <= s < 6000])
        
        print(f"Word '{word}': Total={len(all_spikes)} | L1={l1}, L2={l2}, L3={l3}")
        if len(all_spikes) < 50:
            print("  ⚠ Warning: Low activity detected.")

    print("\n[3] Testing Readout Weights...")
    print("-" * 60)
    indices = np.random.choice(engine.sdr_size, 5, replace=False)
    for idx in indices:
        w = engine.readout_weights[idx]['w']
        print(f"Neuron[{idx}]: Mean Weight={np.mean(w):.4f}, Max={np.max(w):.4f}")

    print("\nDiagnosis Complete.")

def run_learning_test():
    print("=" * 60)
    print("Learning Capability Test")
    print("=" * 60)
    
    engine = SaraGPT(sdr_size=1024)
    pairs = [
        ("hello", "world"),
        ("good", "morning"),
        ("thank", "you")
    ]
    vocab = list(set([w for pair in pairs for w in pair]))
    
    print("Training 3 word pairs (30 iterations)...")
    for _ in range(30):
        for w1, w2 in pairs:
            engine._learn_tokens([w1, w2], reset=True, boost=True)
            
    print("\nTesting Associations:")
    correct = 0
    for w1, w2 in pairs:
        engine.reset_state()
        sdr = engine.encoder.encode(w1)
        engine.forward_step(sdr, training=False)
        
        pred_sdr, _ = engine.forward_step([], training=False, force_output=True)
        pred_word = engine.encoder.decode(pred_sdr, vocab)
        
        res = "✓" if pred_word == w2 else "✗"
        print(f"{res} '{w1}' -> '{pred_word}' (Expected: '{w2}')")
        if pred_word == w2: correct += 1
        
    acc = correct / len(pairs) * 100
    print(f"\nAccuracy: {acc:.1f}%")
    if acc >= 66:
        print("PASS: Basic learning works.")
    else:
        print("FAIL: Learning needs improvement.")

def main():
    parser = argparse.ArgumentParser(description="SARA Diagnostics Tool")
    parser.add_argument("mode", choices=["debug", "test"], help="Mode: debug info or learning test")
    args = parser.parse_args()
    
    if args.mode == "debug":
        run_debug_tool()
    elif args.mode == "test":
        run_learning_test()

if __name__ == "__main__":
    main()