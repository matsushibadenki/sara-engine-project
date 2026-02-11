# examples/test_v42.py
# title: v42 Learning Verification
# description: 学習機能をテスト

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
try:
    from sara_engine.sara_gpt_core import SaraGPT
except ImportError:
    print("Error: 'sara_engine' module not found.")
    sys.exit(1)

def test_learning():
    print("=" * 60)
    print("v42 Learning Test")
    print("=" * 60)
    
    engine = SaraGPT(sdr_size=1024)
    
    # 簡単なペア
    pairs = [
        ("hello", "world"),
        ("who", "are"),
        ("i", "am"),
        ("good", "morning"),
        ("thank", "you"),
    ]
    
    vocab = list(set([w for pair in pairs for w in pair]))
    print(f"Vocabulary: {vocab}\n")
    
    # 学習
    print("Training for 30 iterations...")
    for iteration in range(30):
        for word1, word2 in pairs:
            engine._learn_tokens([word1, word2], reset=True, boost=True)
        
        if (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration + 1}/30")
    
    print("\nTesting learned associations:")
    print("-" * 60)
    
    correct = 0
    total = len(pairs)
    
    for word1, word2 in pairs:
        engine.reset_state()
        
        # 入力
        sdr1 = engine.encoder.encode(word1)
        engine.forward_step(sdr1, training=False)
        
        # 予測
        predicted_sdr, _ = engine.forward_step([], training=False, force_output=True)
        predicted_word = engine.encoder.decode(predicted_sdr, vocab)
        
        result = "✓" if predicted_word == word2 else "✗"
        if predicted_word == word2:
            correct += 1
        
        print(f"{result} '{word1}' → '{predicted_word}' (expected: '{word2}')")
    
    accuracy = correct / total * 100
    print("-" * 60)
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    if accuracy >= 80:
        print("✓ PASS: Learning is working correctly!")
    else:
        print("✗ FAIL: Learning needs improvement.")
    
    print("=" * 60)

if __name__ == "__main__":
    test_learning()
