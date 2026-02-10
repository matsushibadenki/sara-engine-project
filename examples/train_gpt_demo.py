# examples/train_gpt_demo.py
# title: SARA-GPT Training Demo
# description: 階層型リザーバによる簡易的な文章生成実験

import sys
import os
import time

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
try:
    from sara_engine.sara_gpt_core import SaraGPT
except ImportError:
    # 同一ディレクトリに置いた場合
    from sara_gpt_core import SaraGPT

def main():
    # 1. 小規模なコーパスの定義
    # 同じような文脈で異なる単語が続くパターンを含める
    corpus = [
        "the cat likes fish",
        "the dog likes meat",
        "the cat sleeps on bed",
        "the dog sleeps on floor",
        "fish is good food",
        "meat is good food",
        "i like the cat",
        "i like the dog",
    ]
    
    # 語彙リストの作成
    vocabulary = set()
    for sent in corpus:
        for w in sent.split():
            vocabulary.add(w)
    vocab_list = list(vocabulary)
    print(f"Vocabulary Size: {len(vocab_list)}")
    print(f"Corpus: {corpus}")

    # 2. SARA-GPTの初期化
    # SDRサイズ1024, 3層構造
    engine = SaraGPT(sdr_size=1024)
    
    print("\n--- Start Training ---")
    start_time = time.time()
    
    epochs = 50
    for epoch in range(epochs):
        # データセットをループ
        for sentence in corpus:
            words = sentence.split()
            engine.train_sequence(words)
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} completed.")
            
    print(f"Training finished in {time.time() - start_time:.2f}s")

    # 3. 生成テスト
    print("\n--- Generation Test ---")
    
    # Test 1: "the cat" -> ? (expects "likes" or "sleeps")
    engine.generate("the cat", length=3, vocabulary=vocab_list)
    
    # Test 2: "the dog" -> ? (expects "likes" or "sleeps")
    engine.generate("the dog", length=3, vocabulary=vocab_list)
    
    # Test 3: "fish is" -> ? (expects "good food")
    engine.generate("fish is", length=2, vocabulary=vocab_list)

    # Test 4: 未知の文脈からの推論
    # "i like" -> ? (expects "the cat" or "the dog")
    engine.generate("i like", length=2, vocabulary=vocab_list)

if __name__ == "__main__":
    main()