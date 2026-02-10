# examples/chat_sara.py
# title: SARA Chat v17 (Dynamic Vocabulary & Associative Trigger)
# description: 未知語の学習と、連想トリガーによる会話維持を実装したデモ

import sys
import os
import time
import numpy as np

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
try:
    from sara_engine.sara_gpt_core import SaraGPT
except ImportError:
    print("Error: 'sara_engine' module not found.")
    sys.exit(1)

def main():
    # 基本語彙（シード）
    corpus = [
        "hello sara", "hello world", "sara is a good ai", "sara likes to learn",
        "the cat likes fish", "the dog likes meat", "the cat sleeps on bed",
        "the dog sleeps on floor", "fish is good food", "meat is good food",
        "i like the cat", "i like the dog", "good morning sara", "good night sara",
        "sara is smart", "ai is smart"
    ]
    
    # 語彙リストの初期化
    vocabulary = set()
    for sent in corpus:
        for w in sent.split():
            vocabulary.add(w)
    vocab_list = sorted(list(vocabulary))
    
    print("===========================================")
    print(f"  SARA v17: Dynamic Vocabulary Chat")
    print(f"  Initial Vocab: {len(vocab_list)} words")
    print("===========================================")

    engine = SaraGPT(sdr_size=1024)
    model_path = "sara_brain.pkl" 
    
    # 既存の脳があれば読み込む
    if os.path.exists(model_path):
        print(f"\nFound saved brain: {model_path}")
        print("Loading...", end=" ")
        engine.load_model(model_path)
        print("Done!")
    else:
        print("\nNo brain found. Training base knowledge...", end="", flush=True)
        start_time = time.time()
        epochs = 80
        for epoch in range(epochs):
            shuffled_corpus = np.random.permutation(corpus)
            for sentence in shuffled_corpus:
                words = sentence.split()
                engine.train_sequence(words)
            if (epoch+1) % 10 == 0: print(".", end="", flush=True)
        print(f"\nFinished in {time.time() - start_time:.1f}s")
        engine.save_model(model_path)

    print("\n--- Conversation Started ---")
    print("SARA will learn NEW words from you instantly.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip().lower()
            if user_input in ["exit", "quit", "bye"]:
                print("Saving memories...")
                engine.save_model(model_path)
                break
            if not user_input: continue
            
            # 入力単語の分解
            input_words = user_input.split()
            new_words = []
            
            # 未知語の動的登録
            for w in input_words:
                if w not in vocab_list and w != "<eos>":
                    vocab_list.append(w)
                    new_words.append(w)
            
            if new_words:
                vocab_list.sort()
                print(f"SARA: (I learned new words: {', '.join(new_words)})")
            
            # トリガー用の既知語リストを作成（ここが抜けていました）
            known_words = [w for w in input_words if w in vocab_list]

            # 1. Listen (Online Learning)
            # 強力に学習 (Mental Rehearsal)
            engine.listen(user_input, online_learning=True)
            
            # 2. Think & Speak
            # 既知の単語をトリガーにして思考を開始
            trigger_text = " ".join(known_words)
            response = engine.think(length=20, vocabulary=vocab_list, trigger_text=trigger_text)
            print(f"SARA: {response}")
            
            # 3. Relax
            engine.relax(steps=50)
            
        except KeyboardInterrupt:
            break

    print("\nGoodbye!")

if __name__ == "__main__":
    main()