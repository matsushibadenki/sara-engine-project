# examples/chat_sara.py
# title: SARA Chat v33 (Full Knowledge Base)
# description: 拡張された学習データを持つチャットボット。必ず更新してください。

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
    # --- 拡張コーパス (ここが重要！) ---
    corpus = [
        # Greetings
        "hello sara", "hello world", "hi sara", "hi there",
        "good morning sara", "good afternoon", "good evening", "good night",
        "how are you?", "i am fine", "i am good",
        
        # Self Identification
        "who are you?", "i am sara", "i am an ai", "i am a brain",
        "what is your name?", "my name is sara",
        "are you human?", "no i am ai", "i am a machine",
        
        # Simple Facts
        "sara is smart", "sara likes to learn", "ai is intelligence",
        "the cat likes fish", "the dog likes meat",
        "birds fly in sky", "fish swim in water",
        "sun is hot", "ice is cold", "fire is hot",
        "water is wet", "earth is round",
        
        # Preferences
        "i like cats", "i like dogs", "i love learning",
        "cats are cute", "dogs are loyal", "books are good",
        
        # Abstract
        "what is love?", "love is good",
        "what is time?", "time is flowing",
        "knowledge is power", "thinking is fun"
    ]
    
    # 語彙リストの初期化
    vocabulary = set()
    for sent in corpus:
        for w in sent.split():
            vocabulary.add(w)
    vocab_list = sorted(list(vocabulary))
    
    print("===========================================")
    print(f"  SARA v33: Full Knowledge Base")
    print(f"  Initial Vocab: {len(vocab_list)} words")
    print(f"  Training Corpus: {len(corpus)} sentences")
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
        print(f"\nNo brain found. Starting training on {len(corpus)} sentences...")
        start_time = time.time()
        epochs = 80 
        
        for epoch in range(epochs):
            shuffled_corpus = np.random.permutation(corpus)
            for sentence in shuffled_corpus:
                words = sentence.split()
                engine.train_sequence(words)
            
            elapsed = time.time() - start_time
            avg_time = elapsed / (epoch + 1)
            remaining = avg_time * (epochs - epoch - 1)
            
            bar_len = 20
            progress = (epoch + 1) / epochs
            filled_len = int(bar_len * progress)
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            
            sys.stdout.write(f"\r[{bar}] Epoch {epoch+1}/{epochs} | Time: {elapsed:.1f}s | ETA: {remaining:.1f}s ")
            sys.stdout.flush()
            
        print(f"\n\nTraining Finished in {time.time() - start_time:.1f}s")
        engine.save_model(model_path)

    print("\n--- Conversation Started ---")
    print("SARA will learn NEW words from you instantly.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip().lower()
            if user_input in ["exit", "quit", "bye"]:
                print("Dreaming and saving memories...")
                engine.dream(cycles=10) 
                engine.save_model(model_path)
                break
            if not user_input: continue
            
            # 入力単語の分解
            input_words = user_input.split()
            new_words = []
            
            for w in input_words:
                if w not in vocab_list and w != "<eos>":
                    vocab_list.append(w)
                    new_words.append(w)
            
            if new_words:
                vocab_list.sort()
                print(f"SARA: (I learned new words: {', '.join(new_words)})")
            
            known_words = [w for w in input_words if w in vocab_list]

            # 1. Listen
            engine.listen(user_input, online_learning=True)
            
            # 2. Think
            trigger_text = " ".join(known_words)
            response = engine.think(length=20, vocabulary=vocab_list, trigger_text=trigger_text)
            print(f"SARA: {response}")
            
            # 3. Relax & Dream
            print("  (Dreaming...)")
            # v39: Increase dream cycles to consolidate memory
            engine.dream(cycles=5) 
            engine.relax(steps=50)
            
        except KeyboardInterrupt:
            break

    print("\nGoodbye!")

if __name__ == "__main__":
    main()