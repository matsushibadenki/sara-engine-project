_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_chat.py",
    "//": "タイトル: SARA チャット & 学習デモ",
    "//": "目的: SaraGPTクラスを使用した基本的なシーケンス学習と生成のデモ。"
}

import os
import numpy as np
from sara_engine import SaraGPT

def run_chat_demo():
    print("=== SARA Engine Chat Demo ===")
    
    model_path = "sara_brain.pkl"
    engine = SaraGPT(sdr_size=1024)
    
    corpus = [
        "hello sara", "i am an ai", "what is your name", "my name is sara",
        "learning is fun", "sara is smart", "i love learning"
    ]
    
    # トークナイザーのトレーニング
    if hasattr(engine.encoder, 'tokenizer'):
        engine.encoder.tokenizer.train(corpus)
    
    # 語彙リスト（デコード用）
    vocab = set()
    for sent in corpus:
        for w in sent.split(): vocab.add(w)
    vocab_list = sorted(list(vocab))

    if os.path.exists(model_path):
        print(f"Loading brain from {model_path}...")
        engine.load_model(model_path)
    else:
        print("Training on basic corpus...")
        for epoch in range(100):
            for sent in corpus:
                # 入力シーケンスをSDRに変換して順伝播（学習モード）
                for word in sent.split():
                    sdr = engine.encoder.encode(word)
                    engine.forward_step(sdr, training=True)
        engine.save_model(model_path)

    print("\nCommands: 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ").strip().lower()
        if user_input in ["exit", "quit"]: break
        if not user_input: continue
        
        # Listening
        for word in user_input.split():
            sdr = engine.encoder.encode(word)
            engine.forward_step(sdr, training=False)
            
        # Generating
        print("SARA: ", end="", flush=True)
        input_sdr = []
        for _ in range(10):
            # force_output=True で強制的に発火させる
            predicted_sdr, _ = engine.forward_step(input_sdr, training=False, force_output=True)
            next_word = engine.encoder.decode(predicted_sdr, vocab_list + ["<eos>"])
            
            if next_word == "<eos>": break
            print(f"{next_word} ", end="", flush=True)
            input_sdr = engine.encoder.encode(next_word)
        print()

if __name__ == "__main__":
    run_chat_demo()