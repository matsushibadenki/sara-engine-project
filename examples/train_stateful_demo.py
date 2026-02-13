_FILE_INFO = {
    "//": "ディレクトリパス: examples/train_stateful_demo.py",
    "//": "タイトル: Stateful SNN 学習デモ",
    "//": "目的: State-aware Readoutの学習プロセスを検証する。"
}

import sys
import os
import random

# プロジェクトルートへのパスを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")

from sara_engine.stateful_snn import StatefulSaraGPT

def run_training_demo():
    print("Initializing Stateful SaraGPT...")
    brain = StatefulSaraGPT(sdr_size=1024)
    
    # 語彙定義
    vocabulary = [
        "START", "SEARCH", "code", "READ", "CHUNK", "EXTRACT", "answer", "END",
        "What", "is", "the", "master", "override"
    ]
    
    # --- 学習データ定義 ---
    # シナリオ: "What is the code" という入力に対し、
    # 状態遷移: INIT -> SEARCH -> READ -> EXTRACT
    # アクション: START -> SEARCH code -> READ CHUNK -> EXTRACT answer
    
    input_seq = ["What", "is", "the", "code"]
    state_seq = ["INIT", "SEARCH", "READ", "EXTRACT"]
    action_seq = ["START", "SEARCH code", "READ CHUNK", "EXTRACT answer"]
    
    # Encoderのキャッシュを温める（初回エンコード）
    for w in vocabulary + input_seq + action_seq:
        brain.encoder.encode(w)
        
    print("\n--- Before Training ---")
    # 学習前の推論（デタラメなはず）
    result = brain.think_stateful(length=5, vocabulary=vocabulary, trigger_text="What is the code")
    print(f"Generated: {result}")
    
    print("\n--- Training Start ---")
    epochs = 20
    for epoch in range(epochs):
        brain.train_supervised(input_seq, state_seq, action_seq)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1} complete.")
            
    print("\n--- After Training ---")
    # 学習後の推論
    # 期待値: SEARCH code ... READ CHUNK ... のような流れ
    result = brain.think_stateful(length=8, vocabulary=vocabulary, trigger_text="What is the code")
    print(f"Generated: {result}")
    
    # モデル保存テスト
    os.makedirs("models", exist_ok=True)
    brain.save_model("models/stateful_demo.pkl")

if __name__ == "__main__":
    run_training_demo()