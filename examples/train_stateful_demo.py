_FILE_INFO = {
    "//": "ディレクトリパス: examples/train_stateful_demo.py",
    "//": "タイトル: Stateful SNN 教師あり学習デモ",
    "//": "目的: 特定のテキストに対する状態遷移を学習させる。"
}

from sara_engine import StatefulRLMAgent

def run_custom_training():
    print("=== Stateful SNN Training Demo ===")
    
    # Agentを介さず直接Brainを操作して微調整する例
    agent = StatefulRLMAgent()
    brain = agent.brain
    
    input_seq = ["what", "is", "the", "code"]
    print(f"Training on sequence: {input_seq}")
    
    # 教師あり学習のシミュレーション
    # (内部的な forward_step と重み更新を繰り返す)
    for epoch in range(10):
        brain.reset_state()
        for word in input_seq:
            sdr = brain.encoder.encode(word)
            # 学習モードで順伝播
            brain.forward_step(sdr, training=True)
        print(f"Epoch {epoch+1} done.")
        
    print("\nTraining complete.")

if __name__ == "__main__":
    run_custom_training()