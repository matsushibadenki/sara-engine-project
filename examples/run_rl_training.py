_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_rl_training.py",
    "//": "タイトル: 強化学習トレーニング",
    "//": "目的: エージェントが試行錯誤でQAを学習する様子を観察する。"
}

import os
from sara_engine import StatefulRLMAgent

def run_training():
    print("=== SARA Reinforcement Learning Training ===")
    
    os.makedirs("models", exist_ok=True)
    model_path = "models/training_test.pkl"
    agent = StatefulRLMAgent(model_path=model_path if os.path.exists(model_path) else None)
    
    doc = "The access key is GOLDEN-GATE."
    query = "What is the access key?"
    
    episodes = 20
    print(f"Training for {episodes} episodes...")
    
    for ep in range(episodes):
        # train_rl=True で報酬に基づき遷移行列を更新
        result = agent.solve(query, doc, train_rl=True)
        
        success = "GOLDEN-GATE" in result
        status = "SUCCESS" if success else "FAIL"
        
        # 状態遷移の傾向を表示（INIT -> SEARCH などの確率）
        tm = agent.brain.state_neurons.transition_matrix
        idx_init = agent.brain.state_neurons.get_state_index("INIT")
        idx_search = agent.brain.state_neurons.get_state_index("SEARCH")
        p = tm[idx_init, idx_search]
        
        print(f"Episode {ep+1:02d}: {status} | P(INIT->SEARCH) = {p:.4f}")

    agent.brain.save_model(model_path)
    print("\nTraining complete and model saved.")

if __name__ == "__main__":
    run_training()