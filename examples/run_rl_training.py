_FILE_INFO = {
    "//": "ディレクトリパス: examples/run_rl_training.py",
    "//": "タイトル: RLM 強化学習ループ",
    "//": "目的: エージェントが試行錯誤を通じて状態遷移を学習するか検証する。"
}

import sys
import os
import random

from sara_engine import StatefulRLMAgent

def run_rl_loop():
    print("=== Stateful RLM: Reinforcement Learning Loop ===")
    
    # モデルの準備（初期状態またはデモ後のモデル）
    model_path = "models/stateful_demo.pkl"
    agent = StatefulRLMAgent(model_path=model_path)
    
    document = (
        "Project SARA log. Sector 1 clear. "
        "Confidential: The master override code is BLUE-OCEAN-42. "
        "End of log."
    )
    query = "What is the master code?"
    
    episodes = 50
    success_count = 0
    
    print(f"\nTraining for {episodes} episodes...")
    
    for episode in range(episodes):
        # 実行（学習有効）
        result = agent.solve(query, document, train_rl=True)
        
        # 結果判定
        is_success = "BLUE-OCEAN" in result
        if is_success:
            success_count += 1
            status = "SUCCESS"
        else:
            status = "FAIL"
            
        # 遷移確率の確認（INIT -> SEARCH が強化されたか？）
        tm = agent.brain.state_neurons.transition_matrix
        idx_init = agent.brain.state_neurons.get_state_index("INIT")
        idx_search = agent.brain.state_neurons.get_state_index("SEARCH")
        prob_init_search = tm[idx_init, idx_search]
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1:02d}: {status} | P(INIT->SEARCH) = {prob_init_search:.4f}")

    print("\n" + "="*30)
    print(f"Training Complete. Success Rate: {success_count}/{episodes}")
    print("="*30)
    
    # 学習後の遷移行列の一部を表示
    print("\nFinal Transition Matrix (INIT row):")
    idx_init = agent.brain.state_neurons.get_state_index("INIT")
    probs = agent.brain.state_neurons.transition_matrix[idx_init]
    states = agent.brain.state_neurons.state_names
    for s, p in zip(states, probs):
        print(f"  INIT -> {s}: {p:.4f}")
        
    # 保存
    agent.brain.save_model("models/stateful_rl_trained.pkl")

if __name__ == "__main__":
    run_rl_loop()