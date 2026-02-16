# examples/demo_rl_training.py
# 強化学習（RL）モジュールの学習デモ
# 状態空間モデルと強化学習を組み合わせたRLM (Reinforcement Learning Model) の基本的な学習ループを回すデモです。

import numpy as np
from sara_engine.models.rlm import RLM

def create_dummy_env():
    # 非常に単純なダミー環境の状態と報酬を返す関数
    state = np.random.randn(32)
    reward = np.random.choice([-1.0, 0.0, 1.0])
    done = np.random.rand() > 0.9
    return state, reward, done

def main():
    print("=== RLM 強化学習デモンストレーション ===")
    
    state_dim = 32
    action_dim = 4
    num_episodes = 5
    max_steps = 20
    
    print("RLMエージェントを初期化中...")
    agent = RLM(state_dim=state_dim, action_dim=action_dim)
    
    print("\n学習ループを開始します...")
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        
        # 環境の初期状態
        state, _, _ = create_dummy_env()
        total_reward = 0
        
        for step in range(max_steps):
            # 行動の選択
            action = agent.select_action(state)
            
            # 環境のステップ進行
            next_state, reward, done = create_dummy_env()
            total_reward += reward
            
            # 経験の保存と学習
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            
            if step % 5 == 0:
                print(f"  Step {step}: Action={action}, Reward={reward:.1f}")
                
            if done:
                print(f"  エピソード終了 (Doneフラグ到達). 完了ステップ: {step + 1}")
                break
                
        print(f"Episode {episode + 1} 獲得報酬合計: {total_reward:.1f}")

    print("\n強化学習デモが完了しました。")

if __name__ == "__main__":
    main()