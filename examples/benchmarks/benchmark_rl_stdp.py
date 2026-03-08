# ディレクトリパス: examples/benchmarks/benchmark_rl_stdp.py
# ファイルの日本語タイトル: 報酬変調STDPベンチマーク
# ファイルの目的や内容: 誤差逆伝播法を排除し、遅延報酬のみ（正解ラベルなし）でSNNが正しい行動を選択できるようになるかテストするスクリプト。最新のRewardModulatedSTDPManagerを統合し、ドーパミン信号と適格度トレースによる3要素学習則を検証する。

import random
from typing import List, Dict
from sara_engine.learning.reward_modulated_stdp import RewardModulatedSTDPManager

def run_rl_benchmark():
    print("=== SARA Engine: Reward-Modulated STDP (R-STDP) Benchmark ===\n")
    
    # 状態数2 (State 0, State 1) -> 行動数3 (Action 0, Action 1, Action 2)
    # スパースな辞書表現による重み管理: weights[pre_id][post_id] = weight
    # 初期重みは全て均等 (0.5) に設定
    weights: List[Dict[int, float]] = [
        {0: 0.5, 1: 0.5, 2: 0.5},  # State 0 からの結合
        {0: 0.5, 1: 0.5, 2: 0.5}   # State 1 からの結合
    ]
    
    # 先ほど評価した最新のR-STDPマネージャーをインスタンス化
    manager = RewardModulatedSTDPManager(
        learning_rate=0.2,
        w_min=0.0,
        w_max=3.0,
        tau_eligibility=5.0, # トレースの減衰を早めに設定（即時報酬に合わせる）
        tau_dopamine=5.0,
        a_plus=1.0,
        a_minus=0.5,
        homeostatic_target=1.5, # 行動の重み合計が1.5になるよう恒常性を維持
        homeostatic_rate=0.05
    )
    
    # 目標とする未知の環境ルール:
    # State 0 の時は Action 1 が正解 (報酬 +1.0)
    # State 1 の時は Action 2 が正解 (報酬 +1.0)
    # それ以外は失敗 (ペナルティ -1.0)
    def get_reward(state: int, action: int) -> float:
        if state == 0 and action == 1:
            return 1.0
        elif state == 1 and action == 2:
            return 1.0
        else:
            return -1.0

    epochs = 150
    current_time = 0.0
    print("[*] Training with R-STDP (No BP, no target labels)...")
    
    for epoch in range(epochs):
        state = random.choice([0, 1])
        
        # --- SNNによる行動選択 (発火モデルの簡略シミュレーション) ---
        action_potentials = {0: 0.0, 1: 0.0, 2: 0.0}
        for act, w in weights[state].items():
            action_potentials[act] += w
            
        # 探索(Exploration)フェーズ：確率的にランダムな行動をとるか、最大電位の行動をとる
        if random.random() < 0.2:
            action = random.choice([0, 1, 2])
        else:
            action = max(action_potentials, key=action_potentials.get)
            
        # --- 学習フェーズ: 適格度トレースの記録 ---
        # 状態(pre)と行動(post)の発火を記録し、痕跡を残す
        manager.record_spikes(pre_spikes=[state], post_spikes=[action], current_time=current_time)
        
        # --- 環境からのフィードバックとドーパミン放出 ---
        reward = get_reward(state, action)
        dopamine = manager.deliver_reward(reward)
        
        # --- 重みの更新と時間の進行 ---
        updated_count = manager.apply_weight_updates(weights)
        manager.step()
        current_time += 1.0
        
    print("\n[*] Evaluation Phase (Learning Disabled)")
    success = True
    
    for test_state in [0, 1]:
        # 学習後の推論: 最大重みを持つ行動を決定論的に選択
        best_action = max(weights[test_state], key=weights[test_state].get)
        expected_action = 1 if test_state == 0 else 2
        
        print(f"  State {test_state} Weights: {weights[test_state]}")
        print(f"  State {test_state} -> Chosen Action: {best_action} (Expected: {expected_action})")
        
        if best_action != expected_action:
            success = False
            
    if success:
        print("\n=> SUCCESS: The SNN successfully learned the optimal policy purely via delayed rewards!")
    else:
        print("\n=> FAILED: The policy has not converged yet. More epochs or tuning may be needed.")

if __name__ == "__main__":
    run_rl_benchmark()