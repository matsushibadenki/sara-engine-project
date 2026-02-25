_FILE_INFO = {
    "//": "ディレクトリパス: examples/benchmark_rl_stdp.py",
    "//": "ファイルの日本語タイトル: 報酬変調STDPベンチマーク",
    "//": "ファイルの目的や内容: 誤差逆伝播法を排除し、遅延報酬のみ（正解ラベルなし）でSNNが正しい行動を選択できるようになるかテストするスクリプト。"
}

import random
from sara_engine.nn.rstdp import RewardModulatedLinearSpike

def run_rl_benchmark():
    print("=== SARA Engine: Reward-Modulated STDP (R-STDP) Benchmark ===\n")
    
    # 状態数2 (State 0, State 1) -> 行動数3 (Action 0, Action 1, Action 2)
    layer = RewardModulatedLinearSpike(in_features=2, out_features=3, density=1.0)
    
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
    print("[*] Training with R-STDP (No BP, no target labels)...")
    
    for epoch in range(epochs):
        state = random.choice([0, 1])
        
        # SNNによる行動選択 (推論とトレースの記録)
        out_spikes = layer([state], learning=True)
        
        # 探索フェーズ（発火が弱い場合はランダムな行動をとる）
        if not out_spikes or random.random() < 0.2:
            action = random.choice([0, 1, 2])
            # 手動でランダム探索の痕跡（トレース）を残す
            layer.eligibility_traces[(state, action)] = layer.eligibility_traces.get((state, action), 0.0) + 1.0
        else:
            action = out_spikes[0] # 最初に発火した（ポテンシャルが最大の）行動を選択
            
        # 環境からのフィードバック
        reward = get_reward(state, action)
        
        # 大域的なドーパミン（報酬）シグナルによるシナプス更新
        layer.apply_reward(reward, learning_rate=0.2)
        
    print("\n[*] Evaluation Phase (Learning Disabled)")
    success = True
    
    for test_state in [0, 1]:
        out_spikes = layer([test_state], learning=False)
        action = out_spikes[0] if out_spikes else -1
        
        expected_action = 1 if test_state == 0 else 2
        print(f"  State {test_state} -> Chosen Action: {action} (Expected: {expected_action})")
        
        if action != expected_action:
            success = False
            
    if success:
        print("\n=> SUCCESS: The SNN successfully learned the optimal policy purely via delayed rewards!")
    else:
        print("\n=> FAILED: The policy has not converged yet. More epochs or tuning may be needed.")

if __name__ == "__main__":
    run_rl_benchmark()