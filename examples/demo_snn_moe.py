_FILE_INFO = {
    "//": "ディレクトリパス: examples/demo_snn_moe.py",
    "//": "ファイルの日本語タイトル: スパイキングMoE(大脳皮質カラム)のデモンストレーション",
    "//": "ファイルの目的や内容: 新規実装したSpikingCorticalColumnsが、入力パターンに応じて動的にエキスパートを使い分けるルーティング学習をテストする。疲労(ホメオスタシス)により一極集中が回避されることを確認する。"
}

import os
import random
import json
from sara_engine.core.cortical_columns import SpikingCorticalColumns

def main():
    print("[INFO] Starting Spiking Mixture of Experts (Cortical Columns) Demo...")
    
    workspace_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    log_file = os.path.join(workspace_dir, "moe_training_log.json")
    
    embed_dim = 128
    num_experts = 4
    top_k = 1
    
    # Initialize the MoE layer
    moe_layer = SpikingCorticalColumns(embed_dim=embed_dim, num_experts=num_experts, top_k=top_k)
    
    # Create distinct patterns to simulate different concepts (e.g., Math vs Language)
    pattern_A = random.sample(range(embed_dim), 10)
    pattern_B = random.sample(range(embed_dim), 10)
    
    logs = []
    
    print("\n--- Training Phase ---")
    for epoch in range(1, 51):
        # Forward pass for Pattern A
        out_A = moe_layer.forward(pattern_A, learning=True)
        
        # 状態の取得
        if moe_layer.use_rust:
            weights = moe_layer.router.get_weights()
            thresholds = moe_layer.router.get_thresholds()
        else:
            weights = moe_layer.router_weights
            thresholds = moe_layer.router_thresholds
            if isinstance(thresholds, dict):
                thresholds = [thresholds.get(i, 0.0) for i in range(num_experts)]
                
        # 閾値を引いた実効ポテンシャルの計算 (Pattern A)
        potentials_A = {i: -thresholds[i] for i in range(num_experts)}
        for s in pattern_A:
            for exp_id, w in weights[s].items():
                potentials_A[exp_id] += w
        winner_A = sorted(potentials_A.items(), key=lambda x: x[1], reverse=True)[0][0]
        
        # Forward pass for Pattern B
        out_B = moe_layer.forward(pattern_B, learning=True)
        
        # 状態の再取得 (Pattern Aの学習で状態が変わっているため)
        if moe_layer.use_rust:
            weights = moe_layer.router.get_weights()
            thresholds = moe_layer.router.get_thresholds()
        else:
            weights = moe_layer.router_weights
            thresholds = moe_layer.router_thresholds
            if isinstance(thresholds, dict):
                thresholds = [thresholds.get(i, 0.0) for i in range(num_experts)]

        # 閾値を引いた実効ポテンシャルの計算 (Pattern B)
        potentials_B = {i: -thresholds[i] for i in range(num_experts)}
        for s in pattern_B:
            for exp_id, w in weights[s].items():
                potentials_B[exp_id] += w
        winner_B = sorted(potentials_B.items(), key=lambda x: x[1], reverse=True)[0][0]
        
        if epoch % 10 == 0:
            log_entry = {
                "epoch": epoch,
                "winner_A": winner_A,
                "winner_B": winner_B,
                "adjusted_potentials_A": potentials_A,
                "adjusted_potentials_B": potentials_B,
                "fatigue_thresholds": thresholds
            }
            logs.append(log_entry)
            print(f"Epoch {epoch:02d}: Pattern A -> Expert {winner_A}, Pattern B -> Expert {winner_B}")

    # Save logs to workspace
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4)
        
    print(f"\n[SUCCESS] Demo completed. Logs saved to: {log_file}")
    if winner_A != winner_B:
        print("[RESULT] Competitive learning successfully separated the concepts into different experts!")
    else:
        print("[RESULT] Both concepts routed to the same expert. (Consider adjusting fatigue parameters)")

if __name__ == "__main__":
    main()