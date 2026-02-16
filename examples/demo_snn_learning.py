# examples/demo_snn_learning.py
# スパイキングニューラルネットワーク学習デモ
# STDP (Spike-Timing-Dependent Plasticity) を用いた学習や、スパイキングアテンション機構など、SARAエンジンのSNNコア機能の動作を確認します。

import numpy as np
from sara_engine.core.cortex import Cortex
from sara_engine.core.spike_attention import SpikeAttention
from sara_engine.learning.stdp import STDP

def main():
    print("=== SNN Core 学習デモンストレーション ===")
    
    input_size = 64
    hidden_size = 128
    seq_length = 10
    
    # コンポーネントの初期化
    print("SNNコンポーネントを初期化中...")
    cortex_layer = Cortex(input_dim=input_size, hidden_dim=hidden_size)
    attention = SpikeAttention(dim=hidden_size, num_heads=4)
    stdp_learner = STDP(learning_rate=0.01)
    
    # ダミーのスパイク入力列（バッチサイズ1, シーケンス長, 特徴量次元）
    # 0か1のスパイクラスタ
    spike_inputs = np.random.choice([0, 1], size=(1, seq_length, input_size), p=[0.8, 0.2])
    
    print("\n[順伝播とアテンションの計算]")
    # Cortexレイヤーの処理
    cortex_outputs = []
    for t in range(seq_length):
        current_input = spike_inputs[:, t, :]
        out = cortex_layer.forward(current_input)
        cortex_outputs.append(out)
    
    cortex_outputs = np.stack(cortex_outputs, axis=1)
    print(f"Cortex出力形状: {cortex_outputs.shape}")
    
    # スパイキングアテンションの適用
    attended_features, attention_weights = attention.forward(cortex_outputs)
    print(f"Attention適用後の形状: {attended_features.shape}")
    
    print("\n[STDPによる重みの更新]")
    print("シナプス前スパイクとシナプス後スパイクのタイミングに基づきSTDP学習を行います...")
    
    # 疑似的なプレ・ポストスパイク列を使用してSTDPをテスト
    pre_spikes = np.random.choice([0, 1], size=(100, input_size), p=[0.9, 0.1])
    post_spikes = np.random.choice([0, 1], size=(100, hidden_size), p=[0.9, 0.1])
    
    initial_weights = cortex_layer.get_weights()
    updated_weights = stdp_learner.apply(initial_weights, pre_spikes, post_spikes)
    
    # 重みの変化量を計算
    weight_change = np.mean(np.abs(updated_weights - initial_weights))
    print(f"STDPによる平均重み変化量: {weight_change:.6f}")
    
    print("\nSNNデモが完了しました。")

if __name__ == "__main__":
    main()