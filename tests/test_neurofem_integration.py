# SARA Engine: NeuroFEM Integration Test
_FILE_INFO = {
    "//": "ディレクトリパス: tests/test_neurofem_integration.py",
    "//": "タイトル: NeuroFEMとCortex(大脳皮質)の統合テスト",
    "//": "目的: NeuroFEMで計算した物理的な熱拡散状態をエンコーダーでSDRに変換し、Cortexカラムに入力して発火連鎖(認識)が正常に行われるかを検証する。"
}

import sys
import os
import numpy as np

# 実際のディレクトリ構造に合わせてパスを追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sara_engine.core.neurofem import NeuroFEMLayer
from sara_engine.encoders.physical import GridStateEncoder
from sara_engine.core.cortex import CorticalColumn

def test_neurofem_to_cortex():
    print("--- NeuroFEM -> Encoder -> Cortex 統合テスト開始 ---")
    
    # 1. NeuroFEMのセットアップ (3x3の小規模グリッド)
    grid_size = 3
    num_nodes = grid_size * grid_size
    fem_layer = NeuroFEMLayer(num_nodes=num_nodes, threshold=1.0, decay=0.90)
    
    weight = 0.4
    for y in range(grid_size):
        for x in range(grid_size):
            node_id = y * grid_size + x
            if x < grid_size - 1:
                fem_layer.add_connection(node_id, node_id + 1, weight)
            if y < grid_size - 1:
                fem_layer.add_connection(node_id, node_id + grid_size, weight)
                
    # 中央(ノード4)を加熱
    fem_layer.set_boundary_condition(4, 2.0)
    
    # 行列演算なしで熱拡散をシミュレート
    for _ in range(100):
        fem_layer.forward_step([])
        
    steady_state = fem_layer.get_steady_state()
    print(f"1. NeuroFEM 最終状態 (3x3):\n  {[round(v, 2) for v in steady_state]}")
    
    # 2. 物理SDRエンコーダーのセットアップ
    node_sdr_size = 64
    encoder = GridStateEncoder(
        num_nodes=num_nodes, 
        node_sdr_size=node_sdr_size, 
        min_val=0.0, 
        max_val=20.0, 
        active_bits=4
    )
    
    encoded_sdr = encoder.encode_grid(steady_state)
    print(f"\n2. Encoder 変換結果 (SDR発火インデックスの一部):\n  {encoded_sdr[:12]} ... (計 {len(encoded_sdr)} 発火)")
    
    # 3. Cortex(大脳皮質カラム)のセットアップ
    total_input_size = num_nodes * node_sdr_size
    cortex = CorticalColumn(
        input_size=total_input_size,
        hidden_size_per_comp=1024,
        compartment_names=["language", "physics_sim"],
        target_rate=0.05
    )
    
    # 4. 物理コンテキストでのCortex処理
    fired_neurons = cortex.forward_latent_chain(
        active_inputs=encoded_sdr,
        prev_active_hidden=[],
        current_context="physics_sim",
        learning=True,
        reward_signal=1.0
    )
    
    print(f"\n3. Cortex 処理結果:")
    print(f"  発火した皮質ニューロン数: {len(fired_neurons)}")
    
    states = cortex.get_compartment_states()
    print("\n各コンパートメントの活性状態 (膜電位 v > 0 の残存ニューロン):")
    for comp_name, state in states.items():
        print(f"  - {comp_name}: {state['active_neurons']} 個")
        
    # 検証：SNN特有の発火後リセットを考慮し、fired_neuronsの数で発火を判定する。
    # 同時に、language コンテキストが完全に遮断され、膜電位の変動がない(0である)ことを確認する。
    if len(fired_neurons) > 0 and states["language"]["active_neurons"] == 0:
        print("\n✅ テスト成功: 物理状態のSDRがCortexで正常に処理され、かつ他のコンパートメント(言語層)への干渉(破滅的忘却)も完全に防げています！")
    else:
        print("\n❌ テスト失敗: 予期せぬコンパートメントが発火しているか、発火が起きていません。")

if __name__ == "__main__":
    test_neurofem_to_cortex()
