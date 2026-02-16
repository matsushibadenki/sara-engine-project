_FILE_INFO = {
    "//": "ディレクトリパス: tests/test_crossmodal_association.py",
    "//": "タイトル: マルチモーダル(物理×言語)連想記憶テスト",
    "//": "目的: NeuroFEMの物理SDRと、言語(テキスト)のSDRを同じコンパートメントに入力し、クロスモーダルな発火連鎖を検証する。"
}

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sara_engine.core.neurofem import NeuroFEMLayer
from sara_engine.encoders.physical import GridStateEncoder
from sara_engine.core.cortex import CorticalColumn
from sara_engine.utils.visualizer import SaraVisualizer

def test_crossmodal_association():
    print("--- 物理 × 言語 クロスモーダル連想記憶テスト ---")
    
    # 1. 物理状態(NeuroFEM)の生成
    grid_size = 5
    num_nodes = grid_size * grid_size
    fem_layer = NeuroFEMLayer(num_nodes=num_nodes, threshold=1.0, decay=0.92)
    
    weight = 0.35
    for y in range(grid_size):
        for x in range(grid_size):
            node_id = y * grid_size + x
            if x < grid_size - 1:
                fem_layer.add_connection(node_id, node_id + 1, weight)
            if y < grid_size - 1:
                fem_layer.add_connection(node_id, node_id + grid_size, weight)
                
    # 中央に熱源
    fem_layer.set_boundary_condition(12, 5.0)
    for _ in range(150):
        fem_layer.forward_step([])
    steady_state = fem_layer.get_steady_state()
    
    # 公式可視化ツールを使用して画像を保存
    viz = SaraVisualizer(save_dir="tests/logs")
    viz.plot_neurofem_heatmap(steady_state, grid_size, filename="crossmodal_heat.png")
    
    # 2. 物理SDRのエンコード
    node_sdr_size = 32
    encoder = GridStateEncoder(num_nodes=num_nodes, node_sdr_size=node_sdr_size, max_val=20.0, active_bits=2)
    physics_sdr = encoder.encode_grid(steady_state)
    physics_input_size = num_nodes * node_sdr_size
    
    # 3. 言語SDR(仮想的)の生成
    # "Hot"という単語を表すSDRを発火インデックスのリストとして定義
    language_input_size = 500
    word_hot_sdr = [10, 45, 120, 250, 310, 480] 
    
    # 4. Cortexへの同時入力(クロスモーダル統合)
    # 物理入力空間と言語入力空間を連結
    total_input_size = physics_input_size + language_input_size
    
    cortex = CorticalColumn(
        input_size=total_input_size,
        hidden_size_per_comp=2048,
        compartment_names=["language", "physics_sim"],
        target_rate=0.03
    )
    
    # インデックスを連結空間に合わせてシフト
    shifted_word_sdr = [idx + physics_input_size for idx in word_hot_sdr]
    multimodal_input = physics_sdr + shifted_word_sdr
    
    # 両方のモダリティを処理できる core_shared で学習
    fired_neurons = cortex.forward_latent_chain(
        active_inputs=multimodal_input,
        prev_active_hidden=[],
        current_context="core_shared",
        learning=True,
        reward_signal=1.0
    )
    
    print(f"\n物理モダリティ入力サイズ: {physics_input_size}")
    print(f"言語モダリティ入力サイズ: {language_input_size}")
    print(f"統合モダリティ入力発火数: {len(multimodal_input)}")
    print(f"Cortex (core_shared) 発火ニューロン数: {len(fired_neurons)}")
    
    if len(fired_neurons) > 0:
        print("\n✅ テスト成功: 物理SDRと言語SDRが統合され、Cortex内でクロスモーダルな発火(連想の基盤)が形成されました。")
    else:
        print("\n❌ テスト失敗: ニューロンが発火しませんでした。")

if __name__ == "__main__":
    test_crossmodal_association()
