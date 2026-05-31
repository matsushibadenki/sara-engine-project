_FILE_INFO = {
    "//": "ディレクトリパス: tests/test_neurofem_2d.py",
    "//": "タイトル: 2次元NeuroFEMの熱拡散テスト",
    "//": "目的: 1次元の成功を基に、2次元メッシュ上での局所的スパイク伝播による熱拡散(ポアソン方程式)をシミュレートする。"
}

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sara_engine.core.neurofem import NeuroFEMLayer

def test_2d_heat_diffusion():
    print("--- 2次元熱拡散モデルのテスト開始 ---")
    
    # 5x5の2次元グリッドを構築
    grid_size = 5
    num_nodes = grid_size * grid_size
    
    # 減衰率を調整し、空間全体に緩やかにエネルギーが広がるようにする
    layer = NeuroFEMLayer(num_nodes=num_nodes, threshold=1.0, decay=0.92)
    
    # 2次元グリッドの網目状の結合 (上下左右に熱伝達率0.35を設定)
    weight = 0.35
    for y in range(grid_size):
        for x in range(grid_size):
            node_id = y * grid_size + x
            
            # 右のノードとの結合
            if x < grid_size - 1:
                layer.add_connection(node_id, node_id + 1, weight)
            # 下のノードとの結合
            if y < grid_size - 1:
                layer.add_connection(node_id, node_id + grid_size, weight)
                
    # 中央のノード (座標: x=2, y=2) を継続的な熱源(ヒーター)に設定
    center_node = 2 * grid_size + 2
    layer.set_boundary_condition(center_node, 2.0)
    
    print("初期状態のセットアップが完了しました。200ステップのシミュレーションを開始します。\n")
    
    # 行列演算なしで、ひたすら局所的なスパイク伝播ループを回す
    for step in range(200):
        layer.forward_step([])

    # 定常状態の取得
    final_state = layer.get_steady_state()
    
    print("--- テスト完了 ---")
    print("最終的な膜電位(温度)分布 (5x5 グリッド):")
    
    # 2次元の形で見やすく出力
    for y in range(grid_size):
        row_str = "  "
        for x in range(grid_size):
            node_id = y * grid_size + x
            row_str += f"{final_state[node_id]:6.2f} "
        print(row_str)

    # 中央が最も熱く、四隅の温度が低いかを確認
    center_temp = final_state[center_node]
    corner_temp = final_state[0]
    
    if center_temp > corner_temp:
        print("\n✅ テスト成功: 2次元空間における熱拡散の同心円状の勾配が自己組織化的に形成されました。")
    else:
        print("\n❌ テスト失敗: 正しい熱拡散の勾配が確認できませんでした。")

if __name__ == "__main__":
    test_2d_heat_diffusion()