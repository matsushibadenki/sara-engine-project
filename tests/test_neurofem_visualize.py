_FILE_INFO = {
    "//": "ディレクトリパス: tests/test_neurofem_visualize.py",
    "//": "タイトル: NeuroFEM 2D熱拡散の可視化テスト",
    "//": "目的: NeuroFEMの計算結果をMatplotlibを用いてヒートマップ画像として出力し、ロードマップの『可視化ツールの拡充』に貢献する。出力先ディレクトリが存在しない場合の自動作成処理を追加。"
}

import sys
import os
import numpy as np

# 実行環境によってはmatplotlibのバックエンド設定が必要な場合があります
import matplotlib
matplotlib.use('Agg') # 画面出力せずにファイル保存するための設定
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sara_engine.core.neurofem import NeuroFEMLayer

def test_visualize_2d_heat():
    print("--- NeuroFEM 2D 可視化テスト開始 ---")
    
    grid_size = 10 # 10x10の少し大きめのグリッド
    num_nodes = grid_size * grid_size
    layer = NeuroFEMLayer(num_nodes=num_nodes, threshold=1.0, decay=0.95)
    
    weight = 0.4
    for y in range(grid_size):
        for x in range(grid_size):
            node_id = y * grid_size + x
            if x < grid_size - 1:
                layer.add_connection(node_id, node_id + 1, weight)
            if y < grid_size - 1:
                layer.add_connection(node_id, node_id + grid_size, weight)
                
    # 中央付近(ノード44と45)を熱源に設定
    layer.set_boundary_condition(44, 5.0)
    layer.set_boundary_condition(45, 5.0)
    
    print(f"{grid_size}x{grid_size}のグリッドでシミュレーションを実行中...")
    
    for _ in range(300):
        layer.forward_step([])

    steady_state = layer.get_steady_state()
    
    # 1次元配列を2次元(10x10)のNumPy配列に変換
    grid_data = np.array(steady_state).reshape((grid_size, grid_size))
    
    print("シミュレーション完了。画像を生成します...")
    
    # Matplotlibでヒートマップを描画
    plt.figure(figsize=(8, 6))
    plt.imshow(grid_data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Membrane Potential (Temperature Equivalent)')
    plt.title('NeuroFEM 2D Heat Diffusion')
    
    # 画像として保存 (testsディレクトリ内)
    output_path = os.path.join(os.path.dirname(__file__), 'workspace/neurofem_heatmap.png')
    
    # 【追加】ディレクトリが存在しない場合は作成する
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path)
    plt.close()
    
    print(f"✅ テスト成功: ヒートマップ画像を保存しました -> {output_path}")

if __name__ == "__main__":
    test_visualize_2d_heat()