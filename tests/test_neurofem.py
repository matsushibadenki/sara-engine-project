_FILE_INFO = {
    "//": "ディレクトリパス: tests/test_neurofem.py",
    "//": "タイトル: NeuroFEMレイヤーの動作確認テスト",
    "//": "目的: 誤差逆伝播法や行列演算を用いずに、SNNのスパイク伝播だけで1次元熱伝導方程式の近似解（線形な温度勾配）が自己組織化的に得られることを検証する。"
}

import sys
import os

# 実際のディレクトリ構造に合わせてパスを追加し、モジュールを読み込めるようにします
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sara_engine.core.neurofem import NeuroFEMLayer

def test_1d_heat_conduction():
    print("--- 1次元熱伝導モデルのテスト開始 ---")
    
    # 5つのノードを持つ1次元の棒をシミュレート
    num_nodes = 5
    
    # decayを0.95程度に設定し、スパイクが系全体に行き渡りながら平衡に向かうようにします
    layer = NeuroFEMLayer(num_nodes=num_nodes, threshold=1.0, decay=0.95)
    
    # ノード間を直列に結合（隣のノードへ熱が伝わるための剛性行列の代わり）
    # 熱伝達率(weight)として0.8を設定
    weight = 0.8 
    for i in range(num_nodes - 1):
        layer.add_connection(i, i + 1, weight)
        
    # 境界条件（ディリクレ条件）の設定
    # 左端(ノード0)を高温(常にエネルギーが供給される)
    # 右端(ノード4)を低温(常にエネルギーが奪われる)
    layer.set_boundary_condition(0, 1.0)
    layer.set_boundary_condition(4, -1.0)
    
    print("初期状態のセットアップが完了しました。シミュレーションを開始します。\n")
    
    # シミュレーションループ
    # 行列演算の連立方程式ソルバを回す代わりに、局所的なスパイク伝播ステップを繰り返します
    steps = 150
    for step in range(steps):
        # 外部からの動的スパイク入力はなし（境界条件の定常バイアスだけで駆動させる）
        fired = layer.forward_step([])
        
        # 15ステップごとに状態を出力して、波及していく様子を観察します
        if (step + 1) % 15 == 0:
            state = layer.get_steady_state()
            formatted_state = [f"{v:6.2f}" for v in state]
            print(f"Step {step + 1:3d}: 膜電位(温度)分布 = [{', '.join(formatted_state)}], 今回の発火数 = {len(fired)}")

    # 最終状態の確認
    final_state = layer.get_steady_state()
    print("\n--- テスト完了 ---")
    print("最終的な膜電位(温度)分布:")
    for i, v in enumerate(final_state):
        print(f"  ノード {i}: {v:.2f}")

    # 定常状態では、熱源から冷却端に向けて電位が勾配になって分布しているはずです
    if final_state[0] > final_state[1] > final_state[2] > final_state[3] > final_state[4]:
        print("\n✅ テスト成功: スパイクの伝播により、正常に熱伝導の勾配が形成されました。")
    else:
        print("\n❌ テスト失敗: 期待される温度勾配が形成されませんでした。パラメータを調整してください。")

if __name__ == "__main__":
    test_1d_heat_conduction()