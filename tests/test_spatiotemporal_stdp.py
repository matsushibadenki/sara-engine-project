# [配置するディレクトリのパス]: ./tests/test_spatiotemporal_stdp.py
# [ファイルの日本語タイトル]: STDPモデル用 1次元熱伝導ダミーデータ生成および動作検証スクリプト
# [ファイルの目的や内容]: 
# 外部ライブラリ（NumPyなど）の行列演算を一切使わず、純粋なPythonリスト操作のみで
# 1次元の熱伝導方程式（差分法）をシミュレーションし、動的な時系列データを作成する。
# そのデータをSpatioTemporalSNNに流し込み、STDPによるリカレント結合の重み更新が
# 無限に発散せず（恒常性が保たれ）、正常に機能しているかをコンソール出力で検証する。

import math
import sys
import os

# srcディレクトリにパスを通す（実行環境に合わせて調整してください）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.snn_models.spatiotemporal_stdp import SpatioTemporalSNN

def simulate_1d_heat_diffusion(n_nodes, steps):
    """
    行列演算を使わず、純粋なリストで熱分布の時系列データを生成する。
    中央付近を揺れ動く熱源を作り、それが周囲へ拡散する様子をシミュレートする。
    """
    history = []
    T = [0.0 for _ in range(n_nodes)]
    alpha = 0.25 # 拡散係数（安定条件: alpha <= 0.5）
    
    for t in range(steps):
        T_new = [0.0 for _ in range(n_nodes)]
        
        # サイン波で左右に揺れる熱源（ダイナミクス）を生成
        center = int(n_nodes / 2) + int(math.sin(t * 0.05) * (n_nodes / 4))
        # 範囲外エラーを防ぐクリッピング
        center = max(1, min(n_nodes - 2, center))
        T[center] = 1.0 # 熱源を最大温度に設定
        
        # 1次元熱伝導方程式の陽解法（リストのインデックス操作のみ）
        for i in range(1, n_nodes - 1):
            T_new[i] = T[i] + alpha * (T[i-1] - 2.0 * T[i] + T[i+1])
            
        # 境界条件（両端は固定または断熱、ここでは0に近い値として処理）
        T_new[0] = T_new[1]
        T_new[n_nodes-1] = T_new[n_nodes-2]
            
        # 全体が熱で飽和しないよう、毎ステップ微小に自然冷却（減衰）させる
        for i in range(n_nodes):
            T_new[i] *= 0.98
            # NeuroFEMからの入力仕様に合わせ、0.0〜1.0の範囲に収める
            if T_new[i] > 1.0:
                T_new[i] = 1.0
            elif T_new[i] < 0.0:
                T_new[i] = 0.0
            
        history.append(T_new)
        T = T_new
        
    return history

def main():
    n_nodes = 201
    simulation_steps = 500
    
    print("1. 熱伝導ダミー時系列データの生成中...")
    heat_history = simulate_1d_heat_diffusion(n_nodes, simulation_steps)
    
    print("2. SpatioTemporalSNNの初期化...")
    snn = SpatioTemporalSNN(n_in=n_nodes, n_sensory=50, n_cortex=20)
    
    print("3. シミュレーション開始（STDPの学習推移モニタリング）")
    print("-" * 70)
    print(f"{'Step':>5} | {'Cortex Spikes':>15} | {'Sensory->Ctx W':>18} | {'Ctx->Ctx W (Recurrent)':>22}")
    print("-" * 70)
    
    for step, heat_data in enumerate(heat_history):
        # ネットワークの時間を1ステップ進める
        snn.step(heat_data)
        
        # 50ステップごとに内部状態を可視化
        if (step + 1) % 50 == 0:
            # --- 重みの平均値を計算して発散を確認 ---
            
            # 1. Sensory -> Cortex層の重み平均
            w_sens_ctx_sum = 0.0
            count_sens_ctx = 0
            for i in range(snn.n_sensory):
                for j in range(snn.n_cortex):
                    w_sens_ctx_sum += snn.W_sens_ctx[i][j]
                    count_sens_ctx += 1
            avg_w_sens_ctx = w_sens_ctx_sum / count_sens_ctx
            
            # 2. Cortex層内のリカレント重み平均
            w_ctx_ctx_sum = 0.0
            count_ctx_ctx = 0
            for i in range(snn.n_cortex):
                for j in range(snn.n_cortex):
                    if i != j: # 自己結合は除く
                        w_ctx_ctx_sum += snn.W_ctx_ctx[i][j]
                        count_ctx_ctx += 1
            avg_w_ctx_ctx = w_ctx_ctx_sum / max(1, count_ctx_ctx)
            
            # 3. Cortex層の現在ステップでの総発火数
            cortex_spikes = sum(1 for spike in snn.spike_cortex if spike)
            
            print(f"{step + 1:5d} | {cortex_spikes:15d} | {avg_w_sens_ctx:18.4f} | {avg_w_ctx_ctx:22.4f}")

    print("-" * 70)
    print("シミュレーション完了。重み (W) が無限大に発散せず、特定の範囲で安定していれば成功です。")

if __name__ == "__main__":
    main()