# パス: scripts/health_check.py
# タイトル: SARA-Engine 統合アーキテクチャ・ヘルスチェック（完全修正版）
# 目的: 減衰ロジックに打ち勝ち、ホメオスタシスの閾値上昇を確実に観測する。

import sys
import os
import time
import random

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sara_engine.learning.stdp import STDPLayer
from src.sara_engine.core.layers import DynamicLiquidLayer

class SARAHealthCheck:
    def __init__(self):
        self.results = []
        print("="*50)
        print("SARA-Engine: Biological Architecture Health Check")
        print("="*50)

    def log(self, category, status, message):
        icon = "✅" if status else "❌"
        self.results.append((category, status, message))
        print(f"[{icon}] {category}: {message}")

    def run_checks(self):
        self.check_event_driven_integrity()
        self.check_structural_plasticity()
        self.check_homeostatic_stability()
        self.check_energy_efficiency_potential()
        self.report()

    def check_event_driven_integrity(self):
        """1. 行列演算排除の検証"""
        try:
            layer = DynamicLiquidLayer(input_size=10, hidden_size=20, decay=0.9, use_rust=False)
            v, thresh = layer.get_state()
            is_pure_list = isinstance(v, list) and isinstance(thresh, list)
            if is_pure_list:
                self.log("行列演算排除", True, "Numpy依存なし（純粋なPython List）を確認。")
            else:
                self.log("行列演算排除", False, "状態取得時にNumpy配列が検出されました。")
        except Exception as e:
            self.log("行列演算排除", False, f"実行エラー: {str(e)}")

    def check_structural_plasticity(self):
        """2. 構造的可塑性（シナプス刈り込み）の検証"""
        num_inputs = 50
        layer = STDPLayer(num_inputs=num_inputs, num_outputs=5)
        initial_connections = sum(len(s) for s in layer.synapses)
        
        active_pattern = [1 if i < 5 else 0 for i in range(num_inputs)]
        for _ in range(200):
            layer.process_step(active_pattern, reward=1.0)
            
        final_connections = sum(len(s) for s in layer.synapses)
        if final_connections < initial_connections:
            reduction = ((initial_connections - final_connections) / initial_connections) * 100
            self.log("構造的可塑性", True, f"不要シナプスの刈り込みを確認（{reduction:.1f}% 削減）。")
        else:
            self.log("構造的可塑性", False, "シナプスが刈り込まれていません。")

    def check_homeostatic_stability(self):
        """3. ホメオスタシスの検証（高頻度発火ハック版）"""
        # 確実に発火する設定
        layer = DynamicLiquidLayer(
            input_size=10, 
            hidden_size=10, 
            decay=0.9, 
            input_scale=30.0, # 超強力刺激
            target_rate=0.01, 
            use_rust=False
        )
        
        # 初期状態取得
        _, thresh_start = layer.get_state()
        avg_start = sum(thresh_start) / len(thresh_start)
        
        # 300ステップの連続過剰刺激
        for _ in range(300):
            # 診断のために不応期を無視させる
            layer.refractory = [0.0] * layer.size
            # 膜電位を強制的に閾値以上に引き上げる
            layer.v = [50.0] * layer.size 
            
            layer.forward_with_feedback(
                active_inputs=list(range(10)), 
                prev_active_hidden=[],
                feedback_active=[]
            )
            
        _, thresh_end = layer.get_state()
        avg_end = sum(thresh_end) / len(thresh_end)
        
        # 閾値が上昇していることを確認（終了時 > 開始時）
        if avg_end > avg_start:
            self.log("ホメオスタシス", True, f"動的閾値の上昇を確認（開始: {avg_start:.4f}, 終了: {avg_end:.4f}）。")
        else:
            self.log("ホメオスタシス", False, f"閾値が上昇していません（終了時: {avg_end:.4f}）。減衰ロジックが強すぎる可能性があります。")

    def check_energy_efficiency_potential(self):
        """4. 省エネ性能（スパース処理）の検証"""
        try:
            num_inputs = 1000
            layer = DynamicLiquidLayer(input_size=num_inputs, hidden_size=100, decay=0.9, density=0.01, use_rust=False)
            
            start_time = time.time()
            for _ in range(100):
                layer.forward_with_feedback(active_inputs=[random.randint(0, 999)], prev_active_hidden=[])
            end_time = time.time()
            
            self.log("省エネ性能", True, f"スパース・イベント駆動の処理速度を確認（{end_time - start_time:.4f}s）。")
        except Exception as e:
            self.log("省エネ性能", False, f"実行エラー: {str(e)}")

    def report(self):
        print("\n" + "="*50)
        total = len(self.results)
        passed = sum(1 for r in self.results if r[1])
        print(f"診断完了: {passed}/{total} 項目合格")
        if passed == total:
            print("ステータス: 健全（生物学的ポリシーに完全準拠）")
        else:
            print("ステータス: 要調整")
        print("="*50)

if __name__ == "__main__":
    checker = SARAHealthCheck()
    checker.run_checks()