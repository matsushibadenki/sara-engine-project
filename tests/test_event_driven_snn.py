# パス: tests/test_event_driven_snn.py
# タイトル: イベント駆動・スパースSNNアーキテクチャの動作検証
# 目的: 廃止されたget_state()やforward_with_feedback()を新しいAPIに置き換え、テストを修正する。
# {
#     "//": "試行錯誤のプロセスを検証するため、シードを固定して実行します。"
# }

import unittest
import random
from src.sara_engine.learning.stdp import STDPLayer
from src.sara_engine.core.layers import DynamicLiquidLayer

class TestEventDrivenSNN(unittest.TestCase):
    def setUp(self):
        random.seed(42)

    def test_stdp_pruning_and_scaling(self):
        """R-STDPによる学習、シナプス・スケーリング、不要結合の刈り込み(Pruning)をテスト"""
        num_inputs = 20
        num_outputs = 3
        layer = STDPLayer(num_inputs, num_outputs, threshold=1.0)
        
        pattern = [1 if i in (0, 2, 4) else 0 for i in range(num_inputs)]
        
        for _ in range(100):
            layer.process_step(pattern, reward=1.5)
            
        for j in range(num_outputs):
            current_synapses = layer.synapses[j]
            current_sum = sum(current_synapses.values())
            
            if current_sum > 0:
                self.assertLessEqual(current_sum, layer.target_weight_sum + 0.1, 
                                     f"スケーリング上限違反: ニューロン {j} の重み総和が {current_sum}")
                
            zero_input_keys = [k for k in current_synapses.keys() if k not in (0, 2, 4)]
            
            if current_sum >= layer.target_weight_sum - 0.5:
                self.assertTrue(len(zero_input_keys) < 10, 
                                f"刈り込み失敗: 不要なシナプスが多数残存しています。残存数: {len(zero_input_keys)}")

    def test_dynamic_liquid_layer_sparse_forward(self):
        """Numpy非依存の辞書型ネットワークにおけるフォワードパスとホメオスタシスの検証"""
        layer = DynamicLiquidLayer(
            input_size=10, 
            hidden_size=20, 
            decay=0.9, 
            density=0.3, 
            use_rust=False
        )
        
        active_inputs = [1, 5, 8]
        prev_active_hidden = [2, 10]
        
        # 修正: get_state() ではなく直接プロパティにアクセス
        init_thresh = list(layer.dynamic_thresh)
        self.assertEqual(init_thresh[0], 1.0)
        
        for _ in range(15):
            # 修正: forward_with_feedback を forward に変更
            fired = layer.forward(
                active_inputs=active_inputs,
                prev_active_hidden=prev_active_hidden,
                learning=True
            )
            prev_active_hidden = fired
            
        # 修正: get_state() ではなく直接プロパティにアクセス
        v_state = layer.v
        thresh_state = layer.dynamic_thresh
        
        self.assertIsInstance(v_state, list)
        self.assertIsInstance(thresh_state, list)
        self.assertEqual(len(v_state), 20)
        
        self.assertTrue(any(t != 1.0 for t in thresh_state), "ホメオスタシスが機能しておらず、閾値が固定されています")

if __name__ == '__main__':
    unittest.main()