_FILE_INFO = {
    "//": "ディレクトリパス: tests/test_new_features.py",
    "//": "タイトル: 新機能（Attention, Temporal, GPT, STDP）の結合テスト",
    "//": "目的: 新規実装された4つのモジュールの基本動作と、行列演算・BPなしでの制約下での挙動を検証する。"
}

import unittest
import random
from typing import List

# 各モジュールのインポート
# （実際のプロジェクト構成に合わせてパスは調整してください）
from sara_engine.core.attention import SpikeAttention
from sara_engine.core.temporal import TemporalEncoder
from sara_engine.memory.sdr import SDREncoder
from sara_engine.models.gpt import SaraGPT
from sara_engine.learning.stdp import STDPPretrainer

class TestSaraNewFeatures(unittest.TestCase):
    
    def setUp(self):
        # 共通の設定
        self.input_size = 1024
        self.hidden_size = 2048
        self.density = 0.05
        self.active_bits = int(self.input_size * self.density)
        
        # テスト用のダミーSDRを生成
        random.seed(42)
        self.dummy_sdr_a = sorted(random.sample(range(self.input_size), self.active_bits))
        self.dummy_sdr_b = sorted(random.sample(range(self.input_size), self.active_bits))

    def test_spike_attention(self):
        print("\n--- Testing SpikeAttention ---")
        attention = SpikeAttention(input_size=self.input_size, hidden_size=self.hidden_size, num_heads=2)
        
        # 初期状態では記憶がないため空リストが返るはず
        res1 = attention.compute(self.dummy_sdr_a)
        self.assertEqual(res1, [])
        
        # 何度か計算（内部で記憶が蓄積される設計）
        attention.compute(self.dummy_sdr_a)
        attention.compute(self.dummy_sdr_b)
        
        # 記憶が溜まった後、再度Aでクエリをかける
        res_a = attention.compute(self.dummy_sdr_a)
        
        # 出力が生成され、かつスパース性（hidden_size * 0.05程度）が維持されているか確認
        self.assertTrue(len(res_a) > 0)
        self.assertTrue(len(res_a) <= max(1, int(self.hidden_size * 0.05)))
        
        print(f"Attention output size: {len(res_a)}")
        print("SpikeAttention test passed.")

    def test_temporal_coding(self):
        print("\n--- Testing TemporalEncoder ---")
        temporal = TemporalEncoder(input_size=self.input_size, target_density=self.density)
        
        # 同じSDRを異なる位置でエンコード
        encoded_pos0 = temporal.encode(self.dummy_sdr_a, position=0)
        encoded_pos1 = temporal.encode(self.dummy_sdr_a, position=1)
        encoded_pos2 = temporal.encode(self.dummy_sdr_a, position=2)
        
        # 1. 密度が維持されているか確認
        self.assertTrue(abs(len(encoded_pos0) - self.active_bits) <= 2)
        
        # 2. 位置が違えば異なるSDR（Overlapが小さい）になるか確認
        overlap_0_1 = len(set(encoded_pos0).intersection(set(encoded_pos1)))
        print(f"Overlap between pos 0 and pos 1: {overlap_0_1} / {self.active_bits}")
        self.assertTrue(overlap_0_1 < self.active_bits * 0.5) # 重なりは半分未満になるはず
        
        # 3. 同じSDR、同じ位置なら完全に一致するか確認
        encoded_pos0_again = temporal.encode(self.dummy_sdr_a, position=0)
        self.assertEqual(encoded_pos0, encoded_pos0_again)
        
        print("TemporalEncoder test passed.")

    def test_stdp_and_gpt_generation(self):
        print("\n--- Testing STDPPretrainer & SaraGPT ---")
        # エンコーダとGPTモデルの準備
        encoder = SDREncoder(input_size=self.input_size, density=self.density, use_tokenizer=True, apply_vsa=False)
        
        # テスト用の小さな語彙を学習
        corpus = [
            "私 は 猫 が 好き です",
            "犬 は 走る の が 速い",
            "猫 は かわいい です"
        ]
        encoder.tokenizer.train(corpus)
        gpt = SaraGPT(encoder)
        
        # STDPによる事前学習
        pretrainer = STDPPretrainer(window_size=3, a_plus=1.0, a_minus=0.2)
        
        # 初期状態ではシナプス結合は空
        self.assertEqual(len(gpt.synapses), 0)
        
        pretrainer.pretrain(gpt, corpus)
        
        # STDP学習後、シナプスが形成されているか確認
        self.assertTrue(len(gpt.synapses) > 0)
        print(f"Total active presynaptic neurons: {len(gpt.synapses)}")
        
        # 確率的デコーディングのテスト
        # 同じプロンプトでも、temperatureによって出力が変化（または少なくともエラーなく動く）ことを確認
        prompt = "私 は"
        
        # Temperature = 0.0 (Greedy)
        res_greedy = gpt.generate(prompt, temperature=0.0, max_tokens=5)
        print(f"Generation (Temp=0.0): {res_greedy}")
        self.assertTrue(len(res_greedy) > 0)
        
        # Temperature = 1.0 (Stochastic)
        res_stochastic = gpt.generate(prompt, temperature=1.0, max_tokens=5)
        print(f"Generation (Temp=1.0): {res_stochastic}")
        self.assertTrue(len(res_stochastic) > 0)
        
        print("STDP & GPT Generation test passed.")


if __name__ == '__main__':
    unittest.main()