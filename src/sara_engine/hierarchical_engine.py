# src/sara_engine/hierarchical_engine.py
# title: Hierarchical SARA Engine
# description: ロードマップ 1.2 階層的特徴学習の実装。下層から上層への情報伝達を行うDeep SNN。

import numpy as np
import pickle
from typing import List
from .stdp_layer import STDPLiquidLayer

class HierarchicalSaraEngine:
    """
    階層型SARAエンジン (Deep Liquid State Machine)
    構造: Input -> Layer1(Fast) -> Layer2(Medium) -> Layer3(Slow) -> Readout
    特徴: 下層の出力が上層の入力となる。
    """
    
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        
        # 階層構造の定義
        # Layer 1: Input -> Hidden (高速、高解像度)
        self.l1 = STDPLiquidLayer(input_size, 1500, decay=0.3, 
                                  input_scale=1.5, rec_scale=1.2, density=0.1)
        
        # Layer 2: L1 Output -> Hidden (中速、統合)
        # 入力サイズはL1のニューロン数と同じ
        self.l2 = STDPLiquidLayer(1500, 1500, decay=0.6, 
                                  input_scale=1.0, rec_scale=1.5, density=0.08)
        
        # Layer 3: L2 Output -> Hidden (低速、文脈)
        self.l3 = STDPLiquidLayer(1500, 1000, decay=0.9, 
                                  input_scale=0.8, rec_scale=1.8, density=0.08)
        
        self.layers = [self.l1, self.l2, self.l3]
        
        # Readoutは全層からSkip Connectionで受け取る
        self.total_hidden = 1500 + 1500 + 1000
        self.offsets = [0, 1500, 3000]
        
        # Readout Weights
        self.w_ho: List[np.ndarray] = []
        for _ in range(output_size):
            w = np.random.normal(0, 0.05, self.total_hidden).astype(np.float32)
            self.w_ho.append(w)
            
        self.o_v = np.zeros(output_size, dtype=np.float32)
        
        # State
        # 修正: 型ヒント追加
        self.prev_spikes: List[List[int]] = [[], [], []]
        self.lr = 0.002

    def reset_state(self):
        for layer in self.layers:
            layer.reset()
        self.prev_spikes = [[], [], []]
        self.o_v.fill(0)

    def forward_hierarchical(self, input_spikes: List[int], learning: bool = False) -> List[int]:
        """階層的なフォワードパス"""
        
        # 1. Layer 1 (Input -> L1)
        spikes1 = self.l1.forward(input_spikes, self.prev_spikes[0], learning=learning)
        
        # 2. Layer 2 (L1 -> L2)
        # L1のスパイクをL2への入力として扱う
        spikes2 = self.l2.forward(spikes1, self.prev_spikes[1], learning=learning)
        
        # 3. Layer 3 (L2 -> L3)
        spikes3 = self.l3.forward(spikes2, self.prev_spikes[2], learning=learning)
        
        self.prev_spikes = [spikes1, spikes2, spikes3]
        
        # 全層のスパイクを結合して返す (Skip Connection用)
        all_spikes = []
        all_spikes.extend(spikes1)
        all_spikes.extend([x + self.offsets[1] for x in spikes2])
        all_spikes.extend([x + self.offsets[2] for x in spikes3])
        
        return all_spikes

    def train_step(self, spike_train: List[List[int]], target_label: int):
        self.reset_state()
        
        for input_spikes in spike_train:
            # 教師あり学習中も、下層では教師なしSTDPを弱く働かせることが可能（Hybrid）
            # ここではシンプルにするためSTDP=Falseとする
            all_spikes = self.forward_hierarchical(input_spikes, learning=False)
            
            # Readout Update (Delta Rule)
            self.o_v *= 0.9
            if all_spikes:
                for o in range(self.output_size):
                    self.o_v[o] += np.sum(self.w_ho[o][all_spikes]) * 0.1
            
            # Simple Online Delta Learning
            # ターゲットに近づける、他を遠ざける
            prediction = self.o_v
            error = np.zeros(self.output_size)
            
            if prediction[target_label] < 1.0:
                error[target_label] = 1.0 - prediction[target_label]
            
            for o in range(self.output_size):
                if o != target_label and prediction[o] > 0.0:
                    error[o] = 0.0 - prediction[o]
            
            # Weight Update
            if all_spikes:
                for o in range(self.output_size):
                    if abs(error[o]) > 0.01:
                        self.w_ho[o][all_spikes] += self.lr * error[o]

    def predict(self, spike_train: List[List[int]]) -> int:
        self.reset_state()
        potentials = np.zeros(self.output_size)
        
        for input_spikes in spike_train:
            all_spikes = self.forward_hierarchical(input_spikes, learning=False)
            
            potentials *= 0.9
            if all_spikes:
                for o in range(self.output_size):
                    potentials[o] += np.sum(self.w_ho[o][all_spikes]) * 0.1
                    
        return int(np.argmax(potentials))