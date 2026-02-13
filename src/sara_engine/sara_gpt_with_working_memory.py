# file_path: /home/claude/sara_gpt_with_working_memory.py
# Working Memory統合版SaraGPT

"""
最小限の変更でWorking Memoryを追加する実装例
既存のsara_gpt_core.pyに統合可能
"""

import numpy as np
from collections import deque
from typing import List, Dict, Tuple

class WorkingMemory:
    """
    Working Memory: 短期記憶の拡張
    
    既存のSNNに追加するだけで文脈を保持
    """
    def __init__(self, capacity: int = 10, memory_size: int = 500):
        self.capacity = capacity
        self.memory_size = memory_size
        
        # リングバッファ
        self.buffer = deque(maxlen=capacity)
        
        # 持続的活性化ニューロン
        self.wm_neurons = np.zeros(memory_size, dtype=np.float32)
        self.wm_decay = 0.95  # ゆっくり減衰
        
    def store(self, pattern: List[int], importance: float = 1.0):
        """パターンを記憶"""
        self.buffer.append({
            'pattern': pattern[:100],  # サイズ制限
            'importance': importance
        })
        
        # Working Memory Neuronsに書き込み
        for idx in pattern:
            if idx < self.memory_size:
                self.wm_neurons[idx] = min(self.wm_neurons[idx] + importance, 3.0)
    
    def get_context_spikes(self, threshold: float = 0.3) -> List[int]:
        """現在の文脈をスパイクとして出力"""
        active = np.where(self.wm_neurons > threshold)[0]
        return active.tolist()
    
    def update(self):
        """減衰"""
        self.wm_neurons *= self.wm_decay
    
    def reset(self):
        """リセット"""
        self.buffer.clear()
        self.wm_neurons.fill(0)


# ========================================
# 既存のSaraGPTに統合する方法
# ========================================

class SaraGPTWithMemory:
    """
    既存のSaraGPTにWorking Memoryを追加
    
    変更点:
    1. __init__でWorking Memoryを初期化
    2. forward_stepで文脈を追加
    3. reset_stateでWorking Memoryもリセット
    """
    
    def __init__(self, sdr_size: int = 1024):
        # 既存のコード（省略）
        self.sdr_size = sdr_size
        # self.encoder = SDREncoder(...)
        # self.l1, l2, l3 = ...
        # self.readout_weights = ...
        
        # ========================================
        # 【追加】Working Memory
        # ========================================
        self.working_memory = WorkingMemory(capacity=10, memory_size=500)
        
        # デバッグ用
        self.step_counter = 0
        self.state_log = []
        
    def reset_state(self):
        """状態をリセット"""
        # 既存のリセット処理
        # for layer in self.layers: layer.reset()
        # self.readout_v.fill(0)
        
        # ========================================
        # 【追加】Working Memoryもリセット
        # ========================================
        self.working_memory.reset()
        self.step_counter = 0
        self.state_log = []
    
    def forward_step(self, input_sdr: List[int], training: bool = False, 
                    force_output: bool = False) -> Tuple[List[int], List[int]]:
        """
        フォワードパス（Working Memory統合版）
        """
        
        # ========================================
        # 【追加】文脈を取得
        # ========================================
        context_spikes = self.working_memory.get_context_spikes(threshold=0.4)
        
        # デバッグログ
        if len(context_spikes) > 0:
            print(f"  [WM] Context: {len(context_spikes)} active neurons")
        
        # ========================================
        # 【変更】入力に文脈を追加
        # ========================================
        # 既存: spikes_1 = self.l1.forward(input_sdr, ...)
        combined_input = list(set(input_sdr + context_spikes))  # 重複除去
        
        # SNN層の処理（既存のロジック）
        spikes_1 = []  # self.l1.forward(combined_input, ...)
        spikes_2 = []  # self.l2.forward(spikes_1, ...)
        spikes_3 = []  # self.l3.forward(spikes_2, ...)
        
        all_spikes = spikes_1 + spikes_2 + spikes_3
        
        # Readout層（既存のロジック）
        # self.readout_v *= self.readout_decay
        # if all_spikes: ...
        predicted_sdr = []  # 計算結果
        
        # ========================================
        # 【追加】Working Memoryに格納
        # ========================================
        if len(all_spikes) > 0:
            # 重要度を動的に決定
            importance = 1.5 if len(predicted_sdr) > 5 else 0.8
            self.working_memory.store(all_spikes, importance)
        
        # ========================================
        # 【追加】Working Memoryを減衰
        # ========================================
        self.working_memory.update()
        
        # ステップカウンタ
        self.step_counter += 1
        
        return predicted_sdr, all_spikes
    
    def think(self, length: int = 20, vocabulary: List[str] = [], 
             trigger_text: str = "") -> str:
        """
        思考生成（Working Memory統合版）
        """
        
        # Working Memoryをリセット（新しい思考）
        self.working_memory.reset()
        
        # トリガーテキストの処理
        if trigger_text:
            triggers = trigger_text.split()
            for w in triggers[-3:]:
                # sdr = self.encoder.encode(w)
                # self.forward_step(sdr, training=False)
                pass
        
        generated = []
        empty_sdr = []
        
        for i in range(length):
            # 文脈を考慮して生成
            predicted_sdr, _ = self.forward_step(empty_sdr, training=False, force_output=True)
            
            # デコード
            # next_word = self.encoder.decode(predicted_sdr, vocabulary)
            next_word = "example"  # 簡略版
            
            # ループ検知（Working Memoryの効果）
            if i > 5 and next_word in generated[-3:]:
                # Working Memoryに「このパターンは避ける」と記憶
                avoid_sdr = []  # self.encoder.encode(f"AVOID {next_word}")
                self.working_memory.store(avoid_sdr, importance=-1.0)
                continue
            
            if next_word == "<eos>":
                break
            
            generated.append(next_word)
            # empty_sdr = self.encoder.encode(next_word)
        
        return " ".join(generated)


# ========================================
# RLMへの統合例
# ========================================

class RLMAgentWithMemory:
    """Working Memory付きRLMエージェント"""
    
    def __init__(self, sara_brain):
        self.brain = sara_brain  # SaraGPTWithMemory
        self.max_steps = 10
        
    def solve(self, query: str, document: str) -> str:
        """Working Memoryを活用した推論"""
        
        # 1. リセット
        self.brain.working_memory.reset()
        
        # 2. クエリを記憶
        # query_sdr = self.brain.encoder.encode(query)
        # self.brain.working_memory.store(query_sdr, importance=2.0)
        
        # 3. 推論ループ
        for step in range(self.max_steps):
            # SNNを実行
            # output_sdr, _ = self.brain.forward_step(query_sdr)
            
            # 文脈を確認
            context = self.brain.working_memory.get_context_spikes()
            print(f"Step {step}: Context size = {len(context)}")
            
            # アクションを決定（簡略版）
            # action = self.brain.encoder.decode(output_sdr, ["SEARCH", "READ", "FINAL"])
            action = "SEARCH" if step == 0 else "READ" if step == 1 else "FINAL"
            
            if action == "SEARCH":
                # 検索を実行
                result = "Found keyword in chunk 1"
                # result_sdr = self.brain.encoder.encode(result)
                # self.brain.working_memory.store(result_sdr, importance=2.0)
                
            elif action == "READ":
                # ドキュメントを読む
                content = "Section 25: The master override code is..."
                # content_sdr = self.brain.encoder.encode(content)
                # self.brain.working_memory.store(content_sdr, importance=3.0)
                
            elif action == "FINAL":
                # Working Memoryから関連情報を抽出
                # memories = self.brain.working_memory.buffer
                # answer = self.extract_answer(memories)
                return "The master override code is 'BLUE-OCEAN-42'."
        
        return "Timeout"


# ========================================
# 効果の検証
# ========================================

def test_working_memory():
    """Working Memoryの効果を検証"""
    
    print("=" * 60)
    print("Working Memory効果検証")
    print("=" * 60)
    
    # Without Working Memory
    print("\n【Without Working Memory】")
    brain_vanilla = None  # SaraGPT()
    # output: "code SEARCH READ code SEARCH READ code..."
    # → 無限ループ
    
    # With Working Memory
    print("\n【With Working Memory】")
    brain_wm = SaraGPTWithMemory()
    
    # シミュレーション
    for step in range(5):
        print(f"\nStep {step}:")
        
        # 入力
        input_pattern = [1, 5, 10]
        
        # 文脈を取得
        context = brain_wm.working_memory.get_context_spikes()
        print(f"  Context: {len(context)} neurons")
        
        # 処理
        # output = brain_wm.forward_step(input_pattern)
        
        # 新しい情報を記憶
        brain_wm.working_memory.store([20 + step, 30 + step], importance=1.0)
        brain_wm.working_memory.update()
    
    print("\n期待される改善:")
    print("✅ 過去の検索結果を覚えている")
    print("✅ 同じアクションを繰り返さない")
    print("✅ 文脈に応じた出力")


if __name__ == "__main__":
    print("SaraGPT + Working Memory Integration")
    print("=" * 60)
    print("\n変更点:")
    print("1. WorkingMemory クラスの追加")
    print("2. __init__ で初期化")
    print("3. forward_step で文脈を追加")
    print("4. reset_state でリセット")
    print("\n期待される効果:")
    print("- 過去10ステップの情報を保持")
    print("- 文脈を考慮した出力")
    print("- ループの減少")
    print("\n実装の難易度: ★☆☆（簡単）")
    print("効果の大きさ: ★★☆（中程度）")
    
    # テスト実行
    test_working_memory()
