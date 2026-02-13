# file_path: /home/claude/stateful_snn_design.py
# 状態を持つSNN - 設計案

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque

class WorkingMemory:
    """
    Working Memory: SNNの短期記憶を拡張
    
    生物学的な前頭前野のWorking Memoryを模倣
    """
    def __init__(self, capacity: int = 10, memory_size: int = 500):
        self.capacity = capacity
        self.memory_size = memory_size
        
        # リングバッファ（過去N個のパターンを保持）
        self.buffer = deque(maxlen=capacity)
        
        # 持続的活性化ニューロン（Working Memory Neurons）
        self.wm_neurons = np.zeros(memory_size, dtype=np.float32)
        self.wm_decay = 0.95  # ゆっくり減衰（長期保持）
        
        # アテンション重み（どの記憶が重要か）
        self.attention_weights = np.ones(capacity) / capacity
        
    def store(self, pattern: List[int], importance: float = 1.0):
        """パターンを記憶に格納"""
        self.buffer.append({
            'pattern': pattern,
            'importance': importance,
            'timestamp': len(self.buffer)
        })
        
        # Working Memory Neuronsに書き込み
        for idx in pattern:
            if idx < self.memory_size:
                self.wm_neurons[idx] += importance
        
    def recall(self, query_pattern: List[int], top_k: int = 3) -> List[Dict]:
        """クエリに類似した記憶を想起"""
        if not self.buffer:
            return []
        
        query_set = set(query_pattern)
        similarities = []
        
        for item in self.buffer:
            pattern_set = set(item['pattern'])
            # Jaccard類似度
            intersection = len(query_set & pattern_set)
            union = len(query_set | pattern_set)
            similarity = intersection / union if union > 0 else 0
            
            similarities.append({
                'pattern': item['pattern'],
                'similarity': similarity,
                'importance': item['importance']
            })
        
        # 類似度でソート
        similarities.sort(key=lambda x: x['similarity'] * x['importance'], reverse=True)
        return similarities[:top_k]
    
    def get_context_spikes(self) -> List[int]:
        """現在のWorking Memory状態をスパイクとして出力"""
        active_threshold = 0.3
        active_neurons = np.where(self.wm_neurons > active_threshold)[0]
        return active_neurons.tolist()
    
    def update(self):
        """時間経過による減衰"""
        self.wm_neurons *= self.wm_decay
        self.wm_neurons = np.clip(self.wm_neurons, 0, 5.0)
    
    def reset(self):
        """記憶をクリア"""
        self.buffer.clear()
        self.wm_neurons.fill(0)


class StateNeuronGroup:
    """
    State Neuron Group: 明示的な状態表現
    
    各ニューロンが特定の状態を表現
    Winner-Take-All機構で排他的な状態遷移
    """
    def __init__(self, num_states: int = 5):
        self.num_states = num_states
        self.state_names = ["INIT", "SEARCH", "READ", "EXTRACT", "DONE"]
        
        # 各状態の活性度
        self.activations = np.zeros(num_states, dtype=np.float32)
        
        # 状態遷移ルール（学習可能）
        self.transition_matrix = np.random.rand(num_states, num_states) * 0.1
        
        # 現在の状態（Winner-Take-All）
        self.current_state = 0  # INIT
        
    def update(self, input_spikes: List[int], context_spikes: List[int]):
        """入力とコンテキストから状態を更新"""
        
        # 1. 入力の影響
        input_strength = len(input_spikes) / 100.0
        
        # 2. コンテキストの影響
        context_strength = len(context_spikes) / 100.0
        
        # 3. 状態遷移の計算
        current_activation = self.activations[self.current_state]
        
        # 遷移確率を計算
        transition_probs = self.transition_matrix[self.current_state]
        transition_probs += np.random.randn(self.num_states) * 0.05  # ノイズ
        
        # 4. Winner-Take-All
        next_state = np.argmax(transition_probs)
        
        # 5. 状態を更新
        self.activations.fill(0)
        self.activations[next_state] = 1.0
        self.current_state = next_state
        
    def get_state(self) -> str:
        """現在の状態名を取得"""
        return self.state_names[self.current_state]
    
    def set_state(self, state_name: str):
        """状態を強制的に設定（学習用）"""
        if state_name in self.state_names:
            idx = self.state_names.index(state_name)
            self.activations.fill(0)
            self.activations[idx] = 1.0
            self.current_state = idx
    
    def get_state_spikes(self) -> List[int]:
        """状態をスパイクパターンとして出力"""
        # 状態ニューロンのインデックスを返す
        return [self.current_state] if self.activations[self.current_state] > 0.5 else []


class StatefulSNN:
    """
    Stateful SNN: Working MemoryとState Neuronsを統合したSNN
    """
    def __init__(self, input_size: int = 1024, output_size: int = 1024):
        self.input_size = input_size
        self.output_size = output_size
        
        # 既存のSNN層（core.pyのLiquidLayerを使用）
        # self.liquid_layers = [...]
        
        # 新規コンポーネント
        self.working_memory = WorkingMemory(capacity=10, memory_size=500)
        self.state_neurons = StateNeuronGroup(num_states=5)
        
        # State-aware Readout（状態に応じた出力）
        self.state_readout_weights = {}
        for state_name in self.state_neurons.state_names:
            self.state_readout_weights[state_name] = np.random.randn(
                output_size, 6000  # total_hidden + wm_size + state_size
            ).astype(np.float32) * 0.05
    
    def forward_with_state(self, input_spikes: List[int], 
                           store_in_memory: bool = True) -> Tuple[List[int], str]:
        """
        状態を考慮したフォワードパス
        
        Returns:
            (output_spikes, current_state)
        """
        
        # 1. Working Memoryから文脈を取得
        context_spikes = self.working_memory.get_context_spikes()
        
        # 2. 状態ニューロンを更新
        self.state_neurons.update(input_spikes, context_spikes)
        current_state = self.state_neurons.get_state()
        state_spikes = self.state_neurons.get_state_spikes()
        
        # 3. SNN層の処理（既存のロジック + 文脈）
        # all_spikes = self.process_liquid_layers(input_spikes + context_spikes)
        all_spikes = input_spikes + context_spikes  # 簡略化
        
        # 4. State-aware Readout
        # 現在の状態に応じた重みを使用
        readout_weights = self.state_readout_weights[current_state]
        
        # 出力の計算（簡略版）
        output_potential = np.zeros(self.output_size)
        combined_spikes = all_spikes + state_spikes
        
        for spike_idx in combined_spikes:
            if spike_idx < readout_weights.shape[1]:
                output_potential += readout_weights[:, spike_idx]
        
        # Top-k スパイクを出力
        top_k = 20
        output_spikes = np.argsort(output_potential)[-top_k:].tolist()
        
        # 5. Working Memoryに格納
        if store_in_memory:
            importance = 1.0 if current_state in ["SEARCH", "READ"] else 0.5
            self.working_memory.store(output_spikes, importance)
        
        # 6. Working Memoryを減衰
        self.working_memory.update()
        
        return output_spikes, current_state
    
    def train_with_state(self, input_sequence: List[List[int]], 
                        state_sequence: List[str],
                        target_sequence: List[List[int]]):
        """
        状態ラベル付きで学習
        
        Args:
            input_sequence: 入力スパイク列
            state_sequence: 各ステップの正解状態
            target_sequence: 出力の正解スパイク列
        """
        
        for t, (input_spikes, target_state, target_spikes) in enumerate(
            zip(input_sequence, state_sequence, target_sequence)
        ):
            # 状態を強制設定（教師信号）
            self.state_neurons.set_state(target_state)
            
            # フォワードパス
            output_spikes, current_state = self.forward_with_state(input_spikes)
            
            # 誤差計算
            target_set = set(target_spikes)
            output_set = set(output_spikes)
            
            # 正解に含まれるスパイクを強化
            correct_spikes = list(target_set & output_set)
            incorrect_spikes = list(output_set - target_set)
            
            # 重みの更新（簡略版）
            lr = 0.01
            readout_weights = self.state_readout_weights[target_state]
            
            if correct_spikes:
                for out_idx in correct_spikes:
                    for in_idx in input_spikes:
                        if in_idx < readout_weights.shape[1]:
                            readout_weights[out_idx, in_idx] += lr
            
            if incorrect_spikes:
                for out_idx in incorrect_spikes:
                    for in_idx in input_spikes:
                        if in_idx < readout_weights.shape[1]:
                            readout_weights[out_idx, in_idx] -= lr * 0.5


# ========================================
# 使用例: RLMに統合
# ========================================

class StatefulRLMAgent:
    """Stateful SNNを使用したRLMエージェント"""
    
    def __init__(self):
        self.snn = StatefulSNN(input_size=1024, output_size=1024)
        self.encoder = None  # SDREncoder
        
    def solve(self, query: str, document: str) -> str:
        """状態を持つSNNで推論"""
        
        # 1. 初期化
        self.snn.working_memory.reset()
        self.snn.state_neurons.set_state("INIT")
        
        # 2. クエリをエンコード
        query_spikes = self.encoder.encode(query)
        
        # 3. 推論ループ
        max_steps = 10
        for step in range(max_steps):
            # SNNを実行
            output_spikes, current_state = self.snn.forward_with_state(query_spikes)
            
            print(f"Step {step}: State = {current_state}")
            
            # 状態に応じた処理
            if current_state == "SEARCH":
                # 検索を実行
                keyword = self.encoder.decode(output_spikes, ["code", "master", "dream"])
                result = f"Found keyword: {keyword}"
                
                # 結果をWorking Memoryに格納
                result_spikes = self.encoder.encode(result)
                self.snn.working_memory.store(result_spikes, importance=2.0)
                
            elif current_state == "READ":
                # ドキュメントを読む
                pass
                
            elif current_state == "EXTRACT":
                # 答えを抽出
                answer = self.encoder.decode(output_spikes, vocabulary)
                return answer
                
            elif current_state == "DONE":
                break
        
        return "Timeout"


# ========================================
# 学習例
# ========================================

def train_stateful_rlm():
    """状態付きRLMの学習"""
    
    snn = StatefulSNN()
    encoder = None  # SDREncoder
    
    # 教師データ
    training_data = [
        {
            'query': "What is the code?",
            'states': ["INIT", "SEARCH", "READ", "EXTRACT", "DONE"],
            'actions': [
                encoder.encode("START"),
                encoder.encode("SEARCH code"),
                encoder.encode("READ CHUNK 1"),
                encoder.encode("EXTRACT answer"),
                encoder.encode("END")
            ]
        }
    ]
    
    # 学習
    for example in training_data:
        query_spikes = encoder.encode(example['query'])
        
        input_sequence = [query_spikes] * len(example['states'])
        state_sequence = example['states']
        target_sequence = example['actions']
        
        snn.train_with_state(input_sequence, state_sequence, target_sequence)


if __name__ == "__main__":
    print("Stateful SNN Design")
    print("=" * 60)
    print("\n1. Working Memory: 過去の情報を保持")
    print("2. State Neurons: 明示的な状態管理")
    print("3. State-aware Readout: 状態に応じた出力")
    print("\nこの設計により、SNNでも推論が可能になります。")
