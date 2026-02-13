# 状態を持つSNNの理論と実装

## 1. 問題の本質

### 現在のSNNの限界

```python
# 現在の実装（sara_gpt_core.py）
class SaraGPT:
    def forward_step(self, input_sdr):
        # 問題1: 状態が暗黙的
        # → スパイクパターンに埋め込まれている
        
        # 問題2: 短期記憶のみ
        # → self.prev_spikes = 1ステップ前だけ
        
        # 問題3: 文脈の喪失
        # → self.readout_v *= 0.85  # すぐに減衰
        
        return output_sdr
```

### 生物学的な脳との比較

| 機能 | 生物の脳 | 現在のSNN | 必要な改善 |
|------|---------|----------|-----------|
| **短期記憶** | 前頭前野（Working Memory） | prev_spikes (1step) | Working Memory層 |
| **状態管理** | 基底核（状態遷移） | なし | State Neurons |
| **長期記憶** | 海馬 | episodic_memory | 改善必要 |
| **注意機構** | 前頭葉 | Spike Attention | 強化必要 |

## 2. 解決策: 3つのアプローチ

### アプローチA: Working Memory層の追加 ⭐推奨

**コンセプト**: 前頭前野のWorking Memoryを模倣

```python
class WorkingMemory:
    # 特徴
    - リングバッファ: 過去10個のパターンを保持
    - 持続的活性化: ゆっくり減衰（decay=0.95）
    - アテンション: 重要な記憶を強調
    - 想起機能: クエリに類似した記憶を検索
```

**メリット**:
- ✅ 既存のSNN構造をほぼそのまま使える
- ✅ 生物学的に妥当
- ✅ 実装が比較的容易

**デメリット**:
- ❌ 完全な決定論的状態管理ではない
- ❌ 学習に時間がかかる可能性

### アプローチB: State Neuron Group ⭐⭐最も明確

**コンセプト**: 専用の状態表現ニューロン

```python
class StateNeuronGroup:
    state_names = ["INIT", "SEARCH", "READ", "EXTRACT", "DONE"]
    activations = [0, 0, 1, 0, 0]  # Winner-Take-All
    
    # 明示的な状態遷移
    INIT → SEARCH → READ → EXTRACT → DONE
```

**メリット**:
- ✅ 状態が完全に明示的
- ✅ デバッグが容易
- ✅ 決定論的な動作

**デメリット**:
- ❌ SNNの「分散表現」の利点を失う
- ❌ 状態遷移ルールを手動で設計

### アプローチC: Reservoir Computing + Readout States ⭐⭐⭐最もバランス良い

**コンセプト**: 状態ごとに異なるReadout重みを使用

```python
class StatefulSNN:
    # 各状態専用のReadout層
    readout_weights = {
        "SEARCH": W_search,  # 検索に特化
        "READ": W_read,      # 読み取りに特化
        "EXTRACT": W_extract # 抽出に特化
    }
    
    def forward(self, input, current_state):
        hidden = self.reservoir(input)
        output = self.readout_weights[current_state] @ hidden
        return output
```

**メリット**:
- ✅ SNNの分散表現を維持
- ✅ 状態が明確
- ✅ 学習が効率的（状態ごとに独立）

**デメリット**:
- ❌ メモリ使用量が増加（状態数×重み）

## 3. 推奨する統合設計

### 3層アーキテクチャ

```
┌─────────────────────────────────────┐
│  State Layer (State Neurons)        │ ← 明示的な状態管理
│  - INIT, SEARCH, READ, EXTRACT      │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Context Layer (Working Memory)     │ ← 文脈の保持
│  - Past 10 patterns                 │
│  - Attention mechanism              │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Processing Layer (Liquid SNN)      │ ← 認識・処理
│  - L1 (Fast), L2 (Med), L3 (Slow)  │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Output Layer (State-aware Readout) │ ← 状態依存の出力
└─────────────────────────────────────┘
```

### データフロー

```python
def forward_step(self, input_spikes, verbose=False):
    # 1. Working Memoryから文脈を取得
    context = self.working_memory.recall(input_spikes)
    
    # 2. 状態ニューロンを更新
    self.state_neurons.update(input_spikes, context)
    current_state = self.state_neurons.get_state()
    
    if verbose:
        print(f"Current State: {current_state}")
    
    # 3. SNN層で処理（文脈を含む）
    combined_input = input_spikes + context
    hidden_spikes = self.liquid_layers.process(combined_input)
    
    # 4. 状態に応じたReadout
    readout_weights = self.readout[current_state]
    output_spikes = self.compute_output(hidden_spikes, readout_weights)
    
    # 5. Working Memoryを更新
    self.working_memory.store(output_spikes, importance=1.0)
    
    # 6. 状態遷移の学習
    self.update_transition_rules(current_state, output_spikes)
    
    return output_spikes, current_state
```

## 4. 学習戦略

### 4.1 教師あり学習（初期段階）

```python
# 状態ラベル付きデータで学習
training_data = [
    {
        "input": "What is the code?",
        "states": ["INIT", "SEARCH", "READ", "EXTRACT"],
        "actions": ["START", "SEARCH code", "READ CHUNK", "EXTRACT"]
    }
]

# 各ステップで正解状態を強制
for step, (input, state, action) in enumerate(data):
    snn.state_neurons.set_state(state)  # 教師信号
    output = snn.forward(input)
    loss = compute_loss(output, action)
    snn.update_weights(loss)
```

### 4.2 強化学習（発展段階）

```python
# 報酬シグナルで状態遷移を学習
class RLStatefulSNN:
    def learn_from_experience(self, trajectory):
        # trajectory = [(state, action, reward), ...]
        
        for t, (state, action, reward) in enumerate(trajectory):
            # Q値の更新
            Q[state, action] += lr * (reward + gamma * max(Q[next_state]) - Q[state, action])
            
            # 状態遷移行列の更新
            self.state_neurons.transition_matrix[state, next_state] += lr * reward
```

### 4.3 自己組織化（最終段階）

```python
# 状態が自動的に出現
# - 類似した入力パターン → 同じ状態クラスタ
# - 異なる動作が必要 → 状態の分岐

# k-meansや階層的クラスタリング
states = cluster_hidden_patterns(all_hidden_states, n_clusters=5)
```

## 5. 実装の段階的アプローチ

### Phase 1: Working Memoryの追加（最も簡単）

```python
# 既存のsara_gpt_core.pyに追加
class SaraGPT:
    def __init__(self):
        # 既存のコード
        ...
        # 新規追加
        self.working_memory = WorkingMemory(capacity=10)
    
    def forward_step(self, input_sdr):
        # 1. 文脈を取得
        context = self.working_memory.get_context_spikes()
        
        # 2. 既存の処理（文脈を追加）
        combined_input = input_sdr + context
        hidden = self.l1.forward(combined_input, ...)
        
        # 3. Working Memoryを更新
        self.working_memory.store(hidden)
        
        return output
```

**期待される改善**:
- 過去の情報を参照できる
- ループが減る（前の検索結果を覚えている）

### Phase 2: State Neuronsの追加（中程度）

```python
class SaraGPT:
    def __init__(self):
        ...
        self.state_neurons = StateNeuronGroup(num_states=5)
    
    def forward_step(self, input_sdr):
        # 状態を更新
        self.state_neurons.update(input_sdr, context)
        current_state = self.state_neurons.get_state()
        
        # 状態をログ
        print(f"State: {current_state}")
        
        # 既存の処理
        ...
```

**期待される改善**:
- 現在の状態が明確
- デバッグが容易
- 状態遷移を可視化できる

### Phase 3: State-aware Readoutの追加（高度）

```python
class SaraGPT:
    def __init__(self):
        ...
        # 状態ごとのReadout
        self.readout_weights = {
            "SEARCH": np.random.randn(...),
            "READ": np.random.randn(...),
            "EXTRACT": np.random.randn(...)
        }
    
    def forward_step(self, input_sdr):
        current_state = self.state_neurons.get_state()
        
        # 状態に応じた重みを選択
        weights = self.readout_weights[current_state]
        output = weights @ hidden_spikes
        
        return output
```

**期待される改善**:
- 状態に特化した出力
- 学習効率の向上
- タスク専門化

## 6. 難しさとトレードオフ

### 技術的課題

| 課題 | 難易度 | 解決策 |
|-----|-------|--------|
| **状態の表現** | ★★☆ | State Neuronsで明示化 |
| **状態遷移の学習** | ★★★ | 教師あり→強化学習の段階的導入 |
| **メモリ管理** | ★★☆ | Working Memoryの容量制限 |
| **計算コスト** | ★★☆ | 状態ごとのReadoutは重いがバッチ処理可能 |

### 設計のトレードオフ

```
シンプル ←→ 高性能
    |         |
    |         └─ 3層アーキテクチャ（推奨）
    |              - 複雑だが推論可能
    |              - 学習に時間
    |
    └─ Working Memoryのみ
         - 実装が簡単
         - 効果は限定的
```

## 7. 結論

### SNNに状態を持たせることは可能か？

**答え: YES、ただし工夫が必要**

1. **Working Memory層**: 比較的簡単、効果あり
2. **State Neurons**: 中程度の難易度、大きな効果
3. **State-aware Readout**: 高度、最高の効果

### 推奨アプローチ

**段階的実装**:
```
Phase 1: Working Memory追加 → 2-3日
Phase 2: State Neurons追加 → 1週間
Phase 3: State-aware Readout → 2週間
Phase 4: 教師あり学習 → 1週間
Phase 5: ファインチューニング → 継続的
```

### core.pyの設計の問題か？

**Yes、改善の余地は大きい**:

- ❌ 現在: 状態が暗黙的、短期記憶のみ
- ✅ 改善後: 明示的な状態、長期文脈

しかし、**完全に決定論的な推論には限界がある**:
- SNNは確率的な性質を持つ
- 完璧な論理推論はState Machineに任せる
- SNNは「ヒント生成」「曖昧性解消」に特化

### 最終的なハイブリッド設計

```
┌──────────────────────────────┐
│  決定論的制御                │
│  (State Machine)             │ ← 確実な状態遷移
└──────┬───────────────────────┘
       │
┌──────▼───────────────────────┐
│  Stateful SNN                │
│  - Working Memory            │ ← 文脈理解
│  - State Neurons             │ ← 状態認識
│  - State-aware Readout       │ ← 適応的出力
└──────┬───────────────────────┘
       │
┌──────▼───────────────────────┐
│  Pattern Matching            │ ← テキスト処理
└──────────────────────────────┘
```

**この設計により、SNNの強みを活かしつつ、推論も可能になります。**
