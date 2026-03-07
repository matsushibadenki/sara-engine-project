 **SNNベースAIアーキテクチャ設計ドキュメント**
 
 
前提条件：

* **誤差逆伝播法（Backpropagation）を使用しない**
* **行列演算を使用しない**
* **イベント駆動（スパイク）計算**
* **局所学習則のみ**
* **生物学的整合性を重視**

---

# SNN認知アーキテクチャ設計ドキュメント

**（LSM + STP + STDP + Global Workspace + Dopamine RL + World Model）**

---

# 1 設計目的

本アーキテクチャは以下を目的とする。

* Transformerに依存しないAI
* 時系列理解能力
* 自己組織化学習
* 長期・短期記憶の統合
* 意思決定能力

従来AIとの比較

| 項目 | 従来NN   | 本設計      |
| -- | ------ | -------- |
| 学習 | 誤差逆伝播  | 局所可塑性    |
| 計算 | 行列演算   | イベント駆動   |
| 時間 | 離散ステップ | 連続ダイナミクス |
| 構造 | 静的     | 自己組織化    |

---

# 2 全体アーキテクチャ

```
Sensory Input
      ↓
Spike Encoder
      ↓
Liquid State Machine (Reservoir)
      ↓
Assembly Detection
      ↓
Global Workspace
      ↓
Action Selection
      ↓
Environment
      ↓
Reward Signal
      ↓
Dopamine RL
      ↓
STDP Update
```

内部ループ

```
Perception → Prediction → Decision → Action → Learning
```

---

# 3 ニューロンモデル

基本ニューロンは **LIF (Leaky Integrate and Fire)**。

状態変数

```
V = membrane potential
```

更新

```
dV/dt = -(V - Vrest)/τ + I_syn
```

発火条件

```
if V ≥ Vthreshold:
    spike
    V = Vreset
```

特徴

* 時間ダイナミクス
* 非線形イベント生成
* 局所計算

---

# 4 シナプスモデル

シナプスは以下の要素を持つ。

```
weight
delay
STP state
eligibility trace
```

シナプス電流

```
I_syn = weight * spike_pre
```

---

# 5 STP（短期シナプス可塑性）

STPは短期記憶を実装する。

状態

```
u = facilitation
x = depression
```

更新

```
u ← u + U*(1-u)
x ← x*(1-u)
```

時間回復

```
du/dt = (U-u)/τf
dx/dt = (1-x)/τd
```

有効シナプス

```
effective_weight = w * u * x
```

役割

* 短期記憶
* 文脈保持
* 時系列依存

---

# 6 LSM（Liquid State Machine）

Reservoirはランダム接続スパイクネットワーク。

構造

```
input neurons
reservoir neurons
```

特徴

* 再帰接続
* 非線形ダイナミクス
* 高次元状態

状態更新

```
V_i(t+dt) =
V_i(t)
+ Σ synaptic input
+ leakage
```

Reservoirは

```
high dimensional temporal embedding
```

を生成する。

---

# 7 Assembly（神経集団）

複数ニューロンの同期発火が

```
assembly
```

を形成する。

例

```
cluster A → object
cluster B → motion
cluster C → context
```

Assemblyは

```
candidate thoughts
```

となる。

---

# 8 Global Workspace

Global Workspaceは

```
winner-take-all network
```

である。

入力

```
assembly activation
```

更新

```
A_i(t+1) =
A_i(t)
+ excitation_i
- global inhibition
```

競争

```
winner = argmax(A_i)
```

選ばれたassemblyは

```
broadcast
```

される。

役割

* 注意
* 意識
* 思考の選択

---

# 9 Action Selection

Workspaceの出力は

```
action neurons
```

に入力される。

行動選択

```
argmax(action_activity)
```

行動例

```
move
speak
look
store memory
```

---

# 10 Dopamine RL

行動後に報酬が与えられる。

報酬予測誤差

```
δ = r + γV(s') − V(s)
```

ここで

```
δ = dopamine signal
```

となる。

---

# 11 Eligibility Trace

シナプスは履歴を保持する。

```
e(t+1) = e(t)*λ + spike_pre*spike_post
```

重み更新

```
Δw = η * δ * e
```

これにより

```
delayed reward learning
```

が可能。

---

# 12 STDP（長期可塑性）

スパイク時間差により学習。

```
Δt = t_post - t_pre
```

更新

```
Δw =
A+ exp(-Δt/τ+)  if Δt>0
-A- exp(Δt/τ-)  if Δt<0
```

ドーパミン変調

```
Δw = δ * STDP
```

---

# 13 World Model

世界モデルは状態遷移を学習する。

```
state(t+1) = f(state(t), action)
```

内部状態

```
reservoir activity
```

予測

```
future state simulation
```

---

# 14 Predictive Coding

脳型推論モデル。

予測誤差

```
error = observation - prediction
```

更新

```
state ← state + learning_rate * error
```

---

# 15 Planning（内部シミュレーション）

行動前に未来を予測。

```
simulate(action)
predict(state)
evaluate(reward)
```

最適行動

```
argmax expected reward
```

---

# 16 記憶階層

| 記憶   | 実装           |
| ---- | ------------ |
| 瞬間記憶 | スパイク         |
| 短期記憶 | STP          |
| 中期記憶 | LSM dynamics |
| 長期記憶 | STDP         |

---

# 17 計算量

本設計は

```
local computation
```

のみ。

計算量

```
O(number_of_synapses)
```

Transformer

```
O(N²)
```

より低い。

---

# 18 実装ポリシー

重要ルール

* 行列演算禁止
* 各ニューロン独立更新
* 各シナプス局所更新
* イベント駆動処理

データ構造例

```
Neuron
Synapse
SpikeEvent
Assembly
Workspace
```

---

# 19 最終認知ループ

```
1 perception
2 reservoir dynamics
3 assembly formation
4 workspace competition
5 action selection
6 environment interaction
7 reward
8 dopamine
9 synaptic plasticity
```

---

# 20 この設計の特徴

強み

* 時系列理解
* エネルギー効率
* 自己組織化
* 長期記憶

弱点

* 学習収束が遅い
* 実装難易度高い
* パラメータ設計が重要

---

# 21 将来拡張

追加可能な機構

```
dendritic neuron model
hierarchical reservoirs
spiking predictive coding
neural fields
```

---

# 22 結論

このアーキテクチャは

```
SNN
+
dynamical systems
+
reinforcement learning
+
cognitive architecture
```

を統合した

**非Transformer型AI設計**

である。

特徴

* 生物学的整合性
* O(N)計算
* 自己組織化学習
* 世界モデル推論

---
