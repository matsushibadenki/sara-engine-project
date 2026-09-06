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

# 23 検証付き概念凝縮ループ

大量の文章や経験は、それだけで知能を保証しない。一方、同じ構造が異なる表現・状況・時刻・観測者をまたいで現れると、既存の記憶単位では説明しにくい共通構造を切り出せる可能性がある。本設計では、この状態を次の段階に分ける。

```text
未分知
  = 概念名を与えず保持した、出典付きの経験・関係・予測残差

概念圧
  = 局所的な再利用と未説明の予測誤差または記述費用が、
    新しい共有構造の候補を要求する状態

意味凝縮
  = 複数の独立文脈に再利用される匿名構造を仮生成する処理

意味発芽
  = 検証済み構造を使うことで、未知例の予測または構成が改善する現象

語彙地平線
  = 検証済み匿名構造へ、人間が扱える名前と定義を結び付ける境界

知能複利
  = 検証済み構造を次層の入力として再利用し、
    新たな候補形成を可能にする有界な循環
```

処理順序は次の通りとする。

```text
source-aware episodes
  -> sparse typed fragments
  -> bounded local reuse
  -> anonymous concept candidate
  -> held-out prediction and compression checks
  -> contrast, counterexample, revision, and ablation checks
  -> verified structural factor
  -> optional multilingual lexical binding
  -> capped provenance-linked replay
```

匿名候補は、少なくとも次の情報を持つ。

```text
candidate_id
invariant_signature
concrete_bindings
support_evidence_refs
counterexample_refs
independence_groups
prediction_contract
description_cost_before_after
ablation_effect
revision_and_expiry
state_event_cpu_cost
status
```

新しい名称は候補の発見や評価には使わない。構造が先、名前は後とする。英語・日本語・简体中文の名称は同じ匿名IDへ結び付ける表示層であり、別々の概念や独立証拠として数えない。

同じ資料から自動生成した大量の言い換えは、表現の不変性を学ぶ材料にはなるが、独立した根拠にはならない。出典・生成系・改訂系列を一つのevidence lineageとして数え、異なる文脈、異なる反例、異なる将来予測で有用性を確認する。

全コーパスの無制限な再読込は行わない。候補の局所署名と出典索引から固定件数の関連記憶だけをreplayし、候補が生成した説明文を候補自身の追加証拠にしない。概念候補数、生成速度、階層深さ、接続数、replay数、寿命、状態bytes、event costを事前に制限する。

昇格には、独立文脈での再利用、全費用込みの圧縮改善、未知例の予測または構成改善、反例と希少例の保持、対象構造を除去したときの性能低下、決定的replay、固定資源上限をすべて要求する。短い説明や新奇な名前だけでは概念と認めない。

日本語: 名前を作る前に、匿名構造が未知例に役立つことを確認する。

English: Validate an anonymous reusable structure on unseen cases before naming it.

简体中文: 在命名之前，先验证匿名可复用结构能否改善未见样本的表现。

実装はR0/R1の局所学習実証後に行い、Phase 39の匿名再利用、Phase 40の動的検証、Phase 41の構造因子化へ順に接続する。既存の凍結済みPhase 39プロトコルは変更しない。

---
